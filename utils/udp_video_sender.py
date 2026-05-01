"""
udp_video_sender.py  ——  无人机端图传发送模块
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
依赖：
    pip install opencv-python numpy

快速接入：
    from udp_video_sender import VideoSender

    sender = VideoSender(dest_ip="10.147.36.61")
    sender.start()

    # 在主循环中：
    sender.send_frame(color_frame)   # color_frame 是 BGR ndarray (960x540)

    sender.stop()
"""

import cv2
import socket
import struct
import threading
import queue
import time
import logging

# ─── 协议常量（与接收端必须完全一致）────────────────────────────────────────
# 包头格式：网络字节序（大端）
#   frame_id    : uint32  帧序号（0 ~ 2^32-1 循环）
#   frag_id     : uint16  当前分片编号（0-based）
#   total_frags : uint16  该帧总分片数
#   data_len    : uint16  本分片有效数据字节数
HEADER_FMT   = "!IHHH"
HEADER_SIZE  = struct.calcsize(HEADER_FMT)   # 10 字节

# 每个 UDP 分片携带的最大数据量（字节）
# 8192 字节：在路由器/WiFi 下可能触发 IP 分片，但本地网络丢包率极低，
# 且分片数少（约 4~8 片/帧），协议开销小，延迟低。
# 若发现丢帧增多，可调小至 1400（不触发 IP 分片）。
FRAG_DATA_SIZE = 8192

DEFAULT_PORT         = 5600
DEFAULT_JPEG_QUALITY = 60   # 0~100；60 对 960×540 画面质量已足够清晰
DEFAULT_SEND_BUF     = 4 * 1024 * 1024   # 4 MB Socket 发送缓冲区
QUEUE_MAXSIZE        = 2    # 发送队列深度；始终保持最新帧，不堆积旧帧

logging.basicConfig(level=logging.INFO, format="[Sender] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class VideoSender:
    """
    将 BGR ndarray 图像通过 UDP 发送到地面站。
    采用「后台线程 + 丢帧队列」策略：捕获速度 > 发送速度时自动丢弃旧帧，
    确保地面站始终接收到最新画面，不产生积压延迟。
    """

    def __init__(
        self,
        dest_ip: str,
        dest_port: int = DEFAULT_PORT,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        frag_size: int = FRAG_DATA_SIZE,
    ):
        self.dest_ip       = dest_ip
        self.dest_port     = dest_port
        self.jpeg_quality  = jpeg_quality
        self.frag_size     = frag_size

        self._frame_id  = 0
        self._queue     = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._running   = False
        self._thread    = None

        # 建立 UDP Socket 并扩大发送缓冲区
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, DEFAULT_SEND_BUF)

        # （可选）若已知目标固定，connect 后 sendto 开销略低
        self._sock.connect((self.dest_ip, self.dest_port))

        log.info(f"VideoSender 初始化完成 → {dest_ip}:{dest_port}  "
                 f"JPEG quality={jpeg_quality}  frag_size={frag_size}B")

    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """启动后台发送线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, name="VideoSenderThread", daemon=True
        )
        self._thread.start()
        log.info("发送线程已启动。")

    def stop(self):
        """停止后台线程并释放资源。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._sock.close()
        log.info("VideoSender 已停止。")

    # ──────────────────────────────────────────────────────────────────────────
    def send_frame(self, frame):
        """
        将一帧图像放入发送队列（非阻塞）。
        若队列已满，自动丢弃最旧的一帧，保证队列中始终是最新帧。

        参数
        ----
        frame : np.ndarray  BGR 格式图像，形状 (H, W, 3)
        """
        if not self._running:
            log.warning("send_frame 被调用，但发送线程尚未启动或已停止。")
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # 丢弃旧帧，塞入新帧
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    def _worker(self):
        """后台发送循环：JPEG 压缩 → 分片 → 逐片发送。"""
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        dest = (self.dest_ip, self.dest_port)   # fallback（connect 之后实际不用）

        fps_counter = 0
        fps_ts      = time.perf_counter()

        while self._running:
            # 等待队列中的帧（超时后检查 _running 标志）
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # ── Step 1：JPEG 编码 ────────────────────────────────────────────
            ret, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ret:
                log.error("JPEG 编码失败，跳过本帧。")
                continue
            data = buf.tobytes()

            # ── Step 2：分片发送 ─────────────────────────────────────────────
            frags       = [data[i : i + self.frag_size] for i in range(0, len(data), self.frag_size)]
            total_frags = len(frags)
            fid         = self._frame_id & 0xFFFFFFFF
            self._frame_id += 1

            for frag_id, frag in enumerate(frags):
                header = struct.pack(HEADER_FMT, fid, frag_id, total_frags, len(frag))
                try:
                    self._sock.send(header + frag)   # 已 connect，直接 send
                except OSError as e:
                    log.error(f"发送失败: {e}")
                    break   # 本帧剩余分片放弃，等下一帧

            # ── 每秒打印一次发送帧率 ─────────────────────────────────────────
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_ts >= 5.0:
                log.info(f"发送 FPS ≈ {fps_counter / (now - fps_ts):.1f}  "
                         f"JPEG size ≈ {len(data) / 1024:.1f} KB  "
                         f"frags = {total_frags}")
                fps_counter = 0
                fps_ts = now

    # ── 上下文管理器支持 ──────────────────────────────────────────────────────
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ─── 独立运行示例（仅测试用）────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    DEST_IP = "10.147.36.61"
    W_img, H_img = 960, 540

    log.info("自测模式：发送随机噪声帧（模拟 color_frame）")
    log.info(f"目标地址: {DEST_IP}:{DEFAULT_PORT}")

    with VideoSender(dest_ip=DEST_IP) as sender:
        try:
            while True:
                # ── 把这里替换成你的真实 color_frame ──────────────────────
                color_frame = np.random.randint(0, 255, (H_img, W_img, 3), dtype=np.uint8)
                # ──────────────────────────────────────────────────────────
                sender.send_frame(color_frame)
                time.sleep(1 / 60)   # 模拟 60 FPS 产帧速度
        except KeyboardInterrupt:
            log.info("用户中断，退出。")