import sys
import subprocess
import time
import socket
import threading
from dataclasses import dataclass
import struct
from loguru import logger

# WiFi 和 TCP 服务器配置
WIFI_SSID = "uavap_tsang"
WIFI_PWD = "888888887"
TCP_HOST = "0.0.0.0" # 监听所有地址
TCP_PORT = 5001 # 监听 5001 端口

# =========================================================
# 以下是 WiFi 连接部分，用户请勿改动
# =========================================================

def run_cmd(cmd):
    """
    执行系统命令并返回输出
    """
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return out.decode().strip()
    except subprocess.CalledProcessError as e:
        return e.output.decode().strip()
    

def is_connected(target_ssid):
    """
    检查当前 WiFi 是否已连接到目标 SSID
    """
    result = run_cmd("nmcli -t -f ACTIVE,SSID dev wifi")
    # 示例返回： "yes:uavap_tsang\nno:OtherAP"
    for line in result.split("\n"):
        if line.startswith("yes:"):
            connected_ssid = line.split(":")[1]
            return connected_ssid == target_ssid
    return False
    

def connect_wifi(ssid, password, timeoutms=20000):
    """
    自动连接 WiFi:
    1. 若已连接则直接返回
    2. 若未连接则自动连接
    3. 超时仍未连上则返回 False
    """
    logger.info(f"[WiFi] 检查是否已连接到 {ssid}...")

    if is_connected(ssid):
        logger.info(f"[WiFi] 已连接到目标 WiFi: {ssid}，无需重复连接。")
        return True

    logger.info(f"[WiFi] 未连接，将尝试连接 SSID = {ssid}")

    # 删除旧配置避免冲突
    run_cmd(f"nmcli connection delete '{ssid}'")

    # 发起连接
    run_cmd(
        f"nmcli dev wifi connect '{ssid}' password '{password}' "
        f"name '{ssid}'"
    )

    start_ms = int(time.time() * 1000)

    while True:
        now_ms = int(time.time() * 1000)
        elapsed = now_ms - start_ms

        if is_connected(ssid):
            logger.info(f"[WiFi] 成功连接到 {ssid}")
            return True

        if elapsed >= timeoutms:
            logger.info(f"[WiFi] ERROR: 连接超时 ({timeoutms} ms)")
            return False

        remaining = timeoutms - elapsed
        logger.info(f"[WiFi] 正在连接中... (剩余 {remaining} ms)")
        time.sleep(0.5)  # 500 毫秒查询一次


def getLocalIP():
    """
    获取 Jetson Orin NX 在 wlan0 上的 IP 地址
    """
    try:
        result = subprocess.check_output(
            "hostname -I | awk '{print $1}'", shell=True
        ).decode().strip()
        return result
    except:
        return "0.0.0.0"


# =========================================================
# 以下是数据包状态机解析器 + TCP 服务器部分，用户请勿改动
# =========================================================
# 数据包协议常量
FRAME_HEAD_1 = 0xA5
FRAME_HEAD_2 = 0x5A
FRAME_TAIL   = 0xFF

def crc16_modbus(data: bytes, poly=0xA001, init=0xFFFF):
    """
    计算 CRC16-MODBUS 校验值
    """
    crc = init
    for byte in data:
        crc ^= byte  # 异或当前数据字节到 CRC16 校验值
        for _ in range(8):
            if crc & 0x0001:  # 如果最低位为 1
                crc = (crc >> 1) ^ poly  # 应用 CRC16 校验算法
            else:
                crc >>= 1  # 右移一位
    return crc


# =========================================================
# 数据包状态机解析器
# =========================================================
class PacketParser:
    """
    状态机解析器，用于解析自定义数据包协议，数据包格式：
    +--------------+--------------+--------------+--------------+--------------+--------------+
    | Header       | Length       | Command      | Data         | CRC16 Check  | Tail         |
    +--------------+--------------+--------------+--------------+--------------+--------------+
    | A5 5A        | XX           | XX           | ... XX ...   | XX XX        | FF           |
    +--------------+--------------+--------------+--------------+--------------+--------------+
    | 2 bytes      | 1 byte       | 1 byte       | n bytes      | 2 bytes      | 1 byte       |
    +--------------+--------------+--------------+--------------+--------------+--------------+
    解析到完整数据包后，调用回调函数 data_process_callback(cmd, datas, length)
    cmd: 命令字, datas: 数据字段 (bytes), length: 数据长度
    """
    def __init__(self, data_process_callback):
        self.step = 0
        self.cnt = 0
        self.buf = bytearray(300)
        self.data_ptr_index = 0
        self.length = 0
        self.cmd = 0
        self.crc16 = 0
        self.data_process_callback = data_process_callback

    def reset(self):
        self.step = 0
        self.cnt = 0

    def feed(self, byte: int):
        """
        喂入一个字节进行状态机解析
        参数 byte: 输入的字节 (0-255)
        解析到完整数据包后，调用回调函数 data_process_callback(cmd, datas, length)
        cmd: 命令字, datas: 数据字段 (bytes), length: 数据长度
        """
        if self.step == 0:  # 等待帧头 1
            if byte == FRAME_HEAD_1:
                self.step = 1
                self.buf[0] = byte
                self.cnt = 1
            return

        elif self.step == 1:  # 等待帧头 2
            if byte == FRAME_HEAD_2:
                self.step = 2
                self.buf[self.cnt] = byte
                self.cnt += 1
            elif byte == FRAME_HEAD_1:
                self.step = 1   # 回到接收第二个帧头
            else:
                self.reset()
            return

        elif self.step == 2:  # 接收长度
            self.length = byte
            self.buf[self.cnt] = byte
            self.cnt += 1
            self.step = 3
            return

        elif self.step == 3:  # 接收命令字
            self.cmd = byte
            self.buf[self.cnt] = byte
            self.cnt += 1

            self.data_ptr_index = self.cnt  # 数据开始位置

            if self.length == 0:
                self.step = 5  # 跳过数据段
            else:
                self.step = 4
            return

        elif self.step == 4:  # 接收数据字段
            self.buf[self.cnt] = byte
            self.cnt += 1

            if (self.cnt - self.data_ptr_index) == self.length:
                self.step = 5
            return

        elif self.step == 5:  # CRC 高字节
            self.crc16 = byte
            self.step = 6
            return

        elif self.step == 6:  # CRC 低字节
            self.crc16 = (self.crc16 << 8) + byte

            calc_crc = crc16_modbus(self.buf[:self.cnt])
            if self.crc16 == calc_crc:
                self.step = 7
            elif byte == FRAME_HEAD_1:
                self.step = 1
            else:
                self.reset()
            return

        elif self.step == 7:  # 接收尾字节
            if byte == FRAME_TAIL:
                data_start = self.data_ptr_index
                data_end = data_start + self.length

                datas = bytes(self.buf[data_start:data_end])

                # 调用用户传入的回调函数
                self.data_process_callback(self.cmd, datas, self.length)

                self.reset()
            elif byte == FRAME_HEAD_1:
                self.step = 1
            else:
                self.reset()
            return
        
        else:
            self.reset()
            return


# =========================================================
# TCP 服务器部分
# =========================================================
class TCPServer:
    """
    TCP 服务器 + 接收线程 + 状态机解包
    1. 启动服务器并等待连接
    2. 接收数据并按字节喂入 PacketParser 进行解析
    3. 提供发送数据包的接口
    4. 解析到完整数据包后，调用回调函数 data_process_callback(cmd, datas, length)
    参数 host: 监听地址
    参数 port: 监听端口
    参数 data_process_callback: 解析到完整数据包后的回调函数，格式为 data_process_callback(cmd, datas, length)
    """
    def __init__(self, host=TCP_HOST, port=TCP_PORT, data_process_callback=None):
        self.host = host
        self.port = port
        self.client = None
        self.server = None
        self.running = False

        # 构造解析器
        self.parser = PacketParser(data_process_callback)

    def start(self):
        """
        启动 TCP 服务器并等待 ESP8266 连接
        连接成功后启动接收线程
        """
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server.bind((self.host, self.port))
        self.server.listen(1)

        logger.info(f"TCP 服务器正在监听 {self.host}:{self.port}")
        logger.info(f"等待 ESP8266 连接...")

        self.client, addr = self.server.accept()
        logger.info(f"ESP8266 已连接，来源地址: {addr}")

        self.running = True

        # 启动专门解析数据的线程
        th = threading.Thread(target=self.recv_thread, daemon=True)
        th.start()

    def recv_thread(self):
        """
        后台线程：接收数据并按字节喂入状态机
        """
        while self.running:
            try:
                data = self.client.recv(1024)
                if not data:
                    logger.warning(f"ESP8266 连接已断开")
                    break

                # 一次性收到 N 个字节 → 逐个喂给解析器
                for b in data:
                    self.parser.feed(b)

            except Exception as e:
                logger.error("接收线程异常: ", e)
                break

        self.running = False
        self.client.close()
        self.server.close()

    def send_packet(self, pkt: bytes):
        """
        发送打好包的数据
        """
        if self.client:
            self.client.sendall(pkt)


'''''''''''''''''''''''''''''''''''''''''''''''
    以下是数据包解包、打包部分，用户需自行修改
'''''''''''''''''''''''''''''''''''''''''''''''

@dataclass
class RecvPacket: # 接收数据包结构体
    ''' ↓ 注意：每次修改了数据包结构后，要同步修改结构体定义和 FMT 格式字符串 ↓ '''
    speed: float
    distance: float
    yaw: float
    pitch: float
    roll: float
    pos_x: float
    pos_y: float
    pos_z: float
    action: int

    FMT = "<ffffffffH"
    ''' ↑ 注意：每次修改了数据包结构后，要同步修改结构体定义和 FMT 格式字符串 ↑ '''

    @classmethod
    def from_bytes(cls, payload: bytes):
        '''
        从字节流解析出数据包实例
        解析时请确保 payload 长度正确
        Example:
        payload = b'...'  # 长度应为 struct.calcsize(RecvPacket.FMT)，当前 34 字节
        pkt = RecvPacket.from_bytes(payload)
        解析后可通过 pkt.speed, pkt.distance, pkt.yaw, pkt.pos_x 等访问字段
        '''
        if len(payload) != struct.calcsize(cls.FMT):
            raise ValueError(f"RecvPacket payload 长度错误: {len(payload)} != {struct.calcsize(cls.FMT)}")
        values = struct.unpack(cls.FMT, payload)
        return cls(*values)
    

@dataclass
class SendPacket: # 发送数据包结构体
    ''' ↓ 注意：每次修改了数据包结构后，要同步修改结构体定义和 FMT 格式字符串 ↓ '''
    speed: float
    yaw: float
    pitch: float
    roll: float
    distance: float
    action: int

    FMT = "<fffffH"
    ''' ↑ 注意：每次修改了数据包结构后，要同步修改结构体定义和 FMT 格式字符串 ↑ '''

    def to_bytes(self) -> bytes:
        '''
        将数据包实例打包成字节流以便发送
        Example:
        pkt = SendPacket(led_mode=1, buzzer_mode=0, camera_angle=30.0)
        payload = pkt.to_bytes()
        发送时可通过 TCPServer.send_packet(payload) 发送
        '''
        return struct.pack(
            self.FMT,
            self.speed,
            self.yaw,
            self.pitch,
            self.roll,
            self.distance,
            self.action,
        )


def build_packet(cmd: int, payload: bytes) -> bytes:
    """
    根据命令字和数据字段构建完整数据包
    参数 cmd: 命令字 (1 byte)
    参数 payload: 数据字段 (bytes)
    返回: 完整数据包 (bytes)
    """
    packet = bytearray()
    packet.append(FRAME_HEAD_1)
    packet.append(FRAME_HEAD_2)
    packet.append(len(payload))
    packet.append(cmd)
    packet += payload

    crc = crc16_modbus(packet)
    packet.append((crc >> 8) & 0xFF)
    packet.append(crc & 0xFF)
    packet.append(FRAME_TAIL)

    return bytes(packet)


class PeriodicSender:
    """
    定时发送数据包的辅助类
    参数 server: TCPServer 实例
    参数 intervalms: 发送间隔，单位毫秒
    参数 build_packet_callback: 构建数据包的回调函数，格式为 build_packet_callback() -> bytes
    该回调函数应返回要发送的完整数据包 (bytes)
    例子:
        def build_packet_callback():
            pkt = SendPacket(speed=1.0, yaw=0.0, pitch=0.0, roll=0.0, distance=10.0, action=0)
            payload = pkt.to_bytes()
            return build_packet(cmd=0x02, payload=payload)

        sender = PeriodicSender(server, intervalms=1000, build_packet_callback=build_packet_callback)
        sender.start()
    该类会在后台线程中每隔指定间隔调用回调函数并发送数据包
    """
    def __init__(self, server: TCPServer, intervalms: float, build_packet_callback):
        self.server = server
        self.interval = intervalms / 1000.0  # 转换为秒
        self.build_packet_callback = build_packet_callback
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def run(self):
        nextTime = time.perf_counter()

        while self.running:
            pkt = self.build_packet_callback() # 构建数据包
            self.server.send_packet(pkt) # 发送数据包

            # 精准计算下次发送时间并睡眠 (考虑执行时间)
            nextTime += self.interval
            sleepTime = nextTime - time.perf_counter()
            if sleepTime > 0:
                time.sleep(sleepTime)


# =========================================================
# 功能测试
# =========================================================
def data_process(cmd, data, length):
    logger.info(f"[Recv] cmd=0x{cmd:02X}, len={length}, data={data.hex()}")
    if cmd == 0x01:
        recvpkt = RecvPacket.from_bytes(data)
        logger.info(f"[Recv] 解析后的数据包: {recvpkt}")


def build_send_packet():
    sendpkt = SendPacket(
        speed=2.0,
        yaw=1.0,
        pitch=0.5,
        roll=0.0,
        distance=15.0,
        action=4
    )
    payload = sendpkt.to_bytes()
    return build_packet(cmd=0x02, payload=payload)


if __name__ == "__main__":
    wifiStatus = connect_wifi(WIFI_SSID, WIFI_PWD)
    logger.info("连接结果: {}", wifiStatus)
    logger.info("本机IP: {}", getLocalIP())

    server = TCPServer(
        host=TCP_HOST,
        port=TCP_PORT,
        data_process_callback=data_process  # 回调函数
    )
    server.start() # 接收线程在内部启动

    sender = PeriodicSender(
        server=server,
        intervalms=10,  # 每 10 ms 发送一次
        build_packet_callback=build_send_packet  # 构建数据包的回调函数
    )
    sender.start() # 启动定时发送
    
    while True:
        time.sleep(1)
