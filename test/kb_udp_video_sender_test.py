import sys
import os
import cv2
import numpy as np
from pupil_apriltags import Detector
from loguru import logger

sys.path.append('/home/zxt/zxt_ws')
from utils.kb_ImageDumper import ImageDumper
from utils.kb_Image2Video import Image2Video, VideoCodec, SortMethod
from utils.kb_TagVisualizer import TagVisualizer
from utils.udp_video_sender import VideoSender

DEBUG_MODE = False  # 是否启用调试模式，启用后会保存图像并生成视频

# =========================================================
# 相机内参（Matlab 标定结果）
# =========================================================
'''
内参矩阵：
    [ fx    0   cx ]
    [  0   fy   cy ]
    [  0    0    1 ]
'''
cameraMatrix = np.array([
    [1288.028590,         0.0, 522.451003],
    [        0.0, 1294.004313, 254.652409],
    [        0.0,         0.0,        1.0]
], dtype=np.float32)

cameraMatrix[0,2] = 960/2
cameraMatrix[1,2] = 540/2

'''
径向畸变：
    [ k1, k2, k3 ]
切向畸变：
    [ p1, p2 ]
畸变参数：
    [ k1, k2, p1, p2, k3 ]
'''
distCoeffs = np.array([
    0.130990,   # k1
    0.305167,   # k2
   -0.003977,   # p1
    0.006610,   # p2
   -2.142455    # k3
], dtype=np.float32)

# =========================================================
# AprilTag 参数
# =========================================================
TAG_FAMILY = "tag36h11"  # 使用哪种 AprilTag 家族
april_tag_size = 0.1945  # AprilTag 实际大小，单位：米

# 定义 AprilTag 角点的三维坐标（相对于标签中心）
half_tag_size = april_tag_size / 2.0
object_points = np.array([
    [-half_tag_size,  half_tag_size, 0.0],
    [ half_tag_size,  half_tag_size, 0.0],
    [ half_tag_size, -half_tag_size, 0.0],
    [-half_tag_size, -half_tag_size, 0.0]
], dtype=np.float32)

# =========================================================
# GStreamer（Jetson CSI 摄像头）
# =========================================================
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# =========================================================
# AprilTag Detector
# =========================================================
detector = Detector(
    families=TAG_FAMILY,    # 使用哪种 AprilTag 家族
    nthreads=4,             # 使用多少 CPU 线程
    quad_decimate=1.0,      # 图像降采样比例，1.0 表示原分辨率，> 1.0 会更快，但远距离精度下降
    quad_sigma=0.0,         # 高斯模糊参数，0.0 表示不模糊
    refine_edges=True,      # 是否细化边缘检测
    # decode_sharpening=0.25  # 解码锐化参数
)

# =========================================================
# 打开 CSI 摄像头
# =========================================================
frame_count = 0
pipeline = gstreamer_pipeline(flip_method=2)
video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not video_capture.isOpened():
    logger.error("无法打开 CSI 摄像头")
    sys.exit(1)

logger.success("CSI 单目相机打开成功：")
logger.info(pipeline)

# =========================================================
# 主循环
# =========================================================
logger.info("开始 AprilTag 识别与位姿解算")

image_dumper = ImageDumper(
    base_path="./image_output",
    interval_type="time",
    interval_value=0.08,  # 0.04 秒
    time_unit="seconds",
    storage_mode=ImageDumper.MODE_OVERWRITE,
    filename_format=ImageDumper.FORMAT_SEQUENTIAL
)

video_converter = Image2Video(
    input_dir="./image_output",  # 图片目录
    output_path="./image_output/video/output.mp4",  # 输出视频路径
    fps=12.0,
    codec=VideoCodec.XVID
)

sender = VideoSender(dest_ip="10.147.36.61", jpeg_quality=20)
sender.start()

try:
    while True:
        frame_count += 1
        color_ret, color_frame = video_capture.read()
        if not color_ret:
            logger.warning(f"第 {frame_count} 帧图像读取失败")
            continue

        # 将彩色图像转为灰度图像
        gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        # AprilTag 检测
        april_tags = detector.detect(
            gray_frame, # 输入灰度图
            estimate_tag_pose=False, # 是否估计姿态，暂时先关掉，后续使用 solvePnP
            camera_params=None,
            tag_size=None
        )

        # 位姿估计
        for tag in april_tags:
            image_points = tag.corners.astype(np.float32)  # 提取角点

            # 使用 solvePnP 估计位姿
            pnp_ok, pnp_rvec, pnp_tvec = cv2.solvePnP(
                object_points,                  # 三维点
                image_points,                   # 二维点
                cameraMatrix,                   # 相机内参矩阵
                distCoeffs,                     # 畸变系数
                flags=cv2.SOLVEPNP_IPPE_SQUARE  # 使用适合方形标志的算法
            )

            if not pnp_ok:
                logger.warning(f"Tag {tag.tag_id} solvePnP 失败")
                continue

            pnp_x, pnp_y, pnp_z = pnp_tvec.flatten()  # 提取平移向量

            logger.info(
                f"ID: {tag.tag_id} | "
                f"X: {pnp_x:+.3f} m  Y: {pnp_y:+.3f} m  Z: {pnp_z:+.3f} m"
            )

            # ---------- 画 Tag 边框 ----------
            for i in range(4):
                pt1 = tuple(image_points[i].astype(int))
                pt2 = tuple(image_points[(i + 1) % 4].astype(int))
                cv2.line(color_frame, pt1, pt2, (0, 255, 0), 2)

            # ---------- 画中心点 ----------
            draw_center = tuple(tag.center.astype(int))
            cv2.circle(color_frame, draw_center, 5, (0, 0, 255), -1)

            # ---------- 画坐标轴 ----------
            cv2.drawFrameAxes(
                color_frame,
                cameraMatrix,
                distCoeffs,
                pnp_rvec,
                pnp_tvec,
                half_tag_size
            )

            # ---------- 显示数值 ----------
            cv2.putText(
                color_frame,
                f"ID: {tag.tag_id} | X: {pnp_x:+.3f} m  Y: {pnp_y:+.3f} m  Z: {pnp_z:+.3f} m",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            ''' 调用 TagVisualizer 类，计算像素比例尺 '''
            meter_per_pixel = TagVisualizer.compute_pixel_scale_on_tag(tag.corners, april_tag_size)
            img_center = TagVisualizer.draw_image_center(color_frame)
            TagVisualizer.draw_tvec_vector_scaled(
                color_frame,
                img_center,
                pnp_tvec,
                meter_per_pixel,
                scale=1.0,
                color=(255,128,0)
            )

            # 把 3D 点 (0,0,0) 投影回 2D 图像坐标系，验证投影是否正确
            # TagVisualizer.draw_reprojected_center(
            #     color_frame,
            #     pnp_rvec,
            #     pnp_tvec,
            #     cameraMatrix,
            #     distCoeffs,
            #     color=(255,128,0)
            # )

            # 计算所有角点的重投影误差，如果误差 < 0.5px，说明模型是健康的
            mean_error, error = TagVisualizer.compute_reprojection_error(
                object_points,
                image_points,
                pnp_rvec,
                pnp_tvec,
                cameraMatrix,
                distCoeffs
            )
            print("重投影误差：", mean_error)

            # draw_z_bar(color_frame, pnp_tvec[2][0])
            ''' 计算像素比例尺 '''

        sender.send_frame(color_frame)   # 直接传入你的 BGR ndarray

        # 转储图像
        if DEBUG_MODE: image_dumper.dump(color_frame)

        # cv2.imshow("AprilTag Detection", color_frame)
        # if 27 == cv2.waitKey(1): break

finally:
    # =========================================================
    # 资源释放
    # =========================================================
    sender.stop()
    video_capture.release()
    cv2.destroyAllWindows()
    logger.success("程序安全退出")

    if DEBUG_MODE:
        # 获取统计信息
        stats = image_dumper.get_stats()
        logger.info(f"保存了 {stats['saved_frames']} 张图像")

        # 创建视频
        stats = video_converter.create_video(
            pattern="*.jpg",
            recursive=False,
            sort_method=SortMethod.NATURAL,
            frame_interval=1
        )
        
        if stats.get("success"):
            logger.info(f"视频创建成功: {stats['output_path']}")
            logger.info(f"处理了 {stats['processed_frames']} 帧，耗时 {stats['processing_time_seconds']:.2f} 秒")
    else:
        logger.info("调试模式关闭，直接退出")