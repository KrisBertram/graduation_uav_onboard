"""
Jetson CSI 摄像头初始化与 GStreamer 管线。
"""

import sys

import cv2
from loguru import logger


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


def init_camera():
    """打开 CSI 摄像头，返回 VideoCapture 对象"""
    pipeline = gstreamer_pipeline(flip_method=2)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        logger.error("无法打开 CSI 摄像头")
        sys.exit(1)

    logger.success("CSI 单目相机打开成功：")
    logger.info(pipeline)
    return cap
