#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
录制用于颜色阈值标定的干净相机视频。

特点：
- 保存到 image_output/video/，默认文件名带时间戳。
- 写入视频的帧不叠加任何文字、框线或调试标记。
- 可选显示预览窗口；预览窗口同样只显示原始画面。

示例：
    python3 test/record_clean_video.py --source csi --duration 30
    python3 test/record_clean_video.py --source camera --camera-index 0 --duration 30
"""

import argparse
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "image_output" / "video"
WINDOW_NAME = "Clean Video Recorder"


def format_gst_framerate(fps):
    """把 OpenCV 录制 fps 转成 GStreamer caps 接受的整数分数。"""
    if fps <= 0:
        raise ValueError("fps 必须大于 0")
    fraction = Fraction(float(fps)).limit_denominator(1001)
    return f"{fraction.numerator}/{fraction.denominator}"


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    """Jetson CSI 摄像头 GStreamer 管线，保持与主链路相近的默认参数。"""
    gst_framerate = format_gst_framerate(framerate)
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){gst_framerate} ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def build_output_path(args):
    if args.output:
        return Path(args.output).expanduser()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = args.container.lower().lstrip(".")
    return DEFAULT_OUTPUT_DIR / f"clean_color_calibration_{timestamp}.{ext}"


def open_capture(args):
    if args.source == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        if args.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        description = f"camera index={args.camera_index}"

    elif args.source == "csi":
        pipeline = gstreamer_pipeline(
            sensor_id=args.sensor_id,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            display_width=args.display_width,
            display_height=args.display_height,
            framerate=args.fps,
            flip_method=args.flip_method,
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        description = "CSI pipeline:\n" + pipeline

    else:
        raise ValueError(f"未知输入源: {args.source}")

    if not cap.isOpened():
        raise RuntimeError(f"无法打开输入源: {description}")
    return cap, description


def make_writer(output_path, frame_size, fps, fourcc_name):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")
    return writer


def parse_args():
    parser = argparse.ArgumentParser(description="录制干净原始画面，用于后续颜色阈值标定。")
    parser.add_argument("--source", choices=("csi", "camera"), default="csi", help="输入源类型，默认 csi。")
    parser.add_argument("--output", help="输出视频路径；默认写入 image_output/video/。")
    parser.add_argument("--duration", type=float, default=0.0, help="录制时长，秒；0 表示按 q/Esc 手动结束。")
    parser.add_argument("--fps", type=float, default=30.0, help="录制帧率。")
    parser.add_argument("--container", choices=("mp4", "avi"), default="mp4", help="默认输出容器。")
    parser.add_argument("--fourcc", default="mp4v", help="VideoWriter FourCC，mp4 默认建议 mp4v。")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口。")

    parser.add_argument("--camera-index", type=int, default=0, help="普通 USB 摄像头编号。")
    parser.add_argument("--width", type=int, default=0, help="普通摄像头请求宽度，0 表示不设置。")
    parser.add_argument("--height", type=int, default=0, help="普通摄像头请求高度，0 表示不设置。")

    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor-id。")
    parser.add_argument("--capture-width", type=int, default=1920, help="CSI 采集宽度。")
    parser.add_argument("--capture-height", type=int, default=1080, help="CSI 采集高度。")
    parser.add_argument("--display-width", type=int, default=960, help="CSI 输出宽度。")
    parser.add_argument("--display-height", type=int, default=540, help="CSI 输出高度。")
    parser.add_argument("--flip-method", type=int, default=2, help="CSI nvvidconv flip-method。")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = build_output_path(args)
    cap, description = open_capture(args)

    writer = None
    frame_count = 0
    start_time = time.monotonic()
    last_report_time = start_time

    print("=" * 72)
    print("开始录制干净视频：不会向保存的视频帧中绘制任何调试标记。")
    print(description)
    print(f"输出路径: {output_path}")
    if args.duration > 0:
        print(f"录制时长: {args.duration:.1f} s")
    else:
        print("录制时长: 手动结束，按 q 或 Esc 停止。")
    print("=" * 72)

    if not args.no_preview:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读取画面失败，停止录制。")
                break

            if writer is None:
                height, width = frame.shape[:2]
                writer = make_writer(output_path, (width, height), args.fps, args.fourcc)
                print(f"视频尺寸: {width} x {height}, fps={args.fps:.2f}, fourcc={args.fourcc}")

            writer.write(frame)
            frame_count += 1

            if not args.no_preview:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("用户停止录制。")
                    break

            now = time.monotonic()
            if now - last_report_time >= 2.0:
                elapsed = now - start_time
                print(f"录制中: {elapsed:.1f}s, {frame_count} frames")
                last_report_time = now

            if args.duration > 0 and now - start_time >= args.duration:
                print("达到指定录制时长。")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - start_time, 1e-6)
    print("=" * 72)
    print(f"录制完成: {output_path}")
    print(f"帧数: {frame_count}, 平均写入帧率: {frame_count / elapsed:.2f} fps")
    print("=" * 72)


if __name__ == "__main__":
    main()
