#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形状编码彩色标记 HSV 阈值调试工具。

用途：
- 现场使用 CSI/USB 摄像头调试 3 个颜色类的 HSV 阈值。
- 或使用保存的图片/视频离线复盘调参。
- 只输出 JSON 和可复制的 Python 字典片段，不自动修改主链路代码。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_core.camera import gstreamer_pipeline
from uav_core.color_marker_pose import (
    COLOR_CLASS_DRAW_BGR,
    COLOR_CLASS_HSV_RANGES,
    MAX_MARKER_AREA_FRACTION,
    MIN_MARKER_AREA_PX,
    MORPH_KERNEL_SIZE,
)


COLOR_CLASS_NAMES = ["green", "purple", "yellow"]

TRACKBAR_NAMES = ("H_MIN", "H_MAX", "S_MIN", "S_MAX", "V_MIN", "V_MAX")
WINDOW_NAME = "Color Marker HSV Tuner"
OUTPUT_DEFAULT = PROJECT_ROOT / "test" / "color_marker_hsv_thresholds.json"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720


def clamp(value, low, high):
    return int(max(low, min(high, value)))


def noop(_value):
    pass


def normalize_ranges_to_slider_values(ranges):
    """
    把主链路使用的 HSV ranges 转成滑条值。

    若 Hue 跨 0，约定用 H_MIN > H_MAX 表示，例如：
    [((170, ...), (179, ...)), ((0, ...), (8, ...))] -> H_MIN=170, H_MAX=8
    """
    if not ranges:
        return [0, 179, 0, 255, 0, 255]

    if (
        len(ranges) == 2
        and ranges[0][1][0] == 179
        and ranges[1][0][0] == 0
    ):
        lower = ranges[0][0]
        upper = ranges[1][1]
        s_min = min(ranges[0][0][1], ranges[1][0][1])
        v_min = min(ranges[0][0][2], ranges[1][0][2])
        s_max = max(ranges[0][1][1], ranges[1][1][1])
        v_max = max(ranges[0][1][2], ranges[1][1][2])
        return [lower[0], upper[0], s_min, s_max, v_min, v_max]

    lower, upper = ranges[0]
    return [lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]]


def slider_values_to_ranges(values):
    h_min, h_max, s_min, s_max, v_min, v_max = [int(v) for v in values]

    s_min, s_max = sorted((clamp(s_min, 0, 255), clamp(s_max, 0, 255)))
    v_min, v_max = sorted((clamp(v_min, 0, 255), clamp(v_max, 0, 255)))
    h_min = clamp(h_min, 0, 179)
    h_max = clamp(h_max, 0, 179)

    if h_min <= h_max:
        return [((h_min, s_min, v_min), (h_max, s_max, v_max))]

    return [
        ((h_min, s_min, v_min), (179, s_max, v_max)),
        ((0, s_min, v_min), (h_max, s_max, v_max)),
    ]


def make_hsv_mask(hsv_frame, ranges):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower_arr, upper_arr))

    if MORPH_KERNEL_SIZE > 1:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def select_largest_component(mask, frame_area):
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    max_area = frame_area * MAX_MARKER_AREA_FRACTION
    best_label = None
    best_area = 0.0
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_MARKER_AREA_PX or area > max_area:
            continue
        if area > best_area:
            best_label = label
            best_area = area

    if best_label is None:
        return None

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    center = tuple(np.round(centroids[best_label]).astype(int))
    return {
        "area": best_area,
        "bbox": (x, y, w, h),
        "center": center,
    }


def ranges_to_json(ranges):
    return [
        {
            "lower": list(lower),
            "upper": list(upper),
        }
        for lower, upper in ranges
    ]


def format_python_thresholds(thresholds):
    lines = ["COLOR_CLASS_HSV_RANGES = {"]
    for name in COLOR_CLASS_NAMES:
        ranges = slider_values_to_ranges(thresholds[name])
        range_text = ", ".join(f"({lower}, {upper})" for lower, upper in ranges)
        lines.append(f'    "{name}": [{range_text}],')
    lines.append("}")
    return "\n".join(lines)


class FrameSource:
    def __init__(self, args):
        self.args = args
        self.cap = None
        self.image = None

        if args.source == "image":
            if not args.path:
                raise ValueError("--source image 需要指定 --path")
            self.image = cv2.imread(args.path)
            if self.image is None:
                raise FileNotFoundError(f"无法读取图片: {args.path}")

        elif args.source == "video":
            if not args.path:
                raise ValueError("--source video 需要指定 --path")
            self.cap = cv2.VideoCapture(args.path)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频: {args.path}")

        elif args.source == "camera":
            self.cap = cv2.VideoCapture(args.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开普通摄像头: index={args.camera_index}")

        elif args.source == "csi":
            pipeline = gstreamer_pipeline(
                sensor_id=args.sensor_id,
                capture_width=args.capture_width,
                capture_height=args.capture_height,
                display_width=args.display_width,
                display_height=args.display_height,
                framerate=args.framerate,
                flip_method=args.flip_method,
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开 CSI 摄像头")
            print("CSI pipeline:")
            print(pipeline)

        else:
            raise ValueError(f"未知输入源: {args.source}")

    def read(self):
        if self.image is not None:
            return True, self.image.copy()

        ok, frame = self.cap.read()
        if ok:
            return True, frame

        if self.args.source == "video" and self.args.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.cap.read()

        return False, None

    def release(self):
        if self.cap is not None:
            self.cap.release()


class HSVTuner:
    def __init__(self, args):
        self.args = args
        self.source = FrameSource(args)
        self.current_idx = 0
        self.paused = args.source == "image"
        self.last_frame = None
        self.thresholds = {
            name: normalize_ranges_to_slider_values(COLOR_CLASS_HSV_RANGES[name])
            for name in COLOR_CLASS_NAMES
        }

    @property
    def current_name(self):
        return COLOR_CLASS_NAMES[self.current_idx]

    def setup_windows(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.args.window_width, self.args.window_height)
        for name in TRACKBAR_NAMES:
            max_value = 179 if name.startswith("H_") else 255
            cv2.createTrackbar(name, WINDOW_NAME, 0, max_value, noop)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.load_current_to_trackbars()

    def load_current_to_trackbars(self):
        values = self.thresholds[self.current_name]
        for trackbar_name, value in zip(TRACKBAR_NAMES, values):
            cv2.setTrackbarPos(trackbar_name, WINDOW_NAME, int(value))

    def read_trackbars(self):
        return [
            cv2.getTrackbarPos(trackbar_name, WINDOW_NAME)
            for trackbar_name in TRACKBAR_NAMES
        ]

    def save_current_from_trackbars(self):
        self.thresholds[self.current_name] = self.read_trackbars()

    def select_marker(self, idx):
        self.save_current_from_trackbars()
        self.current_idx = idx % len(COLOR_CLASS_NAMES)
        self.load_current_to_trackbars()
        print(f"当前颜色类: {self.current_idx + 1}. {self.current_name}")

    def reset_current(self):
        self.thresholds[self.current_name] = normalize_ranges_to_slider_values(
            COLOR_CLASS_HSV_RANGES[self.current_name]
        )
        self.load_current_to_trackbars()
        print(f"已重置: {self.current_name}")

    def suggest_from_roi(self, frame_x, frame_y):
        if self.last_frame is None:
            return

        h, w = self.last_frame.shape[:2]
        radius = self.args.roi_radius
        x0 = clamp(frame_x - radius, 0, w - 1)
        x1 = clamp(frame_x + radius + 1, 0, w)
        y0 = clamp(frame_y - radius, 0, h - 1)
        y1 = clamp(frame_y + radius + 1, 0, h)
        roi = self.last_frame[y0:y1, x0:x1]
        if roi.size == 0:
            return

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        median = np.median(hsv_roi, axis=0)
        p_low = np.percentile(hsv_roi, 5, axis=0)
        p_high = np.percentile(hsv_roi, 95, axis=0)

        h_center = int(round(median[0]))
        h_low_raw = h_center - self.args.h_margin
        h_high_raw = h_center + self.args.h_margin
        if h_low_raw < 0:
            h_min = 180 + h_low_raw
            h_max = h_high_raw
        elif h_high_raw > 179:
            h_min = h_low_raw
            h_max = h_high_raw - 180
        else:
            h_min = h_low_raw
            h_max = h_high_raw

        s_min = clamp(int(p_low[1]) - self.args.sv_margin, 0, 255)
        s_max = clamp(int(p_high[1]) + self.args.sv_margin, 0, 255)
        v_min = clamp(int(p_low[2]) - self.args.sv_margin, 0, 255)
        v_max = clamp(int(p_high[2]) + self.args.sv_margin, 0, 255)

        self.thresholds[self.current_name] = [
            clamp(h_min, 0, 179),
            clamp(h_max, 0, 179),
            s_min,
            s_max,
            v_min,
            v_max,
        ]
        self.load_current_to_trackbars()
        print(
            f"采样 {self.current_name}: HSV median=({median[0]:.1f}, "
            f"{median[1]:.1f}, {median[2]:.1f}) -> {self.thresholds[self.current_name]}"
        )

    def on_mouse(self, event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or self.last_frame is None:
            return

        frame_h, frame_w = self.last_frame.shape[:2]
        if y < frame_h and x < frame_w:
            self.suggest_from_roi(x, y)
        elif y < frame_h and frame_w <= x < frame_w * 2:
            self.suggest_from_roi(x - frame_w, y)
        elif frame_h <= y < frame_h * 2 and frame_w <= x < frame_w * 2:
            self.suggest_from_roi(x - frame_w, y - frame_h)

    def save_thresholds(self):
        self.save_current_from_trackbars()
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": self.args.source,
            "path": self.args.path,
            "camera_index": self.args.camera_index,
            "opencv_hsv_range": {
                "H": [0, 179],
                "S": [0, 255],
                "V": [0, 255],
            },
            "color_class_order": COLOR_CLASS_NAMES,
            "thresholds": {
                name: ranges_to_json(slider_values_to_ranges(self.thresholds[name]))
                for name in COLOR_CLASS_NAMES
            },
        }
        output_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\n" + "=" * 72)
        print(f"阈值已保存到: {output_path}")
        print("可复制到 uav_core/color_marker_pose.py 的 Python 片段：")
        print(format_python_thresholds(self.thresholds))
        print("=" * 72 + "\n")

    def read_display_frame(self):
        if self.paused and self.last_frame is not None:
            return True, self.last_frame.copy()

        ok, frame = self.source.read()
        if ok:
            self.last_frame = frame.copy()
        return ok, frame

    def draw_panel_title(self, image, text):
        cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    def draw_detection(self, image, detection, name, color, thickness=2):
        x, y, w, h = detection["bbox"]
        cx, cy = detection["center"]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(image, (cx, cy), 5, color, -1)
        cv2.putText(
            image,
            f"{name} area={detection['area']:.0f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    def make_canvas(self, frame):
        self.save_current_from_trackbars()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_area = float(frame.shape[0] * frame.shape[1])

        original = frame.copy()
        all_overlay = frame.copy()
        current_overlay = frame.copy()

        current_name = self.current_name
        current_detection = None
        current_mask = None

        for name in COLOR_CLASS_NAMES:
            ranges = slider_values_to_ranges(self.thresholds[name])
            mask = make_hsv_mask(hsv_frame, ranges)
            detection = select_largest_component(mask, frame_area)

            if name == current_name:
                current_mask = mask
                current_detection = detection

            if detection is not None:
                self.draw_detection(
                    all_overlay,
                    detection,
                    name,
                    COLOR_CLASS_DRAW_BGR.get(name, (255, 255, 255)),
                    thickness=2,
                )

        if current_mask is None:
            current_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)

        if current_detection is not None:
            self.draw_detection(
                current_overlay,
                current_detection,
                current_name,
                COLOR_CLASS_DRAW_BGR.get(current_name, (255, 255, 255)),
                thickness=3,
            )

        values = self.thresholds[current_name]
        ranges = slider_values_to_ranges(values)
        range_text = " | ".join(f"{lower}->{upper}" for lower, upper in ranges)
        status = "PAUSED" if self.paused else "LIVE"
        detect_text = "no component"
        if current_detection is not None:
            cx, cy = current_detection["center"]
            detect_text = f"area={current_detection['area']:.0f} center=({cx},{cy})"

        self.draw_panel_title(original, "Original: click color marker to sample ROI")
        self.draw_panel_title(all_overlay, "All marker detections with current thresholds")
        self.draw_panel_title(mask_bgr, f"Current mask: {self.current_idx + 1}. {current_name}")
        self.draw_panel_title(current_overlay, "Current color-class detection")

        cv2.putText(
            mask_bgr,
            f"{status} | HSV {range_text}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            mask_bgr,
            detect_text,
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            current_overlay,
            "Keys: 1-3 select | n/p switch | space pause | r reset | s save | q/Esc quit",
            (10, current_overlay.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        top = np.hstack([original, all_overlay])
        bottom = np.hstack([mask_bgr, current_overlay])
        return np.vstack([top, bottom])

    def run(self):
        print("颜色类顺序：")
        for idx, name in enumerate(COLOR_CLASS_NAMES, start=1):
            print(f"  {idx}: {name}")
        print("按 s 保存阈值并打印可复制代码片段；按 q 或 Esc 退出。")

        self.setup_windows()

        try:
            while True:
                ok, frame = self.read_display_frame()
                if not ok:
                    print("没有读取到新画面，按 q/Esc 退出。")
                    key = cv2.waitKey(30) & 0xFF
                    if key in (27, ord("q")):
                        break
                    continue

                canvas = self.make_canvas(frame)
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(30) & 0xFF

                if key in (27, ord("q")):
                    break
                if key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("s"):
                    self.save_thresholds()
                elif key == ord("r"):
                    self.reset_current()
                elif key == ord("n"):
                    self.select_marker(self.current_idx + 1)
                elif key == ord("p"):
                    self.select_marker(self.current_idx - 1)
                elif ord("1") <= key <= ord("3"):
                    self.select_marker(key - ord("1"))
        finally:
            self.source.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="调试 AprilTag 形状编码彩色备用标记的 HSV 阈值。"
    )
    parser.add_argument(
        "--source",
        choices=("csi", "camera", "image", "video"),
        default="csi",
        help="输入源类型，默认 csi。",
    )
    parser.add_argument("--path", help="图片或视频路径，仅 image/video 输入源需要。")
    parser.add_argument("--camera-index", type=int, default=0, help="普通摄像头编号。")
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT), help="阈值 JSON 输出路径。")
    parser.add_argument("--loop-video", action="store_true", help="视频播放到末尾后循环。")

    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor-id。")
    parser.add_argument("--capture-width", type=int, default=1920, help="CSI 采集宽度。")
    parser.add_argument("--capture-height", type=int, default=1080, help="CSI 采集高度。")
    parser.add_argument("--display-width", type=int, default=960, help="CSI 输出宽度。")
    parser.add_argument("--display-height", type=int, default=540, help="CSI 输出高度。")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 帧率。")
    parser.add_argument("--flip-method", type=int, default=2, help="CSI nvvidconv flip-method。")

    parser.add_argument("--roi-radius", type=int, default=5, help="鼠标取样 ROI 半径，单位像素。")
    parser.add_argument("--h-margin", type=int, default=8, help="鼠标取样后 Hue 自动扩展范围。")
    parser.add_argument("--sv-margin", type=int, default=35, help="鼠标取样后 S/V 自动扩展范围。")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH, help="OpenCV 窗口初始宽度。")
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT, help="OpenCV 窗口初始高度。")
    return parser.parse_args()


def main():
    args = parse_args()
    tuner = HSVTuner(args)
    tuner.run()


if __name__ == "__main__":
    main()
