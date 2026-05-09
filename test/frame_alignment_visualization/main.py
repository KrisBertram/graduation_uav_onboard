#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FrameAligner 车端坐标系到无人机坐标系在线对齐可视化演示脚本。

该脚本只做感知、估计和画面输出：
- 读取下视相机画面；
- 接收无人车 TCP 位姿；
- 视觉有效时绘制真实 PnP 坐标架并在线对齐车端坐标系；
- 在相机画面下方的俯视地图画布中展示车端原始坐标系如何旋转和平移到无人机坐标系；
- 视觉丢失时用车端位姿估计降落点，并在灰色未知区域继续绘制坐标架；
- 通过 UDP 图传输出带叠加信息的画面，同时可保存 MP4。

安全边界：本脚本不连接飞控，不导入 DataLink，不发送任何飞控控制指令。
"""

import argparse
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_core.apriltag_pose import (  # noqa: E402
    TAG_SIZES,
    cameraMatrix,
    distCoeffs,
    estimate_pose,
    init_detector,
    select_target_tag,
)
from uav_core.color_marker_pose import (  # noqa: E402
    COLOR_MARKER_AXIS_LENGTH_M,
    draw_color_marker_debug,
    estimate_color_marker_pose,
)
from uav_core.frame_alignment import FrameAligner, wrap_angle  # noqa: E402
from uav_core.vehicle_state import VehicleStateReceiver  # noqa: E402
from uav_core.visual_control import (  # noqa: E402
    estimate_tag_forward_yaw_body,
    pnp_to_body_xy,
)
from utils.udp_video_sender import DEFAULT_PORT, VideoSender  # noqa: E402


WINDOW_NAME = "Frame Alignment Visualization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "image_output" / "video"
DEFAULT_UDP_IP = "10.105.26.61"

# 文字叠加层总开关。False 时只保留相机画面、灰色未知区域和三维坐标架。
TEXT_OVERLAY_ENABLED = False
COLOR_MARKER_DEBUG_OVERLAY_ENABLED = False
ALIGNMENT_MAP_ENABLED = True

TAG_FORWARD_AXIS = "+Y"
VEHICLE_STATE_TIMEOUT_S = 0.3
ALIGN_YAW_ALPHA = 0.05
ALIGN_POS_ALPHA = 0.05
ALIGNMENT_ANIMATION_DURATION_S = 1.8

UNKNOWN_GRAY = (214, 214, 214)
FOV_BORDER_COLOR = (245, 245, 245)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (35, 35, 35)
VISION_AXIS_LABEL_COLOR = (0, 255, 0)
EST_AXIS_LABEL_COLOR = (0, 210, 255)
ALIGN_MAP_BG = (24, 28, 32)
ALIGN_MAP_GRID = (62, 68, 72)
ALIGN_MAP_RAW = (130, 130, 130)
ALIGN_MAP_ALIGNED = (255, 210, 0)
ALIGN_MAP_VISUAL = (0, 230, 80)
ALIGN_MAP_RESIDUAL = (0, 120, 255)
ALIGN_MAP_DRONE = (245, 245, 245)
MIN_VIEW_SCALE = 0.22
VIEW_MARGIN_PX = 46


@dataclass
class PoseObservation:
    """视觉或车端估计得到的 Tag/降落点位姿。"""

    source: str
    rvec: np.ndarray
    tvec: np.ndarray
    body_xy: np.ndarray
    yaw_body: float
    axis_length_m: float
    z_m: float
    tag_id: Optional[int] = None
    image_points: Optional[np.ndarray] = None
    tag_center: Optional[np.ndarray] = None
    color_observation: Optional[object] = None


@dataclass
class ViewState:
    """灰区扩展画面的缩放和平移状态。"""

    scale: float = 1.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    initialized: bool = False
    lost_session_active: bool = False
    bounds_min: Optional[np.ndarray] = None
    bounds_max: Optional[np.ndarray] = None
    target_scale: float = 1.0
    target_offset: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))


@dataclass
class ErrorStats:
    """视觉 PnP 与车端估计之间的误差统计。"""

    count: int = 0
    pos_sum: float = 0.0
    pos_sumsq: float = 0.0
    pos_max: float = 0.0
    yaw_abs_sum: float = 0.0
    yaw_abs_max: float = 0.0
    last_pos_error_m: Optional[float] = None
    last_yaw_error_deg: Optional[float] = None

    def update(self, visual_pose: PoseObservation, estimated_pose: PoseObservation):
        pos_error = float(np.linalg.norm(visual_pose.body_xy - estimated_pose.body_xy))
        yaw_error_deg = abs(math.degrees(wrap_angle(visual_pose.yaw_body - estimated_pose.yaw_body)))

        self.count += 1
        self.pos_sum += pos_error
        self.pos_sumsq += pos_error * pos_error
        self.pos_max = max(self.pos_max, pos_error)
        self.yaw_abs_sum += yaw_error_deg
        self.yaw_abs_max = max(self.yaw_abs_max, yaw_error_deg)
        self.last_pos_error_m = pos_error
        self.last_yaw_error_deg = yaw_error_deg

    @property
    def pos_mean(self):
        return self.pos_sum / self.count if self.count else None

    @property
    def pos_rmse(self):
        return math.sqrt(self.pos_sumsq / self.count) if self.count else None

    @property
    def yaw_abs_mean(self):
        return self.yaw_abs_sum / self.count if self.count else None


@dataclass
class RunStats:
    """整次演示的帧计数统计。"""

    frame_count: int = 0
    visual_lost_frames: int = 0
    ugv_pose_est_frames: int = 0


def rotate_xy_2d(vec_xy, yaw):
    """二维向量旋转，用于地图画布播放 FrameAligner 的 R(theta)+t 动画。"""
    vec_xy = np.asarray(vec_xy, dtype=float)
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = vec_xy[..., 0]
    y = vec_xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


@dataclass
class AlignmentMapRenderer:
    """俯视地图画布：展示车端坐标系逐步对齐到无人机坐标系。"""

    enabled: bool = ALIGNMENT_MAP_ENABLED
    max_history: int = 260
    animation_duration_s: float = ALIGNMENT_ANIMATION_DURATION_S
    raw_history: List[np.ndarray] = field(default_factory=list)
    aligned_history: List[np.ndarray] = field(default_factory=list)
    visual_history: List[np.ndarray] = field(default_factory=list)
    last_raw_xy: Optional[np.ndarray] = None
    last_raw_yaw: Optional[float] = None
    last_aligned_xy: Optional[np.ndarray] = None
    last_aligned_yaw: Optional[float] = None
    last_visual_xy: Optional[np.ndarray] = None
    last_visual_yaw: Optional[float] = None
    was_initialized: bool = False
    align_start_time: Optional[float] = None
    lock_theta: float = 0.0
    lock_translation: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    status: str = "WAIT_VISION"

    def update(self, now, vehicle_state, visual_pose, frame_aligner):
        if not self.enabled:
            return

        if vehicle_state is not None:
            self.last_raw_xy = vehicle_xy_for_body_visualization(vehicle_state)
            self.last_raw_yaw = vehicle_yaw_for_body_visualization(vehicle_state)
            self._append(self.raw_history, self.last_raw_xy)

            if frame_aligner.initialized:
                self.last_aligned_xy = frame_aligner.transform_point(self.last_raw_xy)
                self.last_aligned_yaw = wrap_angle(self.last_raw_yaw + frame_aligner.theta)
                self._append(self.aligned_history, self.last_aligned_xy)

        if visual_pose is not None:
            self.last_visual_xy = np.asarray(visual_pose.body_xy, dtype=float)
            self.last_visual_yaw = float(visual_pose.yaw_body)
            self._append(self.visual_history, self.last_visual_xy)

        if frame_aligner.initialized and not self.was_initialized:
            self.align_start_time = now
            self.lock_theta = frame_aligner.theta
            self.lock_translation = frame_aligner.translation.copy()

        self.was_initialized = frame_aligner.initialized
        self.status = self._status(now, visual_pose, frame_aligner)

    def compose(self, camera_canvas, now, frame_aligner):
        if not self.enabled:
            return camera_canvas

        map_canvas = self.render_map_canvas(camera_canvas.shape, now, frame_aligner)
        return np.vstack([camera_canvas, map_canvas])

    def render_map_canvas(self, frame_shape, now, frame_aligner):
        height, width = frame_shape[:2]
        canvas = np.full((height, width, 3), ALIGN_MAP_BG, dtype=np.uint8)
        map_w = width
        map_h = height
        x0 = 0
        y0 = 0
        x1 = width - 1
        y1 = height - 1

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (95, 100, 105), 1, cv2.LINE_AA)

        points = self._map_points(frame_aligner)
        scale, origin_px = self._map_transform(points, x0, y0, map_w, map_h)
        self._draw_grid(canvas, x0, y0, x1, y1, origin_px)

        self._draw_world_axes(
            canvas,
            origin_px,
            scale,
            0.0,
            "UGV Raw Origin",
            ALIGN_MAP_RAW,
            label_offset=(8, 24),
            axis_scale=0.30,
            thickness=2,
        )
        self._draw_world_axes(
            canvas,
            origin_px,
            scale,
            0.0,
            "UAV Frame",
            ALIGN_MAP_DRONE,
            label_offset=(8, -14),
        )
        self._draw_polyline(canvas, self.raw_history, origin_px, scale, ALIGN_MAP_RAW, dotted=True)
        self._draw_polyline(canvas, self.visual_history, origin_px, scale, ALIGN_MAP_VISUAL, dotted=False)

        alpha = self._align_alpha(now)
        theta, translation = self._animated_transform(frame_aligner, alpha)
        if frame_aligner.initialized:
            transformed_origin_px = self._to_px(translation, origin_px, scale)
            if np.linalg.norm(translation) > 1e-4:
                cv2.arrowedLine(
                    canvas,
                    self._to_px(np.zeros(2, dtype=float), origin_px, scale),
                    transformed_origin_px,
                    ALIGN_MAP_RESIDUAL,
                    1,
                    cv2.LINE_AA,
                    tipLength=0.12,
                )
            self._draw_world_axes(
                canvas,
                transformed_origin_px,
                scale,
                theta,
                "UGV Origin Aligned",
                ALIGN_MAP_ALIGNED,
                label_offset=(8, 28),
                axis_scale=0.38,
            )

        animated_history = self._transform_history(self.raw_history, theta, translation)
        self._draw_polyline(canvas, animated_history, origin_px, scale, ALIGN_MAP_ALIGNED, dotted=False)

        if self.last_raw_xy is not None and self.last_raw_yaw is not None:
            raw_axis_xy = self.last_raw_xy
            raw_axis_yaw = self.last_raw_yaw
            self._draw_world_axes(
                canvas,
                self._to_px(raw_axis_xy, origin_px, scale),
                scale,
                raw_axis_yaw,
                "Current Car Raw",
                ALIGN_MAP_RAW,
                label_offset=(8, -12),
                axis_scale=0.28,
                thickness=2,
            )

            aligned_axis_xy = rotate_xy_2d(raw_axis_xy, theta) + translation
            aligned_axis_yaw = wrap_angle(raw_axis_yaw + theta)
            self._draw_world_axes(
                canvas,
                self._to_px(aligned_axis_xy, origin_px, scale),
                scale,
                aligned_axis_yaw,
                "Current Car",
                ALIGN_MAP_ALIGNED,
                axis_scale=0.32,
            )

        if self.last_visual_xy is not None:
            cv2.circle(canvas, self._to_px(self.last_visual_xy, origin_px, scale), 8, ALIGN_MAP_VISUAL, -1, cv2.LINE_AA)

        if self.last_aligned_xy is not None and self.last_visual_xy is not None:
            p0 = self._to_px(self.last_aligned_xy, origin_px, scale)
            p1 = self._to_px(self.last_visual_xy, origin_px, scale)
            cv2.arrowedLine(canvas, p0, p1, ALIGN_MAP_RESIDUAL, 3, cv2.LINE_AA, tipLength=0.18)

        self._draw_legend(canvas, x0, y0, frame_aligner, alpha)
        return canvas

    def _append(self, history, point):
        history.append(np.asarray(point, dtype=float).copy())
        if len(history) > self.max_history:
            del history[0:len(history) - self.max_history]

    def _status(self, now, visual_pose, frame_aligner):
        if not frame_aligner.initialized:
            return "WAIT_VISION"
        if self.align_start_time is not None and now - self.align_start_time < self.animation_duration_s:
            return "ALIGNING"
        if visual_pose is None:
            return "VISION_LOST"
        return "ALIGNED_UPDATING"

    def _align_alpha(self, now):
        if self.align_start_time is None:
            return 1.0 if self.was_initialized else 0.0
        elapsed = max(0.0, now - self.align_start_time)
        raw_alpha = min(1.0, elapsed / max(self.animation_duration_s, 1e-6))
        return raw_alpha * raw_alpha * (3.0 - 2.0 * raw_alpha)

    def _animated_transform(self, frame_aligner, alpha):
        if not frame_aligner.initialized:
            return 0.0, np.zeros(2, dtype=float)
        if self.status == "ALIGNING":
            return alpha * self.lock_theta, alpha * self.lock_translation
        return frame_aligner.theta, frame_aligner.translation

    def _transform_history(self, history, theta, translation):
        if not history:
            return []
        points = np.asarray(history, dtype=float)
        return list(rotate_xy_2d(points, theta) + translation)

    def _map_points(self, frame_aligner):
        points = [np.array([0.0, 0.0], dtype=float)]
        points.extend(self.raw_history)
        points.extend(self.visual_history)
        points.extend(self.aligned_history)
        if self.last_raw_xy is not None:
            points.append(self.last_raw_xy)
            if frame_aligner.initialized:
                points.append(frame_aligner.transform_point(self.last_raw_xy))
        if self.last_visual_xy is not None:
            points.append(self.last_visual_xy)
        return points

    def _map_transform(self, points, x0, y0, map_w, map_h):
        if not points:
            min_xy = np.array([-1.0, -1.0])
            max_xy = np.array([1.0, 1.0])
        else:
            arr = np.asarray(points, dtype=float)
            min_xy = np.min(arr, axis=0)
            max_xy = np.max(arr, axis=0)
            min_xy = np.minimum(min_xy, np.array([-0.5, -0.5]))
            max_xy = np.maximum(max_xy, np.array([0.5, 0.5]))

        size = np.maximum(max_xy - min_xy, 0.5)
        padding = 70
        usable_w = max(1, map_w - 2 * padding)
        usable_h = max(1, map_h - 2 * padding)
        scale = min(usable_w / size[0], usable_h / size[1])
        center_world = (min_xy + max_xy) * 0.5
        center_px = np.array([x0 + map_w * 0.5, y0 + map_h * 0.54], dtype=float)
        origin_px = center_px - np.array([center_world[0] * scale, -center_world[1] * scale], dtype=float)
        return scale, origin_px

    def _to_px(self, xy, origin_px, scale):
        xy = np.asarray(xy, dtype=float)
        return (
            int(round(origin_px[0] + xy[0] * scale)),
            int(round(origin_px[1] - xy[1] * scale)),
        )

    def _draw_grid(self, canvas, x0, y0, x1, y1, origin_px):
        for x in range(x0 + 60, x1, 60):
            cv2.line(canvas, (x, y0), (x, y1), ALIGN_MAP_GRID, 1, cv2.LINE_AA)
        for y in range(y0 + 60, y1, 60):
            cv2.line(canvas, (x0, y), (x1, y), ALIGN_MAP_GRID, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(origin_px[0]), int(origin_px[1])), 7, ALIGN_MAP_DRONE, -1, cv2.LINE_AA)

    def _draw_polyline(self, canvas, history, origin_px, scale, color, dotted=False):
        if len(history) < 2:
            return
        points = [self._to_px(point, origin_px, scale) for point in history]
        for idx in range(1, len(points)):
            if dotted and idx % 2 == 0:
                continue
            cv2.line(canvas, points[idx - 1], points[idx], color, 3, cv2.LINE_AA)

    def _draw_world_axes(
        self,
        canvas,
        origin_px,
        scale,
        yaw,
        label,
        color,
        label_offset=(5, -6),
        axis_scale=0.45,
        thickness=2,
    ):
        axis_len_px = int(np.clip(axis_scale * scale, 24, 88))
        x_dir = np.array([math.cos(yaw), math.sin(yaw)])
        y_dir = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)])
        origin_arr = np.array(origin_px, dtype=float)
        x_end = tuple(np.round(origin_arr + np.array([x_dir[0], -x_dir[1]]) * axis_len_px).astype(int))
        y_end = tuple(np.round(origin_arr + np.array([y_dir[0], -y_dir[1]]) * axis_len_px).astype(int))
        cv2.arrowedLine(canvas, tuple(np.round(origin_arr).astype(int)), x_end, color, thickness, cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(canvas, tuple(np.round(origin_arr).astype(int)), y_end, color, thickness, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(
            canvas,
            label,
            (int(origin_arr[0]) + label_offset[0], int(origin_arr[1]) + label_offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_legend(self, canvas, x0, y0, frame_aligner, alpha):
        lines = [
            f"FrameAligner {self.status}",
            f"alpha={alpha:.2f}",
            f"theta={math.degrees(frame_aligner.theta):+.1f}deg" if frame_aligner.initialized else "theta=n/a",
            f"t=({frame_aligner.translation[0]:+.2f},{frame_aligner.translation[1]:+.2f})m" if frame_aligner.initialized else "t=n/a",
            "gray=UGV raw origin/path  cyan=aligned  green=vision",
        ]
        y = y0 + 30
        for line in lines:
            cv2.putText(canvas, line, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, TEXT_COLOR, 2, cv2.LINE_AA)
            y += 28


def format_gst_framerate(fps):
    """把 fps 转成 GStreamer caps 接受的整数分数。"""
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
    """Jetson CSI 下视相机管线，默认尺寸与主链路保持一致。"""
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
    return DEFAULT_OUTPUT_DIR / f"frame_alignment_visualization_{timestamp}.mp4"


def open_capture(args):
    if args.source == "camera":
        cap = cv2.VideoCapture(args.camera_index)
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


def make_writer(output_path: Path, frame_size, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {output_path}")
    return writer


def vehicle_xy_for_body_visualization(vehicle_state):
    """
    车端坐标转换到本脚本使用的机体系可视化平面。

    车端原始坐标: +X 为车头初始方向，+Y 为车体左侧，yaw 逆时针为正。
    机体系显示:   +X 为无人机前方，+Y 为无人机右侧，yaw 顺时针为正。
    因此进入 FrameAligner 前需要把车端 Y 取反，否则转弯轨迹会左右镜像。
    """
    return np.array([vehicle_state.pos_x, -vehicle_state.pos_y], dtype=float)


def vehicle_yaw_for_body_visualization(vehicle_state):
    """车端逆时针为正的 yaw 转为机体系显示中顺时针为正的 yaw。"""
    return -vehicle_state.yaw_rad


def detect_visual_observation(frame_bgr, detector, color_marker_enabled):
    """检测 AprilTag；失败时按需尝试彩色备用 PnP。"""
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    april_tags = detector.detect(
        gray_frame,
        estimate_tag_pose=False,
        camera_params=None,
        tag_size=None,
    )

    target_tag = select_target_tag(april_tags)
    if target_tag is not None:
        tag_size = TAG_SIZES[target_tag.tag_id]
        pnp_ok, pnp_rvec, pnp_tvec, pnp_x, pnp_y, pnp_z, image_points = estimate_pose(
            target_tag,
            tag_size,
        )
        if pnp_ok:
            body_xy = pnp_to_body_xy(pnp_x, pnp_y)
            yaw_body = estimate_tag_forward_yaw_body(pnp_rvec, TAG_FORWARD_AXIS)
            return PoseObservation(
                source="VISION_TAG",
                rvec=pnp_rvec,
                tvec=pnp_tvec,
                body_xy=body_xy,
                yaw_body=yaw_body,
                axis_length_m=tag_size * 0.5,
                z_m=float(pnp_z),
                tag_id=int(target_tag.tag_id),
                image_points=image_points,
                tag_center=target_tag.center,
            )

    if not color_marker_enabled:
        return None

    color_observation = estimate_color_marker_pose(frame_bgr, cameraMatrix, distCoeffs)
    if color_observation is None:
        return None

    pnp_rvec = color_observation.rvec
    pnp_tvec = color_observation.tvec
    pnp_x, pnp_y, pnp_z = pnp_tvec.flatten()
    body_xy = pnp_to_body_xy(pnp_x, pnp_y)
    yaw_body = estimate_tag_forward_yaw_body(pnp_rvec, TAG_FORWARD_AXIS)
    return PoseObservation(
        source="VISION_COLOR",
        rvec=pnp_rvec,
        tvec=pnp_tvec,
        body_xy=body_xy,
        yaw_body=yaw_body,
        axis_length_m=COLOR_MARKER_AXIS_LENGTH_M,
        z_m=float(pnp_z),
        color_observation=color_observation,
    )


def rotation_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def rvec_with_body_yaw(base_rvec, desired_yaw_body):
    """
    在最近一次视觉 rvec 的基础上绕相机 Z 轴旋转，使 TAG_FORWARD_AXIS 朝向车端估计 yaw。
    同时测试正负旋转方向，选择前向轴误差更小的一组。
    """
    base_yaw = estimate_tag_forward_yaw_body(base_rvec, TAG_FORWARD_AXIS)
    yaw_delta = wrap_angle(desired_yaw_body - base_yaw)
    base_R, _ = cv2.Rodrigues(base_rvec)

    best_rvec = base_rvec
    best_error = float("inf")
    for candidate_delta in (yaw_delta, -yaw_delta, 0.0):
        candidate_R = rotation_z(candidate_delta) @ base_R
        candidate_rvec, _ = cv2.Rodrigues(candidate_R)
        candidate_yaw = estimate_tag_forward_yaw_body(candidate_rvec, TAG_FORWARD_AXIS)
        candidate_error = abs(wrap_angle(candidate_yaw - desired_yaw_body))
        if candidate_error < best_error:
            best_error = candidate_error
            best_rvec = candidate_rvec

    return best_rvec


def estimate_pose_from_vehicle(frame_aligner, vehicle_state, last_visual_pose):
    """由车端位姿估计当前降落点在相机/机体系中的 PnP 风格位姿。"""
    if vehicle_state is None:
        return None
    if last_visual_pose is None:
        return None
    if not frame_aligner.initialized:
        return None
    if last_visual_pose.z_m <= 0.0:
        return None

    body_xy = frame_aligner.transform_point(vehicle_xy_for_body_visualization(vehicle_state))
    desired_yaw_body = wrap_angle(vehicle_yaw_for_body_visualization(vehicle_state) + frame_aligner.theta)
    pnp_x = float(body_xy[1])
    pnp_y = float(-body_xy[0])
    pnp_z = float(last_visual_pose.z_m)
    tvec = np.array([[pnp_x], [pnp_y], [pnp_z]], dtype=np.float32)
    rvec = rvec_with_body_yaw(last_visual_pose.rvec, desired_yaw_body)

    return PoseObservation(
        source="UGV_POSE_EST",
        rvec=rvec,
        tvec=tvec,
        body_xy=np.asarray(body_xy, dtype=float),
        yaw_body=desired_yaw_body,
        axis_length_m=last_visual_pose.axis_length_m,
        z_m=pnp_z,
    )


def update_alignment(frame_aligner, vehicle_state, visual_pose):
    """用当前视觉观测和同帧车端状态更新车/机参考系对齐。"""
    if vehicle_state is None or visual_pose is None:
        return
    frame_aligner.update(
        vehicle_xy_for_body_visualization(vehicle_state),
        vehicle_yaw_for_body_visualization(vehicle_state),
        visual_pose.body_xy,
        visual_pose.yaw_body,
    )


def project_axis_points(pose: PoseObservation):
    object_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [pose.axis_length_m, 0.0, 0.0],
            [0.0, pose.axis_length_m, 0.0],
            [0.0, 0.0, pose.axis_length_m],
        ],
        dtype=np.float32,
    )
    image_points, _ = cv2.projectPoints(
        object_points,
        pose.rvec,
        pose.tvec,
        cameraMatrix,
        distCoeffs,
    )
    return image_points.reshape(-1, 2)


def transform_points(points, view_state: ViewState):
    points = np.asarray(points, dtype=float)
    return points * view_state.scale + view_state.offset


def current_visible_bounds(frame_shape, view_state: ViewState):
    """当前画布对应到原相机图像坐标里的可见范围。"""
    height, width = frame_shape[:2]
    if view_state.scale <= 1e-6:
        return np.array([0.0, 0.0]), np.array([float(width), float(height)])

    canvas_min = (np.array([0.0, 0.0]) - view_state.offset) / view_state.scale
    canvas_max = (np.array([float(width), float(height)]) - view_state.offset) / view_state.scale
    fov_min = np.array([0.0, 0.0], dtype=float)
    fov_max = np.array([float(width), float(height)], dtype=float)
    return np.minimum(canvas_min, fov_min), np.maximum(canvas_max, fov_max)


def pose_near_canvas_edge(pose, frame_shape, view_state: ViewState):
    """估计坐标架靠近或越过当前输出画面边缘时，触发视野扩展。"""
    points = project_axis_points(pose)
    if not np.all(np.isfinite(points)):
        return False

    height, width = frame_shape[:2]
    points = transform_points(points, view_state)
    return bool(
        np.any(points[:, 0] < VIEW_MARGIN_PX)
        or np.any(points[:, 0] > width - VIEW_MARGIN_PX)
        or np.any(points[:, 1] < VIEW_MARGIN_PX)
        or np.any(points[:, 1] > height - VIEW_MARGIN_PX)
    )


def transform_from_bounds(frame_shape, bounds_min, bounds_max):
    """把原图坐标包围盒映射到固定输出画布。"""
    height, width = frame_shape[:2]
    bbox_size = np.maximum(bounds_max - bounds_min, 1.0)
    scale = min(float(width) / bbox_size[0], float(height) / bbox_size[1], 1.0)
    scale = max(MIN_VIEW_SCALE, float(scale))

    bbox_center = (bounds_min + bounds_max) * 0.5
    canvas_center = np.array([width * 0.5, height * 0.5], dtype=float)
    offset = canvas_center - scale * bbox_center
    return scale, offset


def ensure_lost_session_bounds(view_state: ViewState, frame_shape):
    """进入一轮视觉丢失显示后，初始化只扩不缩的可见范围。"""
    if view_state.lost_session_active and view_state.bounds_min is not None and view_state.bounds_max is not None:
        return

    bounds_min, bounds_max = current_visible_bounds(frame_shape, view_state)
    view_state.lost_session_active = True
    view_state.bounds_min = bounds_min
    view_state.bounds_max = bounds_max
    view_state.target_scale, view_state.target_offset = transform_from_bounds(
        frame_shape,
        bounds_min,
        bounds_max,
    )


def expand_lost_session_view(view_state: ViewState, frame_shape, pose):
    """
    视觉丢失期间只扩大视野范围。

    如果小车向回开但尚未重新识别 Tag，bounds 不收缩，因此灰色未知区域不会反复一缩一放。
    """
    ensure_lost_session_bounds(view_state, frame_shape)
    if pose is None or not pose_near_canvas_edge(pose, frame_shape, view_state):
        return

    points = project_axis_points(pose)
    if not np.all(np.isfinite(points)):
        return

    margin_mapped_to_image = VIEW_MARGIN_PX / max(view_state.target_scale, MIN_VIEW_SCALE)
    pose_min = np.min(points, axis=0) - margin_mapped_to_image
    pose_max = np.max(points, axis=0) + margin_mapped_to_image
    view_state.bounds_min = np.minimum(view_state.bounds_min, pose_min)
    view_state.bounds_max = np.maximum(view_state.bounds_max, pose_max)
    view_state.target_scale, view_state.target_offset = transform_from_bounds(
        frame_shape,
        view_state.bounds_min,
        view_state.bounds_max,
    )


def update_view_state(view_state: ViewState, frame_shape, visual_valid, estimated_pose):
    if visual_valid:
        view_state.lost_session_active = False
        view_state.bounds_min = None
        view_state.bounds_max = None
        desired_scale = 1.0
        desired_offset = np.zeros(2, dtype=float)
        view_state.target_scale = desired_scale
        view_state.target_offset = desired_offset.copy()
        alpha = 0.12
    elif estimated_pose is not None:
        expand_lost_session_view(view_state, frame_shape, estimated_pose)
        desired_scale = view_state.target_scale
        desired_offset = view_state.target_offset
        alpha = 0.22
    elif view_state.lost_session_active:
        desired_scale = view_state.target_scale
        desired_offset = view_state.target_offset
        alpha = 0.08
    else:
        desired_scale = view_state.scale
        desired_offset = view_state.offset.copy()
        alpha = 0.08

    if not view_state.initialized:
        view_state.scale = desired_scale
        view_state.offset = desired_offset.copy()
        view_state.initialized = True
        return

    view_state.scale = (1.0 - alpha) * view_state.scale + alpha * desired_scale
    view_state.offset = (1.0 - alpha) * view_state.offset + alpha * desired_offset

    if visual_valid and abs(view_state.scale - 1.0) < 0.003 and np.linalg.norm(view_state.offset) < 1.5:
        view_state.scale = 1.0
        view_state.offset[:] = 0.0


def paste_transformed_frame(canvas, frame, view_state: ViewState):
    height, width = frame.shape[:2]
    scaled_w = max(1, int(round(width * view_state.scale)))
    scaled_h = max(1, int(round(height * view_state.scale)))
    resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    x0 = int(round(view_state.offset[0]))
    y0 = int(round(view_state.offset[1]))
    x1 = x0 + scaled_w
    y1 = y0 + scaled_h

    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(canvas.shape[1], x1)
    dst_y1 = min(canvas.shape[0], y1)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return

    src_x0 = dst_x0 - x0
    src_y0 = dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]


def draw_fov_border(canvas, frame_shape, view_state: ViewState):
    height, width = frame_shape[:2]
    corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=float,
    )
    transformed = np.round(transform_points(corners, view_state)).astype(int)
    cv2.polylines(canvas, [transformed], True, FOV_BORDER_COLOR, 1, cv2.LINE_AA)


def draw_text_with_background(img, text, org, scale=0.55, color=TEXT_COLOR, bg=TEXT_BG, thickness=1):
    x, y = int(org[0]), int(org[1])
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(
        img,
        (x - 4, y - text_h - 5),
        (x + text_w + 4, y + baseline + 4),
        bg,
        -1,
    )
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_pose_axes(canvas, pose: PoseObservation, view_state: ViewState, label_color, text_overlay_enabled=True):
    points = project_axis_points(pose)
    if not np.all(np.isfinite(points)):
        return

    points = np.round(transform_points(points, view_state)).astype(int)
    origin = tuple(points[0])
    cv2.line(canvas, origin, tuple(points[1]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, origin, tuple(points[2]), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(canvas, origin, tuple(points[3]), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(canvas, origin, 4, label_color, -1, cv2.LINE_AA)
    if not text_overlay_enabled:
        return
    draw_text_with_background(
        canvas,
        pose.source,
        (origin[0] + 8, origin[1] - 8),
        scale=0.48,
        color=label_color,
        bg=(45, 45, 45),
        thickness=1,
    )


def draw_tag_annotations(frame, visual_pose: PoseObservation):
    if visual_pose.image_points is None:
        return

    points = visual_pose.image_points.astype(int)
    for i in range(4):
        cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % 4]), (0, 255, 0), 2, cv2.LINE_AA)

    if visual_pose.tag_center is not None:
        center = tuple(np.round(visual_pose.tag_center).astype(int))
        cv2.circle(frame, center, 5, (0, 0, 255), -1, cv2.LINE_AA)


def draw_overlay(
    canvas,
    source,
    vehicle_state,
    frame_aligner,
    errors: ErrorStats,
    run_stats: RunStats,
    output_path,
    recording_started,
):
    now = time.time()
    vehicle_text = "vehicle=none"
    if vehicle_state is not None:
        vehicle_text = (
            f"vehicle=ok age={now - vehicle_state.timestamp:.2f}s "
            f"yaw={vehicle_state.yaw:+.1f}deg"
        )

    align_text = "align=waiting"
    if frame_aligner.initialized:
        align_text = (
            f"align=ok theta={math.degrees(frame_aligner.theta):+.1f}deg "
            f"t=({frame_aligner.translation[0]:+.2f},{frame_aligner.translation[1]:+.2f})m"
        )

    current_error_text = "XY err now=n/a | yaw err now=n/a"
    if errors.last_pos_error_m is not None:
        current_error_text = (
            f"XY err now={errors.last_pos_error_m:.3f}m | "
            f"yaw err now={errors.last_yaw_error_deg:.1f}deg"
        )

    stats_text = "XY mean/RMSE/max=n/a | yaw mean/max=n/a"
    if errors.count:
        stats_text = (
            f"XY mean/RMSE/max={errors.pos_mean:.3f}/{errors.pos_rmse:.3f}/{errors.pos_max:.3f}m | "
            f"yaw mean/max={errors.yaw_abs_mean:.1f}/{errors.yaw_abs_max:.1f}deg"
        )

    lines = [
        f"source={source}",
        vehicle_text,
        align_text,
        current_error_text,
        stats_text,
        f"frames={run_stats.frame_count} lost={run_stats.visual_lost_frames} ugv_est={run_stats.ugv_pose_est_frames}",
    ]
    if output_path is not None and recording_started:
        lines.append(f"mp4={Path(output_path).name}")
    elif output_path is not None:
        lines.append("mp4=waiting first vision")

    scale = 0.47
    thickness = 1
    line_gap = 19
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for line in lines]
    block_w = max(w for w, _ in sizes) + 14
    block_h = line_gap * len(lines) + 10
    x0 = max(8, canvas.shape[1] - block_w - 8)
    y0 = 8
    cv2.rectangle(canvas, (x0, y0), (x0 + block_w, y0 + block_h), TEXT_BG, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + block_w, y0 + block_h), (80, 80, 80), 1)

    y = y0 + 20
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x0 + 7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def render_canvas(
    frame,
    visual_pose,
    estimated_pose,
    source,
    view_state,
    vehicle_state,
    frame_aligner,
    errors,
    run_stats,
    output_path,
    recording_started=False,
    text_overlay_enabled=True,
    color_marker_debug_overlay_enabled=False,
):
    frame_for_canvas = frame.copy()

    if visual_pose is not None and visual_pose.source == "VISION_TAG":
        draw_tag_annotations(frame_for_canvas, visual_pose)
    elif color_marker_debug_overlay_enabled and visual_pose is not None and visual_pose.color_observation is not None:
        draw_color_marker_debug(frame_for_canvas, visual_pose.color_observation)

    update_view_state(
        view_state,
        frame.shape,
        visual_valid=visual_pose is not None,
        estimated_pose=estimated_pose,
    )

    canvas = np.full(frame.shape, UNKNOWN_GRAY, dtype=np.uint8)
    paste_transformed_frame(canvas, frame_for_canvas, view_state)
    draw_fov_border(canvas, frame.shape, view_state)

    if visual_pose is not None:
        draw_pose_axes(canvas, visual_pose, view_state, VISION_AXIS_LABEL_COLOR, text_overlay_enabled)
    elif estimated_pose is not None:
        draw_pose_axes(canvas, estimated_pose, view_state, EST_AXIS_LABEL_COLOR, text_overlay_enabled)

    if text_overlay_enabled:
        draw_overlay(
            canvas,
            source,
            vehicle_state,
            frame_aligner,
            errors,
            run_stats,
            output_path,
            recording_started,
        )
    return canvas


def install_stop_handlers(stop_event):
    """注册 Ctrl+C、kill 和 SSH stdin 退出入口，保证能进入 finally 保存视频。"""
    def request_stop(signum=None, _frame=None):
        if signum is None:
            logger.info("收到停止输入，准备退出")
        else:
            logger.info(f"收到退出信号 {signum}，准备退出")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def stdin_watcher():
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":
                return
            if line.strip().lower() in ("q", "quit", "exit", "stop"):
                request_stop()
                return

    if sys.stdin is not None and sys.stdin.isatty():
        thread = threading.Thread(target=stdin_watcher, name="VisualizationStopInput", daemon=True)
        thread.start()


def print_summary(run_stats: RunStats, errors: ErrorStats, output_path, recording_started, elapsed_s):
    print("=" * 72)
    print("FrameAligner 对齐可视化脚本结束")
    print(f"总帧数: {run_stats.frame_count}")
    print(f"平均处理帧率: {run_stats.frame_count / max(elapsed_s, 1e-6):.2f} fps")
    print(f"视觉丢失帧数: {run_stats.visual_lost_frames}")
    print(f"UGV_POSE_EST 帧数: {run_stats.ugv_pose_est_frames}")
    if errors.count:
        print(
            "位置误差 XY mean/RMSE/max: "
            f"{errors.pos_mean:.4f} / {errors.pos_rmse:.4f} / {errors.pos_max:.4f} m"
        )
        print(
            "姿态误差 yaw mean/max: "
            f"{errors.yaw_abs_mean:.2f} / {errors.yaw_abs_max:.2f} deg"
        )
        print(f"误差样本数: {errors.count}")
    else:
        print("位置/姿态误差: 无有效样本（需要视觉有效且车端估计可用）")
    if output_path is not None and recording_started:
        print(f"MP4 输出: {output_path}")
    elif output_path is not None:
        print("MP4 输出: 未触发录制（本次未观测到视觉标记）")
    else:
        print("MP4 输出: 已关闭")
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="FrameAligner 车端坐标系到无人机坐标系在线对齐可视化脚本。")
    parser.add_argument("--udp-ip", default=DEFAULT_UDP_IP, help="UDP 图传接收端 IP。")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_PORT, help="UDP 图传端口。")
    parser.add_argument("--jpeg-quality", type=int, default=35, help="UDP JPEG 压缩质量，0-100。")
    parser.add_argument("--output", help="MP4 输出路径；默认写入 image_output/video/。")
    parser.add_argument("--fps", type=float, default=30.0, help="相机请求帧率和 MP4 保存帧率。")
    parser.add_argument("--duration", type=float, default=0.0, help="运行时长，秒；0 表示手动 Ctrl+C/窗口按键结束。")
    parser.add_argument("--source", choices=("csi", "camera"), default="csi", help="输入源类型。")
    parser.add_argument("--camera-index", type=int, default=0, help="普通 USB 摄像头编号。")
    parser.add_argument("--no-udp", action="store_true", help="关闭 UDP 图传。")
    parser.add_argument("--preview", action="store_true", help="在本机显示 OpenCV 预览窗口。")
    parser.add_argument("--no-save-video", action="store_true", help="不保存 MP4 视频。")
    parser.add_argument("--fallback-max-s", type=float, default=5.0, help="兼容保留参数；当前不再用于停止车端位姿估计。")
    parser.add_argument("--text-overlay", dest="text_overlay", action="store_true", default=TEXT_OVERLAY_ENABLED, help="显示右上角文字指标和坐标架文字标签。")
    parser.add_argument("--no-text-overlay", dest="text_overlay", action="store_false", help="隐藏文字叠加层，只保留画面、灰区和坐标架。")
    parser.add_argument("--color-marker-debug-overlay", dest="color_marker_debug_overlay", action="store_true", default=COLOR_MARKER_DEBUG_OVERLAY_ENABLED, help="显示彩色备用视觉的检测框和文字调试叠加层。")
    parser.add_argument("--no-color-marker-debug-overlay", dest="color_marker_debug_overlay", action="store_false", help="隐藏彩色备用视觉的检测框和文字调试叠加层。")
    parser.add_argument("--aligner-map", dest="aligner_map", action="store_true", default=ALIGNMENT_MAP_ENABLED, help="在相机画面下方显示 FrameAligner 对齐过程地图画布。")
    parser.add_argument("--no-aligner-map", dest="aligner_map", action="store_false", help="隐藏 FrameAligner 对齐过程地图画布。")
    parser.add_argument("--color-marker", dest="color_marker", action="store_true", default=True, help="启用彩色备用 PnP。")
    parser.add_argument("--no-color-marker", dest="color_marker", action="store_false", help="关闭彩色备用 PnP。")

    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor-id。")
    parser.add_argument("--capture-width", type=int, default=1920, help="CSI 采集宽度。")
    parser.add_argument("--capture-height", type=int, default=1080, help="CSI 采集高度。")
    parser.add_argument("--display-width", type=int, default=960, help="CSI 输出宽度。")
    parser.add_argument("--display-height", type=int, default=540, help="CSI 输出高度。")
    parser.add_argument("--flip-method", type=int, default=2, help="CSI nvvidconv flip-method。")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = None if args.no_save_video else build_output_path(args)
    stop_event = threading.Event()
    install_stop_handlers(stop_event)

    cap, capture_description = open_capture(args)
    detector = init_detector()
    frame_aligner = FrameAligner(yaw_alpha=ALIGN_YAW_ALPHA, pos_alpha=ALIGN_POS_ALPHA)
    vehicle_receiver = VehicleStateReceiver()
    vehicle_receiver.start()
    vehicle_cache = vehicle_receiver.cache

    sender = None
    if not args.no_udp:
        sender = VideoSender(
            dest_ip=args.udp_ip,
            dest_port=args.udp_port,
            jpeg_quality=args.jpeg_quality,
        )
        sender.start()

    writer = None
    recording_started = False
    view_state = ViewState()
    alignment_map = AlignmentMapRenderer(enabled=args.aligner_map)
    errors = ErrorStats()
    run_stats = RunStats()
    last_visual_pose = None
    start_monotonic = time.monotonic()

    logger.info("FrameAligner 对齐可视化脚本已启动")
    logger.info(capture_description)
    logger.info(f"UDP 图传: {'关闭' if sender is None else f'{args.udp_ip}:{args.udp_port}'}")
    logger.info(f"MP4 保存: {'关闭' if output_path is None else str(output_path) + '（首次视觉观测后开始写入）'}")
    logger.info(f"文字叠加层: {'开启' if args.text_overlay else '关闭'}")
    logger.info(f"彩色标记调试叠加层: {'开启' if args.color_marker_debug_overlay else '关闭'}")
    logger.info(f"FrameAligner 地图画布: {'开启' if args.aligner_map else '关闭'}")
    logger.info("结束方式：按 Ctrl+C，或在 SSH 终端输入 q 后回车；如果启用 --preview，也可在窗口中按 q 或 Esc。")

    if args.preview:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("读取相机画面失败，跳过本帧")
                time.sleep(0.02)
                continue

            now_wall = time.time()
            run_stats.frame_count += 1
            vehicle_state = vehicle_cache.get_latest(now=now_wall, max_age=VEHICLE_STATE_TIMEOUT_S)

            visual_pose = detect_visual_observation(frame, detector, args.color_marker)
            estimated_pose = estimate_pose_from_vehicle(
                frame_aligner,
                vehicle_state,
                last_visual_pose,
            )

            if visual_pose is not None and estimated_pose is not None:
                errors.update(visual_pose, estimated_pose)

            if visual_pose is not None:
                source = visual_pose.source
                update_alignment(frame_aligner, vehicle_state, visual_pose)
                last_visual_pose = visual_pose
            else:
                run_stats.visual_lost_frames += 1
                if estimated_pose is not None:
                    source = estimated_pose.source
                    run_stats.ugv_pose_est_frames += 1
                else:
                    source = "NO_TARGET"

            if visual_pose is not None and output_path is not None and not recording_started:
                recording_started = True
                logger.info(f"首次视觉观测成功，开始录制 MP4: {output_path}")

            alignment_map.update(now_wall, vehicle_state, visual_pose, frame_aligner)

            canvas = render_canvas(
                frame,
                visual_pose,
                estimated_pose,
                source,
                view_state,
                vehicle_state,
                frame_aligner,
                errors,
                run_stats,
                output_path,
                recording_started=recording_started,
                text_overlay_enabled=args.text_overlay,
                color_marker_debug_overlay_enabled=args.color_marker_debug_overlay,
            )
            canvas = alignment_map.compose(canvas, now_wall, frame_aligner)

            if recording_started and writer is None and output_path is not None:
                height, width = canvas.shape[:2]
                writer = make_writer(output_path, (width, height), args.fps)
            if recording_started and writer is not None:
                writer.write(canvas)

            if sender is not None:
                sender.send_frame(canvas)

            if args.preview:
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("用户通过预览窗口结束演示")
                    stop_event.set()

            if args.duration > 0.0 and time.monotonic() - start_monotonic >= args.duration:
                logger.info("达到指定运行时长，结束演示")
                stop_event.set()

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，准备退出")
    finally:
        elapsed_s = time.monotonic() - start_monotonic
        if sender is not None:
            sender.stop()
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print_summary(run_stats, errors, output_path, recording_started, elapsed_s)


if __name__ == "__main__":
    main()
