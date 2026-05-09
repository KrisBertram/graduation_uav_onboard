#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考轨迹跟踪实时可视化演示脚本。

该脚本只做感知、估计和画面输出：
- 读取下视相机画面；
- 接收无人车 TCP 位姿；
- 视觉有效时复用 AprilTag/彩色备用 PnP 和 FrameAligner；
- 复用 TargetEstimator、ReferenceTrajectory 和 build_tracking_command()；
- 在相机画面下方绘制参考轨迹地图画布；
- 通过 UDP 图传输出带叠加信息的画面，同时可保存 MP4。

安全边界：本脚本不连接飞控，不发送任何飞控控制指令。
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
from typing import Dict, List, Optional

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
from uav_core.frame_alignment import FrameAligner  # noqa: E402
from uav_core.reference_tracking import ReferenceTrajectory, TargetEstimator  # noqa: E402
from uav_core.tracking_fusion import TrackingFusionConfig, build_tracking_command  # noqa: E402
from uav_core.vehicle_state import VehicleStateReceiver  # noqa: E402
from uav_core.visual_control import (  # noqa: E402
    estimate_tag_forward_yaw_body,
    pnp_to_body_xy,
)
from utils.udp_video_sender import DEFAULT_PORT, VideoSender  # noqa: E402


WINDOW_NAME = "Reference Tracking Visualization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "image_output" / "video"
DEFAULT_UDP_IP = "10.105.26.61"

TEXT_OVERLAY_ENABLED = False
MAP_CANVAS_ENABLED = True

TAG_FORWARD_AXIS = "+Y"
VEHICLE_STATE_TIMEOUT_S = 0.3
ALIGN_YAW_ALPHA = 0.05
ALIGN_POS_ALPHA = 0.05

DEFAULT_LOOKAHEAD_TIME_S = 0.5
DEFAULT_MAX_REF_SPEED_MPS = 0.8
DEFAULT_MAX_CMD_OFFSET_M = 1.2
DEFAULT_PREDICT_TIME_S = 1.0
DEFAULT_UGV_FALLBACK_MAX_S = 5.0

CAMERA_CMD_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (35, 35, 35)
TAG_AXIS_LABEL_COLOR = (0, 255, 0)
COLOR_AXIS_LABEL_COLOR = (255, 255, 0)

MAP_BG = (24, 27, 31)
MAP_GRID = (62, 67, 72)
MAP_AXIS = (245, 245, 245)
MAP_MEASUREMENT = (0, 230, 80)
MAP_ESTIMATOR = (240, 240, 240)
MAP_FUTURE = (0, 165, 255)
MAP_REFERENCE = (255, 230, 0)
MAP_COMMAND = (255, 0, 255)
MAP_LIMIT = (115, 115, 120)
MAP_VELOCITY = (80, 210, 255)


@dataclass
class PoseObservation:
    """视觉得到的 Tag/降落点位姿。"""

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
class RunStats:
    """整次演示的统计信息。"""

    frame_count: int = 0
    visual_lost_frames: int = 0
    tracking_active_frames: int = 0
    cmd_norm_sum: float = 0.0
    cmd_norm_max: float = 0.0
    source_counts: Dict[str, int] = field(default_factory=dict)

    def record_source(self, source):
        self.source_counts[source] = self.source_counts.get(source, 0) + 1

    def record_command(self, cmd_body):
        self.tracking_active_frames += 1
        cmd_norm = float(np.linalg.norm(cmd_body))
        self.cmd_norm_sum += cmd_norm
        self.cmd_norm_max = max(self.cmd_norm_max, cmd_norm)

    @property
    def cmd_norm_mean(self):
        if self.tracking_active_frames <= 0:
            return None
        return self.cmd_norm_sum / self.tracking_active_frames


@dataclass
class TrackingMapRenderer:
    """下方俯视地图画布：展示参考轨迹生成链路。"""

    enabled: bool = MAP_CANVAS_ENABLED
    max_history: int = 320
    measurement_history: List[np.ndarray] = field(default_factory=list)
    future_history: List[np.ndarray] = field(default_factory=list)
    reference_history: List[np.ndarray] = field(default_factory=list)
    command_history: List[np.ndarray] = field(default_factory=list)
    estimator_history: List[np.ndarray] = field(default_factory=list)
    source: str = "LOST"
    measurement_xy: Optional[np.ndarray] = None
    estimator_xy: Optional[np.ndarray] = None
    future_xy: Optional[np.ndarray] = None
    reference_xy: Optional[np.ndarray] = None
    command_xy: Optional[np.ndarray] = None
    fused_vel: Optional[np.ndarray] = None
    used_vehicle_vel: bool = False
    vehicle_pose_status: str = "off"
    residual_m: Optional[float] = None

    def update(self, tracking_result, target_estimator, source, config):
        if not self.enabled:
            return

        self.source = source
        self.measurement_xy = None
        self.future_xy = None
        self.reference_xy = None
        self.command_xy = None
        self.fused_vel = None
        self.used_vehicle_vel = False
        self.vehicle_pose_status = "off"
        self.residual_m = None

        if target_estimator.initialized:
            self.estimator_xy = target_estimator.pos.copy()
            self._append(self.estimator_history, self.estimator_xy)
        else:
            self.estimator_xy = None

        if tracking_result is None:
            return

        if tracking_result.target_xy is not None:
            self.measurement_xy = np.asarray(tracking_result.target_xy, dtype=float).copy()
            self._append(self.measurement_history, self.measurement_xy)

        self.future_xy = np.asarray(tracking_result.future_xy, dtype=float).copy()
        self.reference_xy = np.asarray(tracking_result.ref_xy, dtype=float).copy()
        self.command_xy = np.asarray(tracking_result.cmd_body, dtype=float).copy()
        self.fused_vel = np.asarray(tracking_result.fused_vel, dtype=float).copy()
        self.used_vehicle_vel = bool(tracking_result.used_vehicle_vel)
        self.vehicle_pose_status = vehicle_pose_status(tracking_result, config)
        self.residual_m = tracking_result.vehicle_visual_residual_m

        self._append(self.future_history, self.future_xy)
        self._append(self.reference_history, self.reference_xy)
        self._append(self.command_history, self.command_xy)

    def compose(self, camera_canvas, config):
        if not self.enabled:
            return camera_canvas
        map_canvas = self.render_map_canvas(camera_canvas.shape, config)
        return np.vstack([camera_canvas, map_canvas])

    def render_map_canvas(self, frame_shape, config):
        height, width = frame_shape[:2]
        canvas = np.full((height, width, 3), MAP_BG, dtype=np.uint8)
        x0, y0 = 0, 0
        x1, y1 = width - 1, height - 1
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (95, 100, 105), 1, cv2.LINE_AA)

        points = self._map_points(config)
        scale, origin_px = self._map_transform(points, width, height)
        self._draw_grid(canvas, origin_px)
        cmd_limit_m = current_cmd_limit(config, self.source)
        self._draw_limit_circle(canvas, origin_px, scale, cmd_limit_m)
        self._draw_axes(canvas, origin_px, scale)

        self._draw_polyline(canvas, self.measurement_history, origin_px, scale, MAP_MEASUREMENT, dotted=True, thickness=2)
        self._draw_polyline(canvas, self.estimator_history, origin_px, scale, MAP_ESTIMATOR, dotted=True, thickness=1)
        self._draw_polyline(canvas, self.future_history, origin_px, scale, MAP_FUTURE, dotted=False, thickness=2)
        self._draw_polyline(canvas, self.reference_history, origin_px, scale, MAP_REFERENCE, dotted=False, thickness=3)
        self._draw_polyline(canvas, self.command_history, origin_px, scale, MAP_COMMAND, dotted=False, thickness=2)

        self._draw_current_points(canvas, origin_px, scale)
        self._draw_legend(canvas, config)
        return canvas

    def _append(self, history, point):
        history.append(np.asarray(point, dtype=float).copy())
        if len(history) > self.max_history:
            del history[0:len(history) - self.max_history]

    def _map_points(self, config):
        points = [np.array([0.0, 0.0], dtype=float)]
        radius = max(float(config.max_cmd_offset_m), float(config.ugv_fallback_max_cmd_offset_m), 0.5)
        points.extend(
            [
                np.array([radius, radius], dtype=float),
                np.array([-radius, -radius], dtype=float),
            ]
        )
        points.extend(self.measurement_history)
        points.extend(self.estimator_history)
        points.extend(self.future_history)
        points.extend(self.reference_history)
        points.extend(self.command_history)
        for point in (
            self.measurement_xy,
            self.estimator_xy,
            self.future_xy,
            self.reference_xy,
            self.command_xy,
        ):
            if point is not None:
                points.append(point)
        return points

    def _map_transform(self, points, width, height):
        arr = np.asarray(points, dtype=float)
        min_xy = np.min(arr, axis=0)
        max_xy = np.max(arr, axis=0)
        min_xy = np.minimum(min_xy, np.array([-0.8, -0.8]))
        max_xy = np.maximum(max_xy, np.array([0.8, 0.8]))

        size = np.maximum(max_xy - min_xy, 0.5)
        padding = 70
        usable_w = max(1, width - 2 * padding)
        usable_h = max(1, height - 2 * padding)
        # 地图中 +X/forward 向上，+Y/right 向右。
        scale = min(usable_h / size[0], usable_w / size[1])
        center_world = (min_xy + max_xy) * 0.5
        center_px = np.array([width * 0.5, height * 0.54], dtype=float)
        origin_px = np.array(
            [
                center_px[0] - center_world[1] * scale,
                center_px[1] + center_world[0] * scale,
            ],
            dtype=float,
        )
        return scale, origin_px

    def _to_px(self, xy, origin_px, scale):
        xy = np.asarray(xy, dtype=float)
        return (
            int(round(origin_px[0] + xy[1] * scale)),
            int(round(origin_px[1] - xy[0] * scale)),
        )

    def _draw_grid(self, canvas, origin_px):
        height, width = canvas.shape[:2]
        for x in range(0, width, 60):
            cv2.line(canvas, (x, 0), (x, height - 1), MAP_GRID, 1, cv2.LINE_AA)
        for y in range(0, height, 60):
            cv2.line(canvas, (0, y), (width - 1, y), MAP_GRID, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(origin_px[0]), int(origin_px[1])), 7, MAP_AXIS, -1, cv2.LINE_AA)

    def _draw_limit_circle(self, canvas, origin_px, scale, radius_m):
        radius_px = int(round(max(0.0, radius_m) * scale))
        if radius_px > 2:
            cv2.circle(canvas, tuple(np.round(origin_px).astype(int)), radius_px, MAP_LIMIT, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "Max Cmd Radius",
                (int(origin_px[0]) + radius_px + 8, int(origin_px[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                MAP_LIMIT,
                1,
                cv2.LINE_AA,
            )

    def _draw_axes(self, canvas, origin_px, scale):
        axis_len_px = int(np.clip(0.48 * scale, 35, 95))
        origin = tuple(np.round(origin_px).astype(int))
        x_end = (origin[0], origin[1] - axis_len_px)
        y_end = (origin[0] + axis_len_px, origin[1])
        cv2.arrowedLine(canvas, origin, x_end, MAP_AXIS, 2, cv2.LINE_AA, tipLength=0.22)
        cv2.arrowedLine(canvas, origin, y_end, MAP_AXIS, 2, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(canvas, "UAV", (origin[0] + 8, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, MAP_AXIS, 2, cv2.LINE_AA)
        cv2.putText(canvas, "+X", (x_end[0] + 6, x_end[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MAP_AXIS, 1, cv2.LINE_AA)
        cv2.putText(canvas, "+Y", (y_end[0] + 8, y_end[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MAP_AXIS, 1, cv2.LINE_AA)

    def _draw_polyline(self, canvas, history, origin_px, scale, color, dotted=False, thickness=2):
        if len(history) < 2:
            return
        points = [self._to_px(point, origin_px, scale) for point in history]
        for idx in range(1, len(points)):
            if dotted and idx % 2 == 0:
                continue
            cv2.line(canvas, points[idx - 1], points[idx], color, thickness, cv2.LINE_AA)

    def _draw_point_label(self, canvas, xy, origin_px, scale, color, label, radius=7, offset=(9, -9)):
        px = self._to_px(xy, origin_px, scale)
        cv2.circle(canvas, px, radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (px[0] + offset[0], px[1] + offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        return px

    def _draw_current_points(self, canvas, origin_px, scale):
        if self.measurement_xy is not None:
            self._draw_point_label(canvas, self.measurement_xy, origin_px, scale, MAP_MEASUREMENT, "Measurement")
        if self.estimator_xy is not None:
            self._draw_point_label(canvas, self.estimator_xy, origin_px, scale, MAP_ESTIMATOR, "Estimator", radius=6, offset=(9, 18))
        if self.future_xy is not None:
            self._draw_point_label(canvas, self.future_xy, origin_px, scale, MAP_FUTURE, "Future Point", radius=7)
        if self.reference_xy is not None:
            self._draw_point_label(canvas, self.reference_xy, origin_px, scale, MAP_REFERENCE, "Reference Point", radius=8, offset=(9, 20))
        if self.command_xy is not None:
            origin = self._to_px(np.zeros(2, dtype=float), origin_px, scale)
            cmd_px = self._to_px(self.command_xy, origin_px, scale)
            cv2.arrowedLine(canvas, origin, cmd_px, MAP_COMMAND, 4, cv2.LINE_AA, tipLength=0.18)
            cv2.putText(canvas, "Command", (cmd_px[0] + 9, cmd_px[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.62, MAP_COMMAND, 2, cv2.LINE_AA)
        if self.estimator_xy is not None and self.fused_vel is not None:
            vel_end = self.estimator_xy + 0.45 * self.fused_vel
            cv2.arrowedLine(
                canvas,
                self._to_px(self.estimator_xy, origin_px, scale),
                self._to_px(vel_end, origin_px, scale),
                MAP_VELOCITY,
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    def _draw_legend(self, canvas, config):
        residual = "n/a" if self.residual_m is None else f"{self.residual_m:.2f}m"
        vel_text = "on" if self.used_vehicle_vel else "off"
        lines = [
            f"Source: {self.source}",
            f"lookahead={config.lookahead_time_s:.2f}s  "
            f"ref_speed={getattr(config, 'max_ref_speed_mps', 0.0):.2f}m/s  "
            f"cmd_limit={current_cmd_limit(config, self.source):.2f}m",
            f"vehicle_vel={vel_text}  vehicle_pose={self.vehicle_pose_status}  residual={residual}",
            "green=Measurement  white=Estimator  orange=Future  cyan=Reference  magenta=Command",
        ]
        y = 30
        for line in lines:
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, TEXT_COLOR, 2, cv2.LINE_AA)
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
    return DEFAULT_OUTPUT_DIR / f"reference_tracking_visualization_{timestamp}.mp4"


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
                source="TAG",
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
        source="COLOR",
        rvec=pnp_rvec,
        tvec=pnp_tvec,
        body_xy=body_xy,
        yaw_body=yaw_body,
        axis_length_m=COLOR_MARKER_AXIS_LENGTH_M,
        z_m=float(pnp_z),
        color_observation=color_observation,
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


def draw_tag_annotations(frame, visual_pose: PoseObservation):
    if visual_pose.image_points is None:
        return

    points = visual_pose.image_points.astype(int)
    for i in range(4):
        cv2.line(frame, tuple(points[i]), tuple(points[(i + 1) % 4]), (0, 255, 0), 2, cv2.LINE_AA)

    if visual_pose.tag_center is not None:
        center = tuple(np.round(visual_pose.tag_center).astype(int))
        cv2.circle(frame, center, 5, (0, 0, 255), -1, cv2.LINE_AA)


def draw_pose_axes(canvas, pose: PoseObservation, label_color, text_overlay_enabled=True):
    points = project_axis_points(pose)
    if not np.all(np.isfinite(points)):
        return

    points = np.round(points).astype(int)
    origin = tuple(points[0])
    cv2.line(canvas, origin, tuple(points[1]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(canvas, origin, tuple(points[2]), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(canvas, origin, tuple(points[3]), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(canvas, origin, 4, label_color, -1, cv2.LINE_AA)
    if text_overlay_enabled:
        draw_text_with_background(
            canvas,
            pose.source,
            (origin[0] + 8, origin[1] - 8),
            scale=0.48,
            color=label_color,
            bg=(45, 45, 45),
            thickness=1,
        )


def body_xy_to_image_px(body_xy, frame_shape, pixels_per_m=120.0):
    height, width = frame_shape[:2]
    body_xy = np.asarray(body_xy, dtype=float)
    center = np.array([width * 0.5, height * 0.5], dtype=float)
    return (
        int(round(center[0] + body_xy[1] * pixels_per_m)),
        int(round(center[1] - body_xy[0] * pixels_per_m)),
    )


def draw_command_arrow(frame, tracking_result, config, text_overlay_enabled):
    if tracking_result is None:
        return

    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    cmd_body = np.asarray(tracking_result.cmd_body, dtype=float)
    endpoint = body_xy_to_image_px(cmd_body, frame.shape)
    cv2.arrowedLine(frame, center, endpoint, CAMERA_CMD_COLOR, 4, cv2.LINE_AA, tipLength=0.22)
    cv2.circle(frame, center, 5, CAMERA_CMD_COLOR, -1, cv2.LINE_AA)

    max_radius_px = int(round(current_cmd_limit(config, tracking_result.source) * 120.0))
    if max_radius_px > 3:
        cv2.circle(frame, center, max_radius_px, (170, 90, 170), 1, cv2.LINE_AA)

    if text_overlay_enabled:
        cmd_norm = float(np.linalg.norm(cmd_body))
        draw_text_with_background(
            frame,
            f"cmd dx={cmd_body[0]:+.2f} dy={cmd_body[1]:+.2f} | {cmd_norm:.2f}m",
            (endpoint[0] + 8, endpoint[1] - 8),
            scale=0.48,
            color=CAMERA_CMD_COLOR,
            bg=(45, 45, 45),
            thickness=1,
        )


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


def vehicle_pose_status(tracking_result, config):
    """把融合结果转换成显示用的车端位置状态。"""
    if tracking_result.source == "UGV_POSE":
        return "fallback"
    if tracking_result.fused_vehicle_pose:
        return "fused"
    residual = tracking_result.vehicle_visual_residual_m
    if residual is not None and residual > config.ugv_visual_reject_residual_m:
        return "rejected"
    return "off"


def residual_text(tracking_result):
    residual = tracking_result.vehicle_visual_residual_m if tracking_result is not None else None
    if residual is None:
        return "n/a"
    return f"{residual:.2f}m"


def current_cmd_limit(config, source):
    if source == "UGV_POSE":
        return float(config.ugv_fallback_max_cmd_offset_m)
    return float(config.max_cmd_offset_m)


def draw_overlay(
    canvas,
    source,
    vehicle_state,
    frame_aligner,
    tracking_result,
    config,
    run_stats,
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

    result_lines = [
        f"source={source}",
        f"lookahead={config.lookahead_time_s:.2f}s "
        f"ref_speed={getattr(config, 'max_ref_speed_mps', 0.0):.2f}m/s "
        f"cmd_limit={current_cmd_limit(config, source):.2f}m",
        vehicle_text,
        align_text,
        f"frames={run_stats.frame_count} lost={run_stats.visual_lost_frames} tracking={run_stats.tracking_active_frames}",
    ]
    if tracking_result is not None:
        veh_vel_text = "on" if tracking_result.used_vehicle_vel else "off"
        veh_pose_text = vehicle_pose_status(tracking_result, config)
        result_lines.extend(
            [
                f"vehicle_vel={veh_vel_text} vehicle_pose={veh_pose_text} residual={residual_text(tracking_result)}",
                f"future=({tracking_result.future_xy[0]:+.2f},{tracking_result.future_xy[1]:+.2f}) ref=({tracking_result.ref_xy[0]:+.2f},{tracking_result.ref_xy[1]:+.2f})",
            ]
        )
    if output_path is not None and recording_started:
        result_lines.append(f"mp4={Path(output_path).name}")
    elif output_path is not None:
        result_lines.append("mp4=waiting first vision")

    scale = 0.47
    thickness = 1
    line_gap = 19
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for line in result_lines]
    block_w = max(w for w, _ in sizes) + 14
    block_h = line_gap * len(result_lines) + 10
    x0 = max(8, canvas.shape[1] - block_w - 8)
    y0 = 8
    cv2.rectangle(canvas, (x0, y0), (x0 + block_w, y0 + block_h), TEXT_BG, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + block_w, y0 + block_h), (80, 80, 80), 1)

    y = y0 + 20
    for line in result_lines:
        cv2.putText(canvas, line, (x0 + 7, y), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
        y += line_gap


def render_camera_canvas(
    frame,
    visual_pose,
    tracking_result,
    source,
    vehicle_state,
    frame_aligner,
    config,
    run_stats,
    output_path,
    recording_started=False,
    text_overlay_enabled=True,
):
    canvas = frame.copy()

    if visual_pose is not None and visual_pose.source == "TAG":
        draw_tag_annotations(canvas, visual_pose)
    elif visual_pose is not None and visual_pose.color_observation is not None:
        draw_color_marker_debug(canvas, visual_pose.color_observation)

    if visual_pose is not None:
        label_color = TAG_AXIS_LABEL_COLOR if visual_pose.source == "TAG" else COLOR_AXIS_LABEL_COLOR
        draw_pose_axes(canvas, visual_pose, label_color, text_overlay_enabled)

    draw_command_arrow(canvas, tracking_result, config, text_overlay_enabled)

    if text_overlay_enabled:
        draw_overlay(
            canvas,
            source,
            vehicle_state,
            frame_aligner,
            tracking_result,
            config,
            run_stats,
            output_path,
            recording_started,
        )
    return canvas


def build_tracking_config(args):
    config = TrackingFusionConfig(
        lookahead_time_s=args.lookahead_time,
        max_cmd_offset_m=args.max_cmd_offset,
        ugv_pose_fallback_max_s=args.ugv_fallback_max_s,
        ugv_fallback_max_cmd_offset_m=min(args.max_cmd_offset, 0.8),
    )
    config.max_ref_speed_mps = args.max_ref_speed
    return config


def build_tracking_result(
    *,
    now,
    dt,
    visual_pose,
    vehicle_state,
    frame_aligner,
    target_estimator,
    reference_traj,
    config,
    last_visual_time,
    predict_time_s,
):
    tracking_frame = "local"
    drone_xy = np.zeros(2, dtype=float)
    drone_yaw = 0.0

    if visual_pose is not None:
        return build_tracking_command(
            now=now,
            dt=dt,
            tracking_frame=tracking_frame,
            target_estimator=target_estimator,
            reference_traj=reference_traj,
            config=config,
            body_xy=visual_pose.body_xy,
            drone_xy=drone_xy,
            drone_yaw=drone_yaw,
            vehicle_state=vehicle_state,
            frame_aligner=frame_aligner,
            tag_yaw_drone=visual_pose.yaw_body,
            visual_source=visual_pose.source,
            use_prediction=False,
        )

    tracking_result = build_tracking_command(
        now=now,
        dt=dt,
        tracking_frame=tracking_frame,
        target_estimator=target_estimator,
        reference_traj=reference_traj,
        config=config,
        body_xy=None,
        drone_xy=drone_xy,
        drone_yaw=drone_yaw,
        vehicle_state=vehicle_state,
        frame_aligner=frame_aligner,
        use_vehicle_pose_fallback=True,
        last_visual_time=last_visual_time,
    )
    if tracking_result is not None:
        return tracking_result

    if last_visual_time is None or now - last_visual_time >= predict_time_s:
        return None

    return build_tracking_command(
        now=now,
        dt=dt,
        tracking_frame=tracking_frame,
        target_estimator=target_estimator,
        reference_traj=reference_traj,
        config=config,
        body_xy=None,
        drone_xy=drone_xy,
        drone_yaw=drone_yaw,
        vehicle_state=vehicle_state,
        frame_aligner=frame_aligner,
        use_prediction=True,
    )


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
        thread = threading.Thread(target=stdin_watcher, name="ReferenceTrackingStopInput", daemon=True)
        thread.start()


def print_summary(run_stats: RunStats, output_path, recording_started, elapsed_s):
    print("=" * 72)
    print("参考轨迹跟踪可视化脚本结束")
    print(f"总帧数: {run_stats.frame_count}")
    print(f"平均处理帧率: {run_stats.frame_count / max(elapsed_s, 1e-6):.2f} fps")
    print(f"视觉丢失帧数: {run_stats.visual_lost_frames}")
    print(f"跟踪有效帧数: {run_stats.tracking_active_frames}")
    if run_stats.source_counts:
        source_text = ", ".join(f"{key}:{value}" for key, value in sorted(run_stats.source_counts.items()))
        print(f"source 统计: {source_text}")
    if run_stats.cmd_norm_mean is not None:
        print(f"控制指令范数 mean/max: {run_stats.cmd_norm_mean:.4f} / {run_stats.cmd_norm_max:.4f} m")
    else:
        print("控制指令范数: 无有效样本")
    if output_path is not None and recording_started:
        print(f"MP4 输出: {output_path}")
    elif output_path is not None:
        print("MP4 输出: 未触发录制（本次未观测到视觉标记）")
    else:
        print("MP4 输出: 已关闭")
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="参考轨迹跟踪实时可视化脚本。")
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
    parser.add_argument("--text-overlay", dest="text_overlay", action="store_true", default=TEXT_OVERLAY_ENABLED, help="显示相机画面右上角文字指标和坐标架文字标签。")
    parser.add_argument("--no-text-overlay", dest="text_overlay", action="store_false", help="隐藏相机画面文字叠加层。")
    parser.add_argument("--map-canvas", dest="map_canvas", action="store_true", default=MAP_CANVAS_ENABLED, help="在相机画面下方显示参考轨迹地图画布。")
    parser.add_argument("--no-map-canvas", dest="map_canvas", action="store_false", help="隐藏参考轨迹地图画布。")
    parser.add_argument("--color-marker", dest="color_marker", action="store_true", default=True, help="启用彩色备用 PnP。")
    parser.add_argument("--no-color-marker", dest="color_marker", action="store_false", help="关闭彩色备用 PnP。")

    parser.add_argument("--lookahead-time", type=float, default=DEFAULT_LOOKAHEAD_TIME_S, help="前视预测时间，秒。")
    parser.add_argument("--max-ref-speed", type=float, default=DEFAULT_MAX_REF_SPEED_MPS, help="参考点最大平滑速度，m/s。")
    parser.add_argument("--max-cmd-offset", type=float, default=DEFAULT_MAX_CMD_OFFSET_M, help="单帧水平控制指令限幅，m。")
    parser.add_argument("--predict-time", type=float, default=DEFAULT_PREDICT_TIME_S, help="无视觉且无 UGV fallback 时短时预测时长，秒。")
    parser.add_argument("--ugv-fallback-max-s", type=float, default=DEFAULT_UGV_FALLBACK_MAX_S, help="视觉丢失后允许 UGV_POSE fallback 的最长时间，秒。")

    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor-id。")
    parser.add_argument("--capture-width", type=int, default=1920, help="CSI 采集宽度。")
    parser.add_argument("--capture-height", type=int, default=1080, help="CSI 采集高度。")
    parser.add_argument("--display-width", type=int, default=960, help="CSI 输出宽度。")
    parser.add_argument("--display-height", type=int, default=540, help="CSI 输出高度。")
    parser.add_argument("--flip-method", type=int, default=2, help="CSI nvvidconv flip-method。")
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_tracking_config(args)
    output_path = None if args.no_save_video else build_output_path(args)
    stop_event = threading.Event()
    install_stop_handlers(stop_event)

    cap, capture_description = open_capture(args)
    detector = init_detector()
    frame_aligner = FrameAligner(yaw_alpha=ALIGN_YAW_ALPHA, pos_alpha=ALIGN_POS_ALPHA)
    target_estimator = TargetEstimator()
    reference_traj = ReferenceTrajectory(max_speed_mps=args.max_ref_speed)
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
    map_renderer = TrackingMapRenderer(enabled=args.map_canvas)
    run_stats = RunStats()
    last_visual_time = None
    last_loop_time = None
    start_monotonic = time.monotonic()

    logger.info("参考轨迹跟踪可视化脚本已启动")
    logger.info(capture_description)
    logger.info(f"UDP 图传: {'关闭' if sender is None else f'{args.udp_ip}:{args.udp_port}'}")
    logger.info(f"MP4 保存: {'关闭' if output_path is None else str(output_path) + '（首次视觉观测后开始写入）'}")
    logger.info(f"文字叠加层: {'开启' if args.text_overlay else '关闭'}")
    logger.info(f"参考轨迹地图画布: {'开启' if args.map_canvas else '关闭'}")
    logger.info(
        f"跟踪参数: lookahead={config.lookahead_time_s:.2f}s "
        f"max_ref_speed={args.max_ref_speed:.2f}m/s max_cmd={config.max_cmd_offset_m:.2f}m"
    )
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
            if last_loop_time is None:
                dt = 1.0 / max(args.fps, 1e-6)
            else:
                dt = max(1e-3, min(now_wall - last_loop_time, 0.5))
            last_loop_time = now_wall

            run_stats.frame_count += 1
            vehicle_state = vehicle_cache.get_latest(now=now_wall, max_age=VEHICLE_STATE_TIMEOUT_S)
            visual_pose = detect_visual_observation(frame, detector, args.color_marker)

            if visual_pose is not None:
                last_visual_time = now_wall
            else:
                run_stats.visual_lost_frames += 1

            tracking_result = build_tracking_result(
                now=now_wall,
                dt=dt,
                visual_pose=visual_pose,
                vehicle_state=vehicle_state,
                frame_aligner=frame_aligner,
                target_estimator=target_estimator,
                reference_traj=reference_traj,
                config=config,
                last_visual_time=last_visual_time,
                predict_time_s=args.predict_time,
            )
            source = tracking_result.source if tracking_result is not None else "LOST"
            run_stats.record_source(source)
            if tracking_result is not None:
                run_stats.record_command(tracking_result.cmd_body)

            if visual_pose is not None and output_path is not None and not recording_started:
                recording_started = True
                logger.info(f"首次视觉观测成功，开始录制 MP4: {output_path}")

            map_renderer.update(tracking_result, target_estimator, source, config)
            camera_canvas = render_camera_canvas(
                frame,
                visual_pose,
                tracking_result,
                source,
                vehicle_state,
                frame_aligner,
                config,
                run_stats,
                output_path,
                recording_started=recording_started,
                text_overlay_enabled=args.text_overlay,
            )
            canvas = map_renderer.compose(camera_canvas, config)

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
        print_summary(run_stats, output_path, recording_started, elapsed_s)


if __name__ == "__main__":
    main()
