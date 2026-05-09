#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参考轨迹跟踪实时可视化演示脚本。

该脚本只做感知、估计和画面输出：
- 读取下视相机画面；
- 接收无人车 TCP 位姿；
- 视觉有效时复用 AprilTag/彩色备用 PnP 和 FrameAligner；
- 复用 TargetEstimator、ReferenceTrajectory 和 build_tracking_command()；
- 在相机画面中按物理比例绘制参考轨迹关键点；
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
from typing import Dict, Optional

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
from utils.kb_TagVisualizer import TagVisualizer  # noqa: E402
from utils.udp_video_sender import DEFAULT_PORT, VideoSender  # noqa: E402


WINDOW_NAME = "Reference Tracking Visualization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "image_output" / "video"
DEFAULT_UDP_IP = "10.105.26.61"

TEXT_OVERLAY_ENABLED = False
CAMERA_TRACK_POINTS_ENABLED = True
COLOR_MARKER_DEBUG_OVERLAY_ENABLED = False

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
CAMERA_MEASUREMENT_COLOR = (0, 230, 80)
CAMERA_FUTURE_COLOR = (0, 165, 255)
CAMERA_REFERENCE_COLOR = (255, 230, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (35, 35, 35)
TAG_AXIS_LABEL_COLOR = (0, 255, 0)
COLOR_AXIS_LABEL_COLOR = (255, 255, 0)


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
class CameraPlaneScaleCache:
    """缓存相机画面中平台平面的米/像素比例。"""

    meter_per_pixel: Optional[float] = None
    source: str = "none"
    timestamp: Optional[float] = None

    def update(self, visual_pose: Optional[PoseObservation], now):
        if visual_pose is None:
            return

        meter_per_pixel = None
        source = None
        if visual_pose.source == "TAG" and visual_pose.image_points is not None and visual_pose.tag_id in TAG_SIZES:
            meter_per_pixel = compute_tag_meter_per_pixel(visual_pose)
            source = "tag"

        if meter_per_pixel is None:
            meter_per_pixel = compute_pinhole_meter_per_pixel(visual_pose)
            source = "pinhole"

        if meter_per_pixel is None:
            return

        self.meter_per_pixel = meter_per_pixel
        self.source = source
        self.timestamp = now

    @property
    def available(self):
        return self.meter_per_pixel is not None and self.meter_per_pixel > 0.0


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


def compute_tag_meter_per_pixel(visual_pose: PoseObservation):
    """优先使用 AprilTag 在画面中的真实边长比例。"""
    if visual_pose.image_points is None or visual_pose.tag_id not in TAG_SIZES:
        return None
    try:
        meter_per_pixel = TagVisualizer.compute_pixel_scale_on_tag(
            np.asarray(visual_pose.image_points, dtype=float),
            TAG_SIZES[visual_pose.tag_id],
        )
    except Exception as err:
        logger.warning(f"计算 AprilTag 平面比例失败: {err}")
        return None
    if not np.isfinite(meter_per_pixel) or meter_per_pixel <= 0.0:
        return None
    return float(meter_per_pixel)


def compute_pinhole_meter_per_pixel(visual_pose: PoseObservation):
    """彩色备用或无边框时，用当前高度和相机焦距估算米/像素。"""
    if visual_pose.z_m <= 0.0:
        return None
    fx = float(cameraMatrix[0, 0])
    fy = float(cameraMatrix[1, 1])
    if fx <= 0.0 or fy <= 0.0:
        return None
    meter_per_pixel = 0.5 * (visual_pose.z_m / fx + visual_pose.z_m / fy)
    if not np.isfinite(meter_per_pixel) or meter_per_pixel <= 0.0:
        return None
    return float(meter_per_pixel)


def project_body_xy_to_image(body_xy, frame_shape, meter_per_pixel):
    height, width = frame_shape[:2]
    body_xy = np.asarray(body_xy, dtype=float)
    center = np.array([width * 0.5, height * 0.5], dtype=float)
    raw_px = np.array(
        [
            center[0] + body_xy[1] / meter_per_pixel,
            center[1] - body_xy[0] / meter_per_pixel,
        ],
        dtype=float,
    )
    clipped_px = np.array(
        [
            np.clip(raw_px[0], 0, width - 1),
            np.clip(raw_px[1], 0, height - 1),
        ],
        dtype=float,
    )
    offscreen = bool(np.linalg.norm(raw_px - clipped_px) > 1e-6)
    return tuple(np.round(raw_px).astype(int)), tuple(np.round(clipped_px).astype(int)), offscreen


def draw_camera_track_point(frame, label, body_xy, scale_cache, color, text_overlay_enabled, radius=6):
    if not scale_cache.available:
        return None
    raw_px, clipped_px, offscreen = project_body_xy_to_image(
        body_xy,
        frame.shape,
        scale_cache.meter_per_pixel,
    )
    cv2.circle(frame, clipped_px, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, clipped_px, radius + 3, color, 1, cv2.LINE_AA)
    if text_overlay_enabled:
        suffix = " offscreen" if offscreen else ""
        draw_text_with_background(
            frame,
            f"{label}{suffix}",
            (clipped_px[0] + 8, clipped_px[1] - 8),
            scale=0.45,
            color=color,
            bg=(45, 45, 45),
            thickness=1,
        )
    return raw_px, clipped_px, offscreen


def draw_camera_tracking_points(frame, tracking_result, config, scale_cache, text_overlay_enabled, enabled=True):
    if not enabled or tracking_result is None or not scale_cache.available:
        return

    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    cmd_body = np.asarray(tracking_result.cmd_body, dtype=float)
    _, cmd_px, cmd_offscreen = project_body_xy_to_image(cmd_body, frame.shape, scale_cache.meter_per_pixel)
    cv2.arrowedLine(frame, center, cmd_px, CAMERA_CMD_COLOR, 4, cv2.LINE_AA, tipLength=0.22)
    cv2.circle(frame, center, 5, CAMERA_CMD_COLOR, -1, cv2.LINE_AA)

    max_radius_px = int(round(current_cmd_limit(config, tracking_result.source) / scale_cache.meter_per_pixel))
    if max_radius_px > 3:
        cv2.circle(frame, center, max_radius_px, (170, 90, 170), 1, cv2.LINE_AA)

    if tracking_result.target_xy is not None:
        draw_camera_track_point(
            frame,
            "Measurement",
            tracking_result.target_xy,
            scale_cache,
            CAMERA_MEASUREMENT_COLOR,
            text_overlay_enabled,
            radius=5,
        )
    draw_camera_track_point(
        frame,
        "Future Point",
        tracking_result.future_xy,
        scale_cache,
        CAMERA_FUTURE_COLOR,
        text_overlay_enabled,
        radius=6,
    )
    reference_draw = draw_camera_track_point(
        frame,
        "Reference Point",
        tracking_result.ref_xy,
        scale_cache,
        CAMERA_REFERENCE_COLOR,
        text_overlay_enabled,
        radius=8,
    )
    if reference_draw is not None:
        _, ref_px, _ = reference_draw
        cv2.arrowedLine(frame, center, ref_px, CAMERA_REFERENCE_COLOR, 3, cv2.LINE_AA, tipLength=0.18)

    if text_overlay_enabled:
        cmd_norm = float(np.linalg.norm(cmd_body))
        cmd_suffix = " offscreen" if cmd_offscreen else ""
        draw_text_with_background(
            frame,
            f"Command{cmd_suffix} dx={cmd_body[0]:+.2f} dy={cmd_body[1]:+.2f} | {cmd_norm:.2f}m",
            (cmd_px[0] + 8, cmd_px[1] - 8),
            scale=0.48,
            color=CAMERA_CMD_COLOR,
            bg=(45, 45, 45),
            thickness=1,
        )
        draw_text_with_background(
            frame,
            f"camera scale={scale_cache.meter_per_pixel * 1000.0:.2f} mm/px ({scale_cache.source})",
            (10, 28),
            scale=0.48,
            color=TEXT_COLOR,
            bg=TEXT_BG,
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
    scale_cache,
    recording_started=False,
    text_overlay_enabled=True,
    camera_track_points_enabled=True,
    color_marker_debug_overlay_enabled=False,
):
    canvas = frame.copy()

    if visual_pose is not None and visual_pose.source == "TAG":
        draw_tag_annotations(canvas, visual_pose)
    elif color_marker_debug_overlay_enabled and visual_pose is not None and visual_pose.color_observation is not None:
        draw_color_marker_debug(canvas, visual_pose.color_observation)

    if visual_pose is not None:
        label_color = TAG_AXIS_LABEL_COLOR if visual_pose.source == "TAG" else COLOR_AXIS_LABEL_COLOR
        draw_pose_axes(canvas, visual_pose, label_color, text_overlay_enabled)

    draw_camera_tracking_points(
        canvas,
        tracking_result,
        config,
        scale_cache,
        text_overlay_enabled,
        enabled=camera_track_points_enabled,
    )

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
    parser.add_argument("--camera-track-points", dest="camera_track_points", action="store_true", default=CAMERA_TRACK_POINTS_ENABLED, help="在相机画面中按物理比例绘制参考轨迹关键点。")
    parser.add_argument("--no-camera-track-points", dest="camera_track_points", action="store_false", help="隐藏相机画面中的参考轨迹关键点。")
    parser.add_argument("--color-marker-debug-overlay", dest="color_marker_debug_overlay", action="store_true", default=COLOR_MARKER_DEBUG_OVERLAY_ENABLED, help="显示彩色备用视觉的检测框和文字调试叠加层。")
    parser.add_argument("--no-color-marker-debug-overlay", dest="color_marker_debug_overlay", action="store_false", help="隐藏彩色备用视觉的检测框和文字调试叠加层。")
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
    scale_cache = CameraPlaneScaleCache()
    run_stats = RunStats()
    last_visual_time = None
    last_loop_time = None
    start_monotonic = time.monotonic()

    logger.info("参考轨迹跟踪可视化脚本已启动")
    logger.info(capture_description)
    logger.info(f"UDP 图传: {'关闭' if sender is None else f'{args.udp_ip}:{args.udp_port}'}")
    logger.info(f"MP4 保存: {'关闭' if output_path is None else str(output_path) + '（首次视觉观测后开始写入）'}")
    logger.info(f"文字叠加层: {'开启' if args.text_overlay else '关闭'}")
    logger.info(f"相机画面参考点: {'开启' if args.camera_track_points else '关闭'}")
    logger.info(f"彩色标记调试叠加层: {'开启' if args.color_marker_debug_overlay else '关闭'}")
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
                scale_cache.update(visual_pose, now_wall)
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

            canvas = render_camera_canvas(
                frame,
                visual_pose,
                tracking_result,
                source,
                vehicle_state,
                frame_aligner,
                config,
                run_stats,
                output_path,
                scale_cache,
                recording_started=recording_started,
                text_overlay_enabled=args.text_overlay,
                camera_track_points_enabled=args.camera_track_points,
                color_marker_debug_overlay_enabled=args.color_marker_debug_overlay,
            )

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
