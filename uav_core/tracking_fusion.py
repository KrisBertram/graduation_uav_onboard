"""
视觉观测、车端位姿和短时预测的参考轨迹融合。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from uav_core.reference_tracking import (
    body_to_local_xy,
    clamp_norm,
    local_to_body_xy,
)


@dataclass
class TrackingFusionConfig:
    """
    参考轨迹融合参数。

    实际调参入口在 main.py 顶部同名常量；这里保留默认值，便于离线假数据测试。
    参数分为四组：
    - lookahead/max_cmd：控制参考点预测和飞控水平指令限幅。
    - vision_vel/vehicle_vel：视觉有效或短时预测时的速度融合权重。
    - vehicle_pose_fusion：视觉有效时，是否以及何时融合车端绝对位置。
    - ugv_pose_*：视觉失效后，车端位姿参考点模式的持续时间、速度权重和指令限幅。
    推荐范围和现场调整方法见 main.py 的“视觉 + 车端位姿融合调参”注释块。
    """

    lookahead_time_s: float = 0.5
    max_cmd_offset_m: float = 1.2
    vision_vel_weight: float = 0.4
    vehicle_vel_weight: float = 0.6
    vehicle_pose_fusion_enabled: bool = True
    vehicle_pose_fallback_enabled: bool = True
    ugv_pose_fallback_max_s: float = 5.0
    ugv_visual_fuse_max_residual_m: float = 0.25
    ugv_visual_reject_residual_m: float = 0.60
    tag_visual_pos_weight: float = 0.85
    color_visual_pos_weight: float = 0.70
    ugv_pose_est_vel_weight: float = 0.2
    ugv_pose_vel_weight: float = 0.8
    ugv_fallback_max_cmd_offset_m: float = 0.8


@dataclass
class TrackingCommandResult:
    """融合后的水平跟踪控制结果。"""

    cmd_body: np.ndarray
    target_xy: Optional[np.ndarray]
    ref_xy: np.ndarray
    future_xy: np.ndarray
    fused_vel: np.ndarray
    source: str
    used_vehicle_vel: bool
    used_vehicle_pose: bool
    fused_vehicle_pose: bool
    vehicle_visual_residual_m: Optional[float]
    used_alignment: bool
    use_prediction: bool


def _vehicle_xy(vehicle_state):
    return np.array([vehicle_state.pos_x, vehicle_state.pos_y], dtype=float)


def _vehicle_velocity(vehicle_state, frame_aligner):
    return frame_aligner.vehicle_velocity_to_drone(
        vehicle_state.speed,
        vehicle_state.yaw_rad,
    )


def _visual_source_name(visual_source, fused):
    source = visual_source or "TAG"
    if source == "COLOR":
        return "COLOR_FUSED" if fused else "COLOR_ONLY"
    return "TAG_FUSED" if fused else "TAG_ONLY"


def _visual_pos_weight(config, visual_source):
    if visual_source == "COLOR":
        return config.color_visual_pos_weight
    return config.tag_visual_pos_weight


def _make_local_result(
    *,
    now,
    dt,
    target_xy,
    target_estimator,
    reference_traj,
    drone_xy,
    drone_yaw,
    vehicle_vel,
    config,
    source,
    used_vehicle_pose=False,
    fused_vehicle_pose=False,
    vehicle_visual_residual_m=None,
    used_alignment=False,
    use_prediction=False,
    vision_vel_weight=None,
    vehicle_vel_weight=None,
    max_cmd_offset_m=None,
):
    if target_xy is not None:
        target_estimator.update_measurement(target_xy, now)

    if vision_vel_weight is None:
        vision_vel_weight = config.vision_vel_weight
    if vehicle_vel_weight is None:
        vehicle_vel_weight = config.vehicle_vel_weight
    if max_cmd_offset_m is None:
        max_cmd_offset_m = config.max_cmd_offset_m

    future_xy, fused_vel, used_vehicle_vel = target_estimator.make_future_point(
        now,
        config.lookahead_time_s,
        vehicle_vel_xy=vehicle_vel,
        vision_weight=vision_vel_weight,
        vehicle_weight=vehicle_vel_weight,
    )
    if future_xy is None:
        return None

    ref_xy = reference_traj.update(future_xy, dt)
    cmd_local = clamp_norm(ref_xy - drone_xy, max_cmd_offset_m)
    cmd_body = local_to_body_xy(cmd_local, drone_yaw)

    return TrackingCommandResult(
        cmd_body=cmd_body,
        target_xy=target_xy,
        ref_xy=ref_xy,
        future_xy=future_xy,
        fused_vel=fused_vel,
        source=source,
        used_vehicle_vel=used_vehicle_vel,
        used_vehicle_pose=used_vehicle_pose,
        fused_vehicle_pose=fused_vehicle_pose,
        vehicle_visual_residual_m=vehicle_visual_residual_m,
        used_alignment=used_alignment,
        use_prediction=use_prediction,
    )


def _make_body_result(
    *,
    now,
    dt,
    body_xy,
    target_estimator,
    reference_traj,
    config,
    source,
    use_prediction=False,
):
    if body_xy is not None:
        target_estimator.update_measurement(body_xy, now)

    future_xy, fused_vel, used_vehicle_vel = target_estimator.make_future_point(
        now,
        config.lookahead_time_s,
        vehicle_vel_xy=None,
        vision_weight=config.vision_vel_weight,
        vehicle_weight=config.vehicle_vel_weight,
    )
    if future_xy is None:
        return None

    ref_xy = reference_traj.update(future_xy, dt)
    cmd_body = clamp_norm(ref_xy, config.max_cmd_offset_m)

    return TrackingCommandResult(
        cmd_body=cmd_body,
        target_xy=body_xy,
        ref_xy=ref_xy,
        future_xy=future_xy,
        fused_vel=fused_vel,
        source=source,
        used_vehicle_vel=used_vehicle_vel,
        used_vehicle_pose=False,
        fused_vehicle_pose=False,
        vehicle_visual_residual_m=None,
        used_alignment=False,
        use_prediction=use_prediction,
    )


def build_tracking_command(
    *,
    now,
    dt,
    tracking_frame,
    target_estimator,
    reference_traj,
    config,
    body_xy=None,
    drone_xy=None,
    drone_yaw=None,
    vehicle_state=None,
    frame_aligner=None,
    tag_yaw_drone=None,
    visual_source=None,
    use_prediction=False,
    use_vehicle_pose_fallback=False,
    last_visual_time=None,
):
    """
    由视觉测量、车端位姿或预测状态生成机体系水平控制量。

    tracking_frame == "local" 时，估计器状态在无人机局部坐标系；
    tracking_frame == "body" 时，估计器状态直接在机体系水平平面。
    """
    if tracking_frame == "body":
        if use_vehicle_pose_fallback:
            return None
        source = "PREDICT" if use_prediction else _visual_source_name(visual_source, False)
        return _make_body_result(
            now=now,
            dt=dt,
            body_xy=body_xy,
            target_estimator=target_estimator,
            reference_traj=reference_traj,
            config=config,
            source=source,
            use_prediction=use_prediction,
        )

    if drone_xy is None or drone_yaw is None:
        return None

    drone_xy = np.asarray(drone_xy, dtype=float)
    vehicle_vel = None
    used_alignment = False
    used_vehicle_pose = False
    fused_vehicle_pose = False
    residual_m = None

    if body_xy is not None:
        target_xy = drone_xy + body_to_local_xy(body_xy, drone_yaw)

        if vehicle_state is not None and frame_aligner is not None and tag_yaw_drone is not None:
            frame_aligner.update(
                _vehicle_xy(vehicle_state),
                vehicle_state.yaw_rad,
                target_xy,
                tag_yaw_drone,
            )

        if vehicle_state is not None and frame_aligner is not None and frame_aligner.initialized:
            vehicle_vel = _vehicle_velocity(vehicle_state, frame_aligner)
            used_alignment = True

            if config.vehicle_pose_fusion_enabled:
                vehicle_target_xy = frame_aligner.transform_point(_vehicle_xy(vehicle_state))
                residual_m = float(np.linalg.norm(target_xy - vehicle_target_xy))

                if residual_m <= config.ugv_visual_fuse_max_residual_m:
                    w_vis = _visual_pos_weight(config, visual_source)
                    target_xy = w_vis * target_xy + (1.0 - w_vis) * vehicle_target_xy
                    used_vehicle_pose = True
                    fused_vehicle_pose = True
                elif residual_m > config.ugv_visual_reject_residual_m:
                    logger.warning(
                        f"车端位置与视觉残差过大，拒绝融合车端位置: residual={residual_m:.2f} m"
                    )

        source = _visual_source_name(visual_source, fused_vehicle_pose)
        return _make_local_result(
            now=now,
            dt=dt,
            target_xy=target_xy,
            target_estimator=target_estimator,
            reference_traj=reference_traj,
            drone_xy=drone_xy,
            drone_yaw=drone_yaw,
            vehicle_vel=vehicle_vel,
            config=config,
            source=source,
            used_vehicle_pose=used_vehicle_pose,
            fused_vehicle_pose=fused_vehicle_pose,
            vehicle_visual_residual_m=residual_m,
            used_alignment=used_alignment,
            use_prediction=False,
        )

    if use_vehicle_pose_fallback:
        if not config.vehicle_pose_fallback_enabled:
            return None
        if vehicle_state is None or frame_aligner is None or not frame_aligner.initialized:
            return None
        if last_visual_time is not None and now - last_visual_time > config.ugv_pose_fallback_max_s:
            return None

        target_xy = frame_aligner.transform_point(_vehicle_xy(vehicle_state))
        vehicle_vel = _vehicle_velocity(vehicle_state, frame_aligner)
        return _make_local_result(
            now=now,
            dt=dt,
            target_xy=target_xy,
            target_estimator=target_estimator,
            reference_traj=reference_traj,
            drone_xy=drone_xy,
            drone_yaw=drone_yaw,
            vehicle_vel=vehicle_vel,
            config=config,
            source="UGV_POSE",
            used_vehicle_pose=True,
            fused_vehicle_pose=False,
            vehicle_visual_residual_m=None,
            used_alignment=True,
            use_prediction=False,
            vision_vel_weight=config.ugv_pose_est_vel_weight,
            vehicle_vel_weight=config.ugv_pose_vel_weight,
            max_cmd_offset_m=config.ugv_fallback_max_cmd_offset_m,
        )

    if not use_prediction:
        return None

    if vehicle_state is not None and frame_aligner is not None and frame_aligner.initialized:
        vehicle_vel = _vehicle_velocity(vehicle_state, frame_aligner)
        used_alignment = True

    return _make_local_result(
        now=now,
        dt=dt,
        target_xy=None,
        target_estimator=target_estimator,
        reference_traj=reference_traj,
        drone_xy=drone_xy,
        drone_yaw=drone_yaw,
        vehicle_vel=vehicle_vel,
        config=config,
        source="PREDICT",
        used_vehicle_pose=False,
        fused_vehicle_pose=False,
        vehicle_visual_residual_m=None,
        used_alignment=used_alignment,
        use_prediction=True,
    )
