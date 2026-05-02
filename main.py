"""
下视摄像头 AprilTag 位姿解算 / 无人机 Datalink 通信链路主入口。
"""

import threading
import time
import math

import cv2
import numpy as np
from loguru import logger

from mavlink.kb_DataLink import DataLink
from utils.kb_TagVisualizer import TagVisualizer
from uav_core.apriltag_pose import (
    TAG_SIZES,
    cameraMatrix,
    distCoeffs,
    estimate_pose,
    init_detector,
    select_target_tag,
)
from uav_core.camera import init_camera
from uav_core.color_marker_pose import (
    COLOR_MARKER_AXIS_LENGTH_M,
    draw_color_marker_debug,
    estimate_color_marker_pose,
)
from uav_core.control_modes import handle_control_mode, keyboard_listener
from uav_core.debug_tools import finish_debug_tools, init_debug_tools
from uav_core.frame_alignment import FrameAligner
from uav_core.reference_tracking import (
    ReferenceTrajectory,
    TargetEstimator,
    body_to_local_xy,
)
from uav_core.tracking_fusion import TrackingFusionConfig, build_tracking_command
from uav_core.vehicle_state import VehicleStateReceiver
from uav_core.visual_control import (
    compute_yaw_cmd,
    estimate_tag_forward_yaw_body,
    pnp_to_body_xy,
)


DEBUG_MODE_ENABLED = False  # 是否启用调试模式，启用后会保存图像并生成视频
UDP_SENDER_ENABLED = True  # 是否启用 UDP 高速图传
DATALINK_ENABLED = False  # 是否启用飞控通信链路，启用后会根据控制模式发送指令到飞控
VEHICLE_TCP_ENABLED = False  # 是否启用无人车 TCP 状态接入，默认关闭，避免运行 main.py 时阻塞网络
COLOR_MARKER_ENABLED = True  # AprilTag 失败时是否启用彩色标记 PnP 备用视觉

UDP_RECEIVER_IP = "10.105.26.61"  # UDP 图传接收端 IP 地址，需与接收端设置一致
UDP_SENDER_QUALITY = 20  # JPEG 压缩质量，范围 0-100，数值越小压缩越强，传输更快但图像质量更差

LOOKAHEAD_TIME_S = 0.5  # 目标前视时间，用于预测无人车未来位置
VEHICLE_STATE_TIMEOUT_S = 0.3  # 车端状态超过该时间未更新则视为失效
TAG_LOST_PREDICT_TIME_S = 1.0  # Tag 短时丢失时允许使用预测轨迹的最长时间
MAX_REF_SPEED_MPS = 0.8  # 平滑参考点最大移动速度
MAX_CMD_OFFSET_M = 1.2  # 单次发送给飞控的水平相对位移限幅
ALIGN_YAW_ALPHA = 0.05  # 坐标系 yaw 在线修正低通系数
ALIGN_POS_ALPHA = 0.05  # 坐标系平移在线修正低通系数
VISION_VEL_WEIGHT = 0.4  # 视觉估速权重
VEHICLE_VEL_WEIGHT = 0.6  # 车端速度前馈权重
VEHICLE_POSE_FUSION_ENABLED = True  # 视觉有效时是否融合车端位置参考
VEHICLE_POSE_FALLBACK_ENABLED = True  # 视觉失效时是否允许车端位姿接管参考点
UGV_POSE_FALLBACK_MAX_S = 5.0  # 距离上次视觉成功超过该时间后，不再单独使用车端位姿接管
UGV_VISUAL_FUSE_MAX_RESIDUAL_M = 0.25  # 视觉与车端位置残差小于该值时融合车端位置
UGV_VISUAL_REJECT_RESIDUAL_M = 0.60  # 视觉与车端位置残差大于该值时拒绝车端位置融合
TAG_VISUAL_POS_WEIGHT = 0.85  # AprilTag 视觉位置融合权重
COLOR_VISUAL_POS_WEIGHT = 0.70  # 彩色 PnP 视觉位置融合权重
UGV_POSE_EST_VEL_WEIGHT = 0.2  # 车端位姿 fallback 中估计器速度权重
UGV_POSE_VEL_WEIGHT = 0.8  # 车端位姿 fallback 中车端速度权重
UGV_FALLBACK_MAX_CMD_OFFSET_M = 0.8  # 车端位姿 fallback 阶段水平指令限幅
TAG_FORWARD_AXIS = "+Y"  # 默认 Tag +Y 方向与无人车车头方向一致

TRACKING_FUSION_CONFIG = TrackingFusionConfig(
    lookahead_time_s=LOOKAHEAD_TIME_S,
    max_cmd_offset_m=MAX_CMD_OFFSET_M,
    vision_vel_weight=VISION_VEL_WEIGHT,
    vehicle_vel_weight=VEHICLE_VEL_WEIGHT,
    vehicle_pose_fusion_enabled=VEHICLE_POSE_FUSION_ENABLED,
    vehicle_pose_fallback_enabled=VEHICLE_POSE_FALLBACK_ENABLED,
    ugv_pose_fallback_max_s=UGV_POSE_FALLBACK_MAX_S,
    ugv_visual_fuse_max_residual_m=UGV_VISUAL_FUSE_MAX_RESIDUAL_M,
    ugv_visual_reject_residual_m=UGV_VISUAL_REJECT_RESIDUAL_M,
    tag_visual_pos_weight=TAG_VISUAL_POS_WEIGHT,
    color_visual_pos_weight=COLOR_VISUAL_POS_WEIGHT,
    ugv_pose_est_vel_weight=UGV_POSE_EST_VEL_WEIGHT,
    ugv_pose_vel_weight=UGV_POSE_VEL_WEIGHT,
    ugv_fallback_max_cmd_offset_m=UGV_FALLBACK_MAX_CMD_OFFSET_M,
)


def init_datalink():
    """创建 DataLink 实例，启动收发线程，等待飞控连接"""
    # 创建 DataLink 实例并启动线程
    data_link = DataLink(port='/dev/ttyTHS0', baudrate=115200)  # 创建 DataLink 实例
    data_link.init_mavlink()  # 初始化 MAVLink

    # 启动接收线程
    receive_thread = threading.Thread(target=data_link.receive_loop, daemon=True)
    receive_thread.start()

    # 启动心跳线程
    heartbeat_thread = threading.Thread(target=data_link.send_heartbeat, daemon=True)
    heartbeat_thread.start()

    # 等待飞控连接
    while data_link.state.heartbeat_count < 5:
        logger.info("等待飞控连接...")
        time.sleep(1)

    return data_link


def drone_state_valid(data_link):
    """DataLink 状态有效时，set_pose 才会真正发送控制目标。"""
    if data_link is None:
        return False
    return (data_link.state.x != 0) and (data_link.state.y != 0) and (data_link.state.yaw != 0)


def get_drone_xy(data_link):
    return np.array([data_link.state.x, data_link.state.y], dtype=float)


def tag_yaw_to_drone_yaw(pnp_rvec, drone_yaw):
    """把视觉估计的 Tag 前向角从机体系转换到无人机局部坐标系。"""
    tag_yaw_body = estimate_tag_forward_yaw_body(pnp_rvec, TAG_FORWARD_AXIS)
    tag_dir_body = np.array([math.cos(tag_yaw_body), math.sin(tag_yaw_body)], dtype=float)
    tag_dir_drone = body_to_local_xy(tag_dir_body, drone_yaw)
    return math.atan2(float(tag_dir_drone[1]), float(tag_dir_drone[0]))


def draw_tracking_debug(color_frame, debug_lines):
    """在图像左下角叠加参考轨迹调试信息。"""
    if not debug_lines:
        return

    x = 10
    y = color_frame.shape[0] - 20 - 24 * (len(debug_lines) - 1)
    for line in debug_lines:
        cv2.putText(
            color_frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )
        y += 24


def get_tracking_frame(data_link):
    """根据飞控状态选择参考轨迹所在坐标系。"""
    return "local" if (DATALINK_ENABLED and drone_state_valid(data_link)) else "body"


def ensure_tracking_frame(tracking_frame, last_tracking_frame, target_estimator, reference_traj):
    """坐标系切换时重置估计器，避免 body/local 状态混用。"""
    if tracking_frame != last_tracking_frame:
        target_estimator.reset()
        reference_traj.reset()
        logger.info(f"参考轨迹坐标系切换为: {tracking_frame}")
        return tracking_frame
    return last_tracking_frame


def get_tracking_drone_state(data_link, tracking_frame):
    """local 跟踪需要无人机局部位置和 yaw；body 跟踪不需要。"""
    if tracking_frame != "local":
        return None, None
    return get_drone_xy(data_link), data_link.state.yaw


def vehicle_pose_status(tracking_result):
    """把融合结果转换成调试显示用的车端位置状态。"""
    if tracking_result.source == "UGV_POSE":
        return "fallback"
    if tracking_result.fused_vehicle_pose:
        return "fused"
    residual = tracking_result.vehicle_visual_residual_m
    if residual is not None and residual > TRACKING_FUSION_CONFIG.ugv_visual_reject_residual_m:
        return "rejected"
    return "off"


def residual_text(tracking_result):
    residual = tracking_result.vehicle_visual_residual_m
    if residual is None:
        return "n/a"
    return f"{residual:.2f}m"


def main():
    # ---------- 硬件初始化 ----------
    video_capture = init_camera()
    detector = init_detector()
    data_link = None
    if DATALINK_ENABLED:
        data_link = init_datalink()

    # ---------- 启动键盘监听线程 ----------
    threading.Thread(target=keyboard_listener, daemon=True).start()

    if DATALINK_ENABLED:
        logger.info("飞控已连接，开始 AprilTag 识别与位姿解算")
    else:
        logger.info("飞控链路关闭，开始 AprilTag 识别与位姿解算")

    # ---------- 无人车状态接入（默认关闭，避免阻塞网络） ----------
    vehicle_receiver = None
    vehicle_state_cache = None
    if VEHICLE_TCP_ENABLED:
        vehicle_receiver = VehicleStateReceiver()
        vehicle_state_cache = vehicle_receiver.cache
        vehicle_receiver.start()
    else:
        logger.info("无人车 TCP 状态接入关闭，仅使用视觉参考轨迹")

    # ---------- 调试工具初始化 ----------
    image_dumper, video_converter, udp_sender = init_debug_tools(
        UDP_SENDER_ENABLED,
        UDP_RECEIVER_IP,
        UDP_SENDER_QUALITY,
    )

    # ---------- 主循环 ----------
    frame_count = 0
    last_loop_time = time.time()
    last_visual_detected_time = time.time()  # 记录最后一次视觉观测成功的时间戳，用于计算视觉丢失持续时长
    last_tracking_frame = None
    target_estimator = TargetEstimator()
    reference_traj = ReferenceTrajectory(max_speed_mps=MAX_REF_SPEED_MPS)
    frame_aligner = FrameAligner(
        yaw_alpha=ALIGN_YAW_ALPHA,
        pos_alpha=ALIGN_POS_ALPHA,
    )
    try:
        while True:
            now = time.time()
            dt = max(1e-3, min(now - last_loop_time, 0.2))
            last_loop_time = now

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

            # ---------- 逐标签处理：位姿解算 + 控制指令计算 ----------
            tag_detected = 0
            control_target_valid = 0
            cmd_dx = cmd_dy = cmd_dz = cmd_dyaw = 0  # 默认零指令
            tracking_debug_lines = []
            tracking_result = None

            vehicle_state = None
            if vehicle_state_cache is not None:
                vehicle_state = vehicle_state_cache.get_latest(
                    now=now,
                    max_age=VEHICLE_STATE_TIMEOUT_S,
                )

            # v3: 按优先级选择目标 Tag
            target_tag = select_target_tag(april_tags)

            # 位姿估计
            if target_tag is not None:
                # 根据目标 Tag 的 ID 获取对应的实际边长
                target_tag_size = TAG_SIZES[target_tag.tag_id]
                target_half_tag_size = target_tag_size * 0.5

                pnp_ok, pnp_rvec, pnp_tvec, pnp_x, pnp_y, pnp_z, image_points = estimate_pose(target_tag, target_tag_size)
                if pnp_ok:
                    tag_detected = 1
                    last_visual_detected_time = now  # 视觉观测成功，刷新时间戳

                    body_xy = pnp_to_body_xy(pnp_x, pnp_y)
                    cmd_dyaw, _, _ = compute_yaw_cmd(pnp_rvec,
                                                     kp_yaw=0.5,
                                                     yaw_deadband=0.08,
                                                     max_dyaw=0.6,
                                                     hysteresis_bonus=0.15)

                    tracking_frame = get_tracking_frame(data_link)
                    last_tracking_frame = ensure_tracking_frame(
                        tracking_frame,
                        last_tracking_frame,
                        target_estimator,
                        reference_traj,
                    )
                    drone_xy, drone_yaw = get_tracking_drone_state(data_link, tracking_frame)
                    tag_yaw_drone = (
                        tag_yaw_to_drone_yaw(pnp_rvec, drone_yaw)
                        if tracking_frame == "local"
                        else None
                    )

                    tracking_result = build_tracking_command(
                        now=now,
                        dt=dt,
                        tracking_frame=tracking_frame,
                        target_estimator=target_estimator,
                        reference_traj=reference_traj,
                        config=TRACKING_FUSION_CONFIG,
                        body_xy=body_xy,
                        drone_xy=drone_xy,
                        drone_yaw=drone_yaw,
                        vehicle_state=vehicle_state,
                        frame_aligner=frame_aligner,
                        tag_yaw_drone=tag_yaw_drone,
                        visual_source="TAG",
                        use_prediction=False,
                    )

                    if tracking_result is not None:
                        cmd_body = tracking_result.cmd_body
                        cmd_dx = float(cmd_body[0])
                        cmd_dy = float(cmd_body[1])
                        control_target_valid = 1

                    """ 图像信息可视化 """
                    # ---------- 画 Tag 边框 ----------
                    for i in range(4):
                        pt1 = tuple(image_points[i].astype(int))
                        pt2 = tuple(image_points[(i + 1) % 4].astype(int))
                        cv2.line(color_frame, pt1, pt2, (0, 255, 0), 2)

                    # ---------- 画中心点 ----------
                    draw_center = tuple(target_tag.center.astype(int))
                    cv2.circle(color_frame, draw_center, 5, (0, 0, 255), -1)

                    # ---------- 画坐标轴 ----------
                    cv2.drawFrameAxes(
                        color_frame,
                        cameraMatrix,
                        distCoeffs,
                        pnp_rvec,
                        pnp_tvec,
                        target_half_tag_size  # v3: 使用目标 Tag 对应的 half_tag_size
                    )

                    # ---------- 显示数值 ----------
                    display_x = 10
                    display_y = (target_tag.tag_id % 65) * 30 + 30  # 根据 Tag ID 计算显示位置，避免重叠
                    cv2.putText(
                        color_frame,
                        f"ID: {target_tag.tag_id} | X: {pnp_x:+.3f} m  Y: {pnp_y:+.3f} m  Z: {pnp_z:+.3f} m",
                        (display_x, display_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # 显示电池电压
                    if DATALINK_ENABLED:
                        voltage_text = f"Battery: {data_link.state.battery_voltage:.2f} V"
                        cv2.putText(
                            color_frame,
                            voltage_text,
                            (color_frame.shape[1] - 200, 30),  # 电压信息显示在图像右上角
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

                    # ---------- 调用 TagVisualizer 类，计算像素比例尺 ----------
                    meter_per_pixel = TagVisualizer.compute_pixel_scale_on_tag(target_tag.corners, target_tag_size)  # v3: 使用目标 Tag 对应的边长
                    img_center = TagVisualizer.draw_image_center(color_frame)
                    TagVisualizer.draw_tvec_vector_scaled(
                        color_frame,
                        img_center,
                        pnp_tvec,
                        meter_per_pixel,
                        scale=1.0,
                        color=(255, 128, 0)
                    )

            # ---------- AprilTag 失败时，尝试彩色标记 PnP 备用视觉 ----------
            if tag_detected == 0 and COLOR_MARKER_ENABLED:
                color_observation = estimate_color_marker_pose(
                    color_frame,
                    cameraMatrix,
                    distCoeffs,
                )

                if color_observation is not None:
                    tag_detected = 1
                    last_visual_detected_time = now  # 彩色 PnP 成功，也视为有效视觉观测

                    pnp_rvec = color_observation.rvec
                    pnp_tvec = color_observation.tvec
                    pnp_x, pnp_y, pnp_z = pnp_tvec.flatten()

                    body_xy = pnp_to_body_xy(pnp_x, pnp_y)
                    cmd_dyaw, _, _ = compute_yaw_cmd(pnp_rvec,
                                                     kp_yaw=0.5,
                                                     yaw_deadband=0.08,
                                                     max_dyaw=0.6,
                                                     hysteresis_bonus=0.15)

                    tracking_frame = get_tracking_frame(data_link)
                    last_tracking_frame = ensure_tracking_frame(
                        tracking_frame,
                        last_tracking_frame,
                        target_estimator,
                        reference_traj,
                    )
                    drone_xy, drone_yaw = get_tracking_drone_state(data_link, tracking_frame)
                    tag_yaw_drone = (
                        tag_yaw_to_drone_yaw(pnp_rvec, drone_yaw)
                        if tracking_frame == "local"
                        else None
                    )

                    tracking_result = build_tracking_command(
                        now=now,
                        dt=dt,
                        tracking_frame=tracking_frame,
                        target_estimator=target_estimator,
                        reference_traj=reference_traj,
                        config=TRACKING_FUSION_CONFIG,
                        body_xy=body_xy,
                        drone_xy=drone_xy,
                        drone_yaw=drone_yaw,
                        vehicle_state=vehicle_state,
                        frame_aligner=frame_aligner,
                        tag_yaw_drone=tag_yaw_drone,
                        visual_source="COLOR",
                        use_prediction=False,
                    )

                    if tracking_result is not None:
                        cmd_body = tracking_result.cmd_body
                        cmd_dx = float(cmd_body[0])
                        cmd_dy = float(cmd_body[1])
                        control_target_valid = 1

                    draw_color_marker_debug(color_frame, color_observation)
                    cv2.drawFrameAxes(
                        color_frame,
                        cameraMatrix,
                        distCoeffs,
                        pnp_rvec,
                        pnp_tvec,
                        COLOR_MARKER_AXIS_LENGTH_M
                    )
                    cv2.putText(
                        color_frame,
                        f"COLOR | X: {pnp_x:+.3f} m  Y: {pnp_y:+.3f} m  Z: {pnp_z:+.3f} m",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

            # ---------- 控制模式分发 ----------
            tag_lost_duration = now - last_visual_detected_time  # 计算自上次视觉观测成功以来已经过去了多少秒

            if tag_detected == 0:
                tracking_frame = get_tracking_frame(data_link)
                last_tracking_frame = ensure_tracking_frame(
                    tracking_frame,
                    last_tracking_frame,
                    target_estimator,
                    reference_traj,
                )
                drone_xy, drone_yaw = get_tracking_drone_state(data_link, tracking_frame)

                # 视觉失效时优先尝试车端位姿参考点；只有不可用时才退回短时预测。
                tracking_result = build_tracking_command(
                    now=now,
                    dt=dt,
                    tracking_frame=tracking_frame,
                    target_estimator=target_estimator,
                    reference_traj=reference_traj,
                    config=TRACKING_FUSION_CONFIG,
                    body_xy=None,
                    drone_xy=drone_xy,
                    drone_yaw=drone_yaw,
                    vehicle_state=vehicle_state,
                    frame_aligner=frame_aligner,
                    use_vehicle_pose_fallback=True,
                    last_visual_time=last_visual_detected_time,
                )

                if tracking_result is not None:
                    cmd_body = tracking_result.cmd_body
                    cmd_dx = float(cmd_body[0])
                    cmd_dy = float(cmd_body[1])
                    cmd_dyaw = 0.0
                    control_target_valid = 1

                elif tag_lost_duration < TAG_LOST_PREDICT_TIME_S:
                    tracking_result = build_tracking_command(
                        now=now,
                        dt=dt,
                        tracking_frame=tracking_frame,
                        target_estimator=target_estimator,
                        reference_traj=reference_traj,
                        config=TRACKING_FUSION_CONFIG,
                        body_xy=None,
                        drone_xy=drone_xy,
                        drone_yaw=drone_yaw,
                        vehicle_state=vehicle_state,
                        frame_aligner=frame_aligner,
                        use_prediction=True,
                    )

                    if tracking_result is not None:
                        cmd_body = tracking_result.cmd_body
                        cmd_dx = float(cmd_body[0])
                        cmd_dy = float(cmd_body[1])
                        cmd_dyaw = 0.0
                        control_target_valid = 1

            if tracking_result is not None:
                source_text = tracking_result.source
                veh_vel_text = "veh_vel=on" if tracking_result.used_vehicle_vel else "veh_vel=off"
                veh_pose_text = f"veh_pose={vehicle_pose_status(tracking_result)}"
                residual_info = f"residual={residual_text(tracking_result)}"
                frame_text = f"frame={last_tracking_frame}"
                tracking_debug_lines = [
                    f"Ref {source_text} {frame_text} {veh_vel_text} {veh_pose_text}",
                    f"cmd body dx={cmd_dx:+.2f} dy={cmd_dy:+.2f} yaw={math.degrees(cmd_dyaw):+.1f}deg",
                    f"vel=({tracking_result.fused_vel[0]:+.2f},{tracking_result.fused_vel[1]:+.2f}) m/s {residual_info}",
                ]

                if frame_count % 15 == 0:
                    logger.debug(
                        f"[RefTrack] {source_text} {frame_text} {veh_vel_text} {veh_pose_text} {residual_info} | "
                        f"cmd=({cmd_dx:+.2f}, {cmd_dy:+.2f}) m | "
                        f"vel=({tracking_result.fused_vel[0]:+.2f}, {tracking_result.fused_vel[1]:+.2f}) m/s"
                    )

            draw_tracking_debug(color_frame, tracking_debug_lines)

            if DATALINK_ENABLED:
                handle_control_mode(data_link, control_target_valid, cmd_dx, cmd_dy, cmd_dz, cmd_dyaw, tag_lost_duration)

            # 高速图传
            if UDP_SENDER_ENABLED:
                udp_sender.send_frame(color_frame)

            # 转储图像
            if DEBUG_MODE_ENABLED:
                image_dumper.dump(color_frame)

            # 显示图像
            if DEBUG_MODE_ENABLED:
                cv2.imshow("AprilTag Detection", color_frame)
                if 27 == cv2.waitKey(1):
                    break

    finally:
        # =========================================================
        # 资源释放
        # =========================================================
        if UDP_SENDER_ENABLED:
            udp_sender.stop()
        video_capture.release()
        cv2.destroyAllWindows()
        logger.success("程序安全退出")

        finish_debug_tools(DEBUG_MODE_ENABLED, image_dumper, video_converter)


if __name__ == "__main__":
    main()
