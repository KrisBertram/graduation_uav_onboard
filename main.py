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
    clamp_norm,
    local_to_body_xy,
)
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
TAG_FORWARD_AXIS = "+Y"  # 默认 Tag +Y 方向与无人车车头方向一致


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


def build_tracking_command(
    *,
    now,
    dt,
    tracking_frame,
    target_estimator,
    reference_traj,
    body_xy=None,
    data_link=None,
    vehicle_state=None,
    frame_aligner=None,
    pnp_rvec=None,
    use_prediction=False,
):
    """
    由视觉测量/预测状态生成机体系水平控制量。

    tracking_frame == "local" 时，估计器状态在无人机局部坐标系；
    tracking_frame == "body" 时，估计器状态直接在机体系水平平面。
    """
    vehicle_vel = None
    used_alignment = False
    target_xy = None

    if tracking_frame == "local":
        drone_xy = get_drone_xy(data_link)
        drone_yaw = data_link.state.yaw

        if body_xy is not None:
            target_xy = drone_xy + body_to_local_xy(body_xy, drone_yaw)

            if vehicle_state is not None:
                tag_yaw_drone = tag_yaw_to_drone_yaw(pnp_rvec, drone_yaw)
                vehicle_xy = np.array([vehicle_state.pos_x, vehicle_state.pos_y], dtype=float)
                frame_aligner.update(
                    vehicle_xy,
                    vehicle_state.yaw_rad,
                    target_xy,
                    tag_yaw_drone,
                )

            target_estimator.update_measurement(target_xy, now)

        if vehicle_state is not None and frame_aligner is not None and frame_aligner.initialized:
            vehicle_vel = frame_aligner.vehicle_velocity_to_drone(
                vehicle_state.speed,
                vehicle_state.yaw_rad,
            )
            used_alignment = True

        future_xy, fused_vel, used_vehicle = target_estimator.make_future_point(
            now,
            LOOKAHEAD_TIME_S,
            vehicle_vel_xy=vehicle_vel,
            vision_weight=VISION_VEL_WEIGHT,
            vehicle_weight=VEHICLE_VEL_WEIGHT,
        )
        if future_xy is None:
            return None

        ref_xy = reference_traj.update(future_xy, dt)
        cmd_local = clamp_norm(ref_xy - drone_xy, MAX_CMD_OFFSET_M)
        cmd_body = local_to_body_xy(cmd_local, drone_yaw)

    else:
        if body_xy is not None:
            target_estimator.update_measurement(body_xy, now)

        future_xy, fused_vel, used_vehicle = target_estimator.make_future_point(
            now,
            LOOKAHEAD_TIME_S,
            vehicle_vel_xy=None,
            vision_weight=VISION_VEL_WEIGHT,
            vehicle_weight=VEHICLE_VEL_WEIGHT,
        )
        if future_xy is None:
            return None

        ref_xy = reference_traj.update(future_xy, dt)
        cmd_body = clamp_norm(ref_xy, MAX_CMD_OFFSET_M)
        target_xy = body_xy

    return {
        "cmd_body": cmd_body,
        "target_xy": target_xy,
        "ref_xy": ref_xy,
        "future_xy": future_xy,
        "fused_vel": fused_vel,
        "used_vehicle": used_vehicle,
        "used_alignment": used_alignment,
        "use_prediction": use_prediction,
    }


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
    last_tag_detected_time = time.time()  # 记录最后一次成功检测到目标 Tag 的时间戳，用于计算丢失持续时长
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
            tracking_source = None

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
                    tracking_source = "TAG"
                    last_tag_detected_time = now  # 检测成功，刷新时间戳

                    body_xy = pnp_to_body_xy(pnp_x, pnp_y)
                    cmd_dyaw, _, _ = compute_yaw_cmd(pnp_rvec,
                                                     kp_yaw=0.5,
                                                     yaw_deadband=0.08,
                                                     max_dyaw=0.6,
                                                     hysteresis_bonus=0.15)

                    tracking_frame = "local" if (DATALINK_ENABLED and drone_state_valid(data_link)) else "body"
                    if tracking_frame != last_tracking_frame:
                        target_estimator.reset()
                        reference_traj.reset()
                        last_tracking_frame = tracking_frame
                        logger.info(f"参考轨迹坐标系切换为: {tracking_frame}")

                    tracking_result = build_tracking_command(
                        now=now,
                        dt=dt,
                        tracking_frame=tracking_frame,
                        target_estimator=target_estimator,
                        reference_traj=reference_traj,
                        body_xy=body_xy,
                        data_link=data_link,
                        vehicle_state=vehicle_state,
                        frame_aligner=frame_aligner,
                        pnp_rvec=pnp_rvec,
                        use_prediction=False,
                    )

                    if tracking_result is not None:
                        cmd_body = tracking_result["cmd_body"]
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
                    tracking_source = "COLOR"
                    last_tag_detected_time = now  # 彩色 PnP 成功，也视为有效视觉观测

                    pnp_rvec = color_observation.rvec
                    pnp_tvec = color_observation.tvec
                    pnp_x, pnp_y, pnp_z = pnp_tvec.flatten()

                    body_xy = pnp_to_body_xy(pnp_x, pnp_y)
                    cmd_dyaw, _, _ = compute_yaw_cmd(pnp_rvec,
                                                     kp_yaw=0.5,
                                                     yaw_deadband=0.08,
                                                     max_dyaw=0.6,
                                                     hysteresis_bonus=0.15)

                    tracking_frame = "local" if (DATALINK_ENABLED and drone_state_valid(data_link)) else "body"
                    if tracking_frame != last_tracking_frame:
                        target_estimator.reset()
                        reference_traj.reset()
                        last_tracking_frame = tracking_frame
                        logger.info(f"参考轨迹坐标系切换为: {tracking_frame}")

                    tracking_result = build_tracking_command(
                        now=now,
                        dt=dt,
                        tracking_frame=tracking_frame,
                        target_estimator=target_estimator,
                        reference_traj=reference_traj,
                        body_xy=body_xy,
                        data_link=data_link,
                        vehicle_state=vehicle_state,
                        frame_aligner=frame_aligner,
                        pnp_rvec=pnp_rvec,
                        use_prediction=False,
                    )

                    if tracking_result is not None:
                        cmd_body = tracking_result["cmd_body"]
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
            tag_lost_duration = now - last_tag_detected_time  # 计算自上次检测到目标 Tag 以来已经过去了多少秒

            if tag_detected == 0 and tag_lost_duration < TAG_LOST_PREDICT_TIME_S:
                tracking_frame = "local" if (DATALINK_ENABLED and drone_state_valid(data_link)) else "body"
                if tracking_frame == last_tracking_frame and target_estimator.initialized:
                    tracking_result = build_tracking_command(
                        now=now,
                        dt=dt,
                        tracking_frame=tracking_frame,
                        target_estimator=target_estimator,
                        reference_traj=reference_traj,
                        body_xy=None,
                        data_link=data_link,
                        vehicle_state=vehicle_state,
                        frame_aligner=frame_aligner,
                        pnp_rvec=None,
                        use_prediction=True,
                    )

                    if tracking_result is not None:
                        cmd_body = tracking_result["cmd_body"]
                        cmd_dx = float(cmd_body[0])
                        cmd_dy = float(cmd_body[1])
                        cmd_dyaw = 0.0
                        control_target_valid = 1

            if tracking_result is not None:
                mode_text = "PRED" if tracking_result["use_prediction"] else (tracking_source or "TAG")
                veh_text = "veh=on" if tracking_result["used_vehicle"] else "veh=off"
                frame_text = f"frame={last_tracking_frame}"
                tracking_debug_lines = [
                    f"Ref {mode_text} {frame_text} {veh_text}",
                    f"cmd body dx={cmd_dx:+.2f} dy={cmd_dy:+.2f} yaw={math.degrees(cmd_dyaw):+.1f}deg",
                    f"vel=({tracking_result['fused_vel'][0]:+.2f},{tracking_result['fused_vel'][1]:+.2f}) m/s",
                ]

                if frame_count % 15 == 0:
                    logger.debug(
                        f"[RefTrack] {mode_text} {frame_text} {veh_text} | "
                        f"cmd=({cmd_dx:+.2f}, {cmd_dy:+.2f}) m | "
                        f"vel=({tracking_result['fused_vel'][0]:+.2f}, {tracking_result['fused_vel'][1]:+.2f}) m/s"
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
