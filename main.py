"""
下视摄像头 AprilTag 位姿解算 / 无人机 Datalink 通信链路主入口。
"""

import threading
import time
import math
from pathlib import Path

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
from uav_core import control_modes
from uav_core.codex_flight_log import CodexFlightLogger
from uav_core.control_modes import get_control_mode_snapshot, handle_control_mode, keyboard_listener
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
    compute_forward_yaw_cmd,
    estimate_tag_forward_yaw_body,
    pnp_to_body_xy,
)


DEBUG_MODE_ENABLED = False  # 是否启用调试模式，启用后会保存图像并生成视频
UDP_SENDER_ENABLED = True  # 是否启用 UDP 高速图传
DATALINK_ENABLED = True  # 是否启用飞控通信链路，启用后会根据控制模式发送指令到飞控
VEHICLE_TCP_ENABLED = True  # 是否启用无人车 TCP 状态接入，默认关闭，避免运行 main.py 时阻塞网络
COLOR_MARKER_ENABLED = True  # AprilTag 失败时是否启用彩色标记 PnP 备用视觉

# 是否启用 Codex 飞行复盘日志。该日志写入 JSONL 文件，供飞行后诊断视觉/融合/控制链路。
# 调试建议：默认保持 True；若极限测试主循环耗时，可临时设 False 排除日志开销。
CODEX_FLIGHT_LOG_ENABLED = True

# Codex 飞行日志目录，使用相对路径时相对于项目根目录。
# 推荐保持 logs；该目录已加入 .gitignore，避免把飞行数据误提交到 Git。
CODEX_FLIGHT_LOG_DIR = "logs"

# Codex 飞行日志 sample 采样间隔。sample 会记录视觉、车端、无人机、融合结果和控制量快照。
# 调大：日志更小、写盘更少；调小：复盘更细，但文件增长更快。
# 推荐范围：0.1~0.5 s；初始推荐 0.2 s，约 5 Hz。
CODEX_FLIGHT_LOG_SAMPLE_INTERVAL_S = 0.5

# Codex 飞行日志重复事件最小间隔。带状态值的事件仍会在状态变化时立即记录。
# 调大：异常重复事件更少；调小：更容易看清短时间内反复触发的保护逻辑。
# 推荐范围：0.2~2.0 s；初始推荐 0.5 s。
CODEX_FLIGHT_LOG_EVENT_MIN_INTERVAL_S = 0.8

# Codex 飞行日志刷盘间隔。日志每次写入先进入文件缓冲，超过该间隔会 flush 到磁盘。
# 调大：写盘更少；调小：意外断电时丢失的最后日志更少。
# 推荐范围：0.5~3.0 s；初始推荐 1.0 s。
CODEX_FLIGHT_LOG_FLUSH_INTERVAL_S = 1.0

UDP_RECEIVER_IP = "10.105.26.61"  # UDP 图传接收端 IP 地址，需与接收端设置一致
UDP_SENDER_QUALITY = 30  # JPEG 压缩质量，范围 0-100，数值越小压缩越强，传输更快但图像质量更差

# 飞控连接等待轮询间隔：等待 MAVLink 心跳时每次 sleep 的时间。
# 调大：等待日志更少、CPU 占用更低；调小：连接成功后进入主流程更快。
# 推荐范围：0.5~2.0 s；初始推荐 1.0 s。
DATALINK_HEARTBEAT_WAIT_S = 1.0

# 参考轨迹 debug 日志输出间隔，单位为主循环帧数。
# 调大：日志更清爽；调小：便于调参时观察每帧变化，但会刷屏。
# 推荐范围：5~30 帧；UDP 图传/视觉调试时推荐 10~20 帧。
TRACKING_DEBUG_LOG_INTERVAL_FRAMES = 15

# ---------- 视觉 + 车端位姿融合调参 ----------
# 调参顺序建议：
# 1. 先确认 TAG_FORWARD_AXIS、车端 yaw 方向和 FrameAligner 首次对齐日志正确。
# 2. 再调 LOOKAHEAD_TIME_S、MAX_REF_SPEED_MPS、MAX_CMD_OFFSET_M，让正常视觉跟踪稳定。
# 3. 再打开 VEHICLE_TCP_ENABLED 后调 VISION_VEL_WEIGHT / VEHICLE_VEL_WEIGHT。
# 4. 最后调车端位置融合残差阈值和 UGV_POSE fallback 参数。

# 视觉偏航比例增益：Tag/车辆前向偏航误差乘以该系数，得到 dyaw 指令。
# 调大：机头对齐车头更快；过大会在 rvec 抖动或近距离时引起偏航振荡。
# 调小：偏航更柔和；过小会导致无人机机头长期落后车头方向。
# 推荐范围：0.3~0.8；初始推荐 0.4~0.6。
FORWARD_YAW_KP = 0.5

# 视觉偏航死区：偏航误差小于该弧度时输出 0，避免小角度来回抖动。
# 调大：偏航更稳，但允许更大的残余角度误差。
# 调小：对齐更精确，但可能出现小幅抖动。
# 推荐范围：0.04~0.12 rad（约 2.3~6.9 deg）；初始推荐 0.08 rad。
FORWARD_YAW_DEADBAND_RAD = 0.08

# 视觉偏航单帧限幅：限制每次 set_pose 发送的相对偏航角。
# 调大：偏航修正更激进；过大可能让机头快速摆动。
# 调小：偏航更平滑；过小会在大角度误差时收敛慢。
# 推荐范围：0.3~0.8 rad；初始推荐 0.5~0.6 rad。
MAX_FORWARD_DYAW_RAD = 0.6

# 目标 alpha-beta 估计器的位置修正系数 alpha。
# 调大：估计位置更贴近最新视觉/车端测量；过大会放大检测噪声。
# 调小：估计位置更平滑；过小会让目标位置响应滞后。
# 推荐范围：0.2~0.6；初始推荐 0.3~0.4。
TARGET_ESTIMATOR_ALPHA = 0.35

# 目标 alpha-beta 估计器的速度修正系数 beta。
# 调大：速度估计更快响应目标变速；过大会让速度估计抖动。
# 调小：速度估计更稳定；过小会让前视预测跟不上车辆加减速。
# 推荐范围：0.04~0.15；初始推荐 0.06~0.10。
TARGET_ESTIMATOR_BETA = 0.08

''' 以下参数是添加到 TrackingFusionConfig 中的，调整后会影响视觉与车端位姿融合的行为和效果 '''
# 目标前视时间：把目标当前位置沿融合速度向前预测多少秒，补偿无人车运动和控制滞后。
# 调大：无人机会更提前追车，适合车速高或系统延迟大；过大会超前、画圈或来回修正。
# 调小：跟踪更保守，适合慢速调试；过小会明显落后移动平台。
# 推荐范围：0.3~0.8 s；室内慢速先用 0.4~0.6 s。
LOOKAHEAD_TIME_S = 0.5

# 车端状态超时时间：超过该时间没有收到新 TCP 状态，就认为车端数据不可用。
# 调大：更能容忍 WiFi 抖动，但会使用更旧的车端速度/位置，可能追错。
# 调小：数据更新更严格，但 TCP 频率低或偶发丢包时会频繁退出融合。
# 推荐范围：车端发送周期的 2~4 倍；若车端 20Hz，推荐 0.15~0.30 s。
VEHICLE_STATE_TIMEOUT_S = 0.3

# 纯预测宽限时间：视觉和车端位姿 fallback 都不可用时，仅靠历史估计继续预测多久。
# 调大：短时遮挡更平滑；过大会在目标真实变向后继续按旧方向飞。
# 调小：更安全保守；过小会频繁进入悬停保护。
# 推荐范围：0.5~1.5 s；初始推荐 1.0 s。
TAG_LOST_PREDICT_TIME_S = 1.0

# 参考点最大移动速度：限制平滑参考点每秒最多移动多少米。
# 调大：响应更快，适合无人车速度高；过大可能导致控制指令突变。
# 调小：指令更柔和，适合早期联调；过小会跟不上车。
# 推荐范围：略高于无人车最大实测速度，例如车速 0.5 m/s 时取 0.6~0.9 m/s。
MAX_REF_SPEED_MPS = 0.8

# 正常跟踪水平指令限幅：单次 set_pose 发送的机体系水平相对位移上限。
# 调大：无人机追偏差更积极；过大时视觉误检或坐标系错误会带来较大位移风险。
# 调小：更安全，但大偏差时追踪恢复慢。
# 推荐范围：0.6~1.5 m；实测前先用 0.8~1.2 m。
MAX_CMD_OFFSET_M = 1.2

# 坐标系 yaw 在线修正低通系数：视觉有效时，FrameAligner 对车端坐标系朝向的修正速度。
# 调大：对安装角误差/车端 yaw 漂移修正更快；过大会把视觉抖动注入坐标系。
# 调小：坐标系更稳；过小则初始误差或慢漂移修正很慢。
# 推荐范围：0.02~0.15；初始推荐 0.03~0.08。
ALIGN_YAW_ALPHA = 0.05

# 坐标系平移在线修正低通系数：视觉有效时，FrameAligner 对坐标平移 t 的修正速度。
# 调大：对无人机局部位置漂移/车端位置误差适应更快；过大会跟随视觉噪声抖动。
# 调小：参考系更平滑；过小则长期偏差修正慢。
# 推荐范围：0.02~0.15；初始推荐 0.03~0.08。
ALIGN_POS_ALPHA = 0.05

# 融合速度中的视觉估速权重。视觉估速来自相邻视觉/融合测量差分，短时可能受检测抖动影响。
# 调大：更相信视觉相对运动，适合车端速度不准或 yaw 漂移明显；过大会受视觉漏检/抖动影响。
# 调小：更相信车端速度前馈，适合车端速度/yaw 稳定的场景。
# 推荐范围：0.2~0.6；应与 VEHICLE_VEL_WEIGHT 大致相加为 1。
VISION_VEL_WEIGHT = 0.4

# 融合速度中的车端速度前馈权重。车端速度由 speed+yaw 经 FrameAligner 转到无人机局部系。
# 调大：预测更跟随无人车运动，能改善运动模糊导致的视觉间歇；车端 yaw 错时会带偏。
# 调小：降低错误车端速度影响，但移动平台跟踪会更滞后。
# 推荐范围：0.4~0.8；应与 VISION_VEL_WEIGHT 大致相加为 1。
VEHICLE_VEL_WEIGHT = 0.6

# 视觉有效时是否允许融合车端位置。关闭后仍可使用车端速度前馈和更新 FrameAligner。
# 调试建议：车端坐标、yaw、FrameAligner 首次对齐未确认前可先设 False；确认后设 True。
VEHICLE_POSE_FUSION_ENABLED = False

# 视觉失效时是否允许车端位姿接管参考点。必须先完成 FrameAligner 对齐，否则不会启用。
# 调试建议：首次实测可先保持 True 但让 DATALINK_ENABLED=False 观察日志；确认 source=UGV_POSE 稳定后再接飞控。
VEHICLE_POSE_FALLBACK_ENABLED = False

# 车端位姿 fallback 最长持续时间：距离上次视觉成功超过该时间后，不再单独相信车端位姿。
# 调大：长时间视觉丢失仍能跟随车端轨迹；过大时坐标系漂移后风险增加。
# 调小：更依赖视觉重新捕获；过小则车端 fallback 作用不明显。
# 推荐范围：2~8 s；若车端定位稳定、WiFi 稳定可取 5~8 s。
UGV_POSE_FALLBACK_MAX_S = 5.0

# 视觉与车端位置残差小于该值时，认为二者一致，可融合车端位置。
# 调大：更容易融合车端位置，参考点更平滑；过大会把错误车端坐标混入视觉测量。
# 调小：只在非常一致时融合，更安全；过小则融合很少发生。
# 推荐范围：0.15~0.35 m；初始推荐 0.25 m。
UGV_VISUAL_FUSE_MAX_RESIDUAL_M = 0.25

# 视觉与车端位置残差大于该值时，拒绝车端位置融合并打印 warning。
# 调大：更宽容，但可能掩盖坐标系/协议错误；调小：更早暴露问题，但可能误报。
# 推荐范围：0.4~1.0 m；通常应为 UGV_VISUAL_FUSE_MAX_RESIDUAL_M 的 2~3 倍。
UGV_VISUAL_REJECT_RESIDUAL_M = 0.60

# AprilTag 视觉位置融合权重：残差足够小时，融合位置 = 该权重*视觉 + (1-该权重)*车端位置。
# 调大：更相信 AprilTag，适合 Tag 稳定清晰；调小：更平滑、更依赖车端轨迹。
# 推荐范围：0.75~0.95；AprilTag 通常比彩色 PnP 更可靠，建议高一些。
TAG_VISUAL_POS_WEIGHT = 0.85

# 彩色 PnP 视觉位置融合权重：彩色备用视觉成功时的位置融合权重。
# 调大：更相信彩色 PnP；调小：彩色检测抖动时让车端位置多参与平滑。
# 推荐范围：0.55~0.85；初始推荐 0.65~0.75。
COLOR_VISUAL_POS_WEIGHT = 0.85

# UGV_POSE fallback 中估计器速度权重：视觉失效后，历史估计速度占融合速度的比例。
# 调大：参考点更延续视觉丢失前的运动趋势；过大会削弱车端速度接管效果。
# 调小：更快切到车端速度；车端速度不准时可能出现突变。
# 推荐范围：0.0~0.4；应与 UGV_POSE_VEL_WEIGHT 大致相加为 1。
UGV_POSE_EST_VEL_WEIGHT = 0.2

# UGV_POSE fallback 中车端速度权重：视觉失效后，车端 speed+yaw 前馈占融合速度的比例。
# 调大：更积极跟随无人车实时运动；车端 yaw 或 speed 错误时会带偏。
# 调小：更保守，但视觉失效后会更滞后。
# 推荐范围：0.6~1.0；应与 UGV_POSE_EST_VEL_WEIGHT 大致相加为 1。
UGV_POSE_VEL_WEIGHT = 0.8

# UGV_POSE fallback 阶段水平指令限幅：视觉失效时单次水平相对位移上限。
# 调大：视觉丢失后追车更快；过大时坐标系对齐误差会带来更大风险。
# 调小：更安全保守；过小可能跟不上车。
# 推荐范围：0.4~1.0 m；建议小于或等于 MAX_CMD_OFFSET_M。
UGV_FALLBACK_MAX_CMD_OFFSET_M = 0.8

# Tag 前向轴：实际安装中哪条 Tag 坐标轴代表无人车车头方向。
# 推荐安装为 +Y；如果实物不可调整，才改为 +X/-X/+Y/-Y。
# 该参数会影响 FrameAligner yaw 对齐和车端速度方向，改错会让车端前馈明显变差。
TAG_FORWARD_AXIS = "+Y"

# 模式 4 是否必须等待首次车机坐标系对齐完成后才允许发送跟踪控制指令。
# 开启后：进入模式 4 仍继续检测视觉标记和更新 FrameAligner，但 frame_aligner.initialized=True 前不发 set_pose。
# 关闭后：模式 4 会恢复为只要有视觉/预测目标就可立即发送控制，适合不接车端坐标的纯视觉调试。
# 推荐：实车任务保持 True；只有确认不需要车端坐标系时才临时设 False。
TRACKING_REQUIRE_INITIAL_ALIGNMENT = True
TRACKING_WAIT_FRAME_ALIGNMENT_REASON = "WAIT_FRAME_ALIGNMENT"

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
        time.sleep(DATALINK_HEARTBEAT_WAIT_S)

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


def log_float(value, digits=3):
    """飞行复盘日志用的安全浮点转换。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def log_xy(vec_xy, digits=3):
    if vec_xy is None:
        return None
    arr = np.asarray(vec_xy, dtype=float).reshape(-1)
    return [log_float(item, digits) for item in arr[:2]]


def build_session_config_log(flight_log):
    """记录一次运行的关键配置，便于飞行后确认当时的开关和调参值。"""
    return {
        "git": {
            "commit": flight_log.git_sha,
        },
        "switches": {
            "debug_mode_enabled": DEBUG_MODE_ENABLED,
            "udp_sender_enabled": UDP_SENDER_ENABLED,
            "datalink_enabled": DATALINK_ENABLED,
            "vehicle_tcp_enabled": VEHICLE_TCP_ENABLED,
            "color_marker_enabled": COLOR_MARKER_ENABLED,
        },
        "codex_flight_log": {
            "enabled": CODEX_FLIGHT_LOG_ENABLED,
            "dir": CODEX_FLIGHT_LOG_DIR,
            "sample_interval_s": CODEX_FLIGHT_LOG_SAMPLE_INTERVAL_S,
            "event_min_interval_s": CODEX_FLIGHT_LOG_EVENT_MIN_INTERVAL_S,
            "flush_interval_s": CODEX_FLIGHT_LOG_FLUSH_INTERVAL_S,
        },
        "vision": {
            "tag_forward_axis": TAG_FORWARD_AXIS,
            "tag_sizes_m": TAG_SIZES,
            "color_marker_axis_length_m": COLOR_MARKER_AXIS_LENGTH_M,
        },
        "yaw_control": {
            "kp": FORWARD_YAW_KP,
            "deadband_rad": FORWARD_YAW_DEADBAND_RAD,
            "max_dyaw_rad": MAX_FORWARD_DYAW_RAD,
        },
        "target_estimator": {
            "alpha": TARGET_ESTIMATOR_ALPHA,
            "beta": TARGET_ESTIMATOR_BETA,
            "max_ref_speed_mps": MAX_REF_SPEED_MPS,
        },
        "tracking_fusion": dict(TRACKING_FUSION_CONFIG.__dict__),
        "control_modes": {
            "arm_wait_s": control_modes.ARM_WAIT_S,
            "takeoff_wait_s": control_modes.TAKEOFF_WAIT_S,
            "takeoff_hold_z_m": control_modes.TAKEOFF_HOLD_Z_M,
            "takeoff_hold_wait_s": control_modes.TAKEOFF_HOLD_WAIT_S,
            "land_wait_s": control_modes.LAND_WAIT_S,
            "track_target_z_m": control_modes.TRACK_TARGET_Z_M,
            "track_direct_z_enable_below_m": control_modes.TRACK_DIRECT_Z_ENABLE_BELOW_M,
            "track_lost_grace_s": control_modes.TRACK_LOST_GRACE_S,
        },
        "runtime": {
            "datalink_heartbeat_wait_s": DATALINK_HEARTBEAT_WAIT_S,
            "tracking_debug_log_interval_frames": TRACKING_DEBUG_LOG_INTERVAL_FRAMES,
            "vehicle_state_timeout_s": VEHICLE_STATE_TIMEOUT_S,
            "tag_lost_predict_time_s": TAG_LOST_PREDICT_TIME_S,
            "tracking_require_initial_alignment": TRACKING_REQUIRE_INITIAL_ALIGNMENT,
        },
    }


def build_drone_log(data_link):
    if data_link is None:
        return None
    state = data_link.state
    return {
        "x": log_float(state.x),
        "y": log_float(state.y),
        "z": log_float(state.z),
        "yaw_deg": log_float(math.degrees(state.yaw), 2),
        "relative_alt": log_float(state.relative_alt),
        "battery_voltage": log_float(state.battery_voltage, 2),
        "heartbeat_count": int(state.heartbeat_count),
    }


def build_vehicle_log(vehicle_state, now):
    if vehicle_state is None:
        return {"available": False}
    return {
        "available": True,
        "speed": log_float(vehicle_state.speed),
        "yaw_deg": log_float(vehicle_state.yaw, 2),
        "pos_x": log_float(vehicle_state.pos_x),
        "pos_y": log_float(vehicle_state.pos_y),
        "pos_z": log_float(vehicle_state.pos_z),
        "age_s": log_float(now - vehicle_state.timestamp, 3),
        "action": int(vehicle_state.action),
    }


def build_alignment_log(frame_aligner):
    return {
        "initialized": bool(frame_aligner.initialized),
        "theta_deg": log_float(math.degrees(frame_aligner.theta), 2),
        "translation_xy": log_xy(frame_aligner.translation),
    }


def build_tracking_log(tracking_result, tracking_frame):
    if tracking_result is None:
        return {
            "source": "NONE",
            "frame": tracking_frame,
            "used_vehicle_vel": False,
            "veh_pose": "off",
            "residual": None,
            "fused_vel": None,
            "ref_xy": None,
            "future_xy": None,
        }

    return {
        "source": tracking_result.source,
        "frame": tracking_frame,
        "used_vehicle_vel": bool(tracking_result.used_vehicle_vel),
        "veh_pose": vehicle_pose_status(tracking_result),
        "residual": log_float(tracking_result.vehicle_visual_residual_m),
        "fused_vel": log_xy(tracking_result.fused_vel),
        "target_xy": log_xy(tracking_result.target_xy),
        "ref_xy": log_xy(tracking_result.ref_xy),
        "future_xy": log_xy(tracking_result.future_xy),
        "used_alignment": bool(tracking_result.used_alignment),
        "use_prediction": bool(tracking_result.use_prediction),
    }


def get_tracking_task_gate(mode_snapshot, frame_aligner):
    """模式 4 任务门控：首次坐标系对齐完成前，只允许主循环感知和对齐，不允许发跟踪控制。"""
    if mode_snapshot["id"] != 4:
        return True, None
    if not TRACKING_REQUIRE_INITIAL_ALIGNMENT:
        return True, None
    if frame_aligner.initialized:
        return True, None
    return False, TRACKING_WAIT_FRAME_ALIGNMENT_REASON


def command_send_valid(mode_snapshot, control_target_valid, tracking_task_ready):
    """本帧是否允许把跟踪控制量真正交给模式 4 发送。"""
    return bool(mode_snapshot["id"] == 4 and control_target_valid and tracking_task_ready)


def build_command_log(control_target_valid,
                      tracking_task_ready,
                      tracking_wait_reason,
                      mode_snapshot,
                      cmd_dx,
                      cmd_dy,
                      cmd_dz,
                      cmd_dyaw):
    return {
        "valid": command_send_valid(mode_snapshot, control_target_valid, tracking_task_ready),
        "target_valid": bool(control_target_valid),
        "task_ready": bool(tracking_task_ready),
        "wait_reason": tracking_wait_reason,
        "dx": log_float(cmd_dx),
        "dy": log_float(cmd_dy),
        "dz": log_float(cmd_dz),
        "dyaw_deg": log_float(math.degrees(cmd_dyaw), 2),
    }


def tracking_source_name(tracking_result):
    return tracking_result.source if tracking_result is not None else "NONE"


def loss_phase_name(vision_source, tracking_source, control_target_valid):
    if vision_source in ("TAG", "COLOR"):
        return "visual"
    if tracking_source == "UGV_POSE":
        return "ugv_pose"
    if tracking_source == "PREDICT":
        return "predict"
    if control_target_valid:
        return "valid_no_visual"
    return "lost_no_target"


def record_flight_events(flight_log,
                         frame_count,
                         vision_log,
                         tracking_result,
                         tracking_frame,
                         vehicle_state_cache,
                         vehicle_state,
                         frame_aligner,
                         control_target_valid,
                         tracking_task_ready,
                         tracking_wait_reason,
                         mode_snapshot):
    """只在关键状态变化时写 event，避免飞行日志膨胀。"""
    tracking_source = tracking_source_name(tracking_result)
    vehicle_status = "off"
    if vehicle_state_cache is not None:
        vehicle_status = "available" if vehicle_state is not None else "timeout"

    flight_log.record_event(
        "vision_source_change",
        data={"source": vision_log["source"], "tag_id": vision_log.get("tag_id")},
        frame=frame_count,
        dedupe_key="vision_source",
        value=vision_log["source"],
    )
    flight_log.record_event(
        "tracking_source_change",
        data={"source": tracking_source, "frame": tracking_frame},
        frame=frame_count,
        dedupe_key="tracking_source",
        value=tracking_source,
    )
    flight_log.record_event(
        "vehicle_state_status",
        data={"status": vehicle_status},
        frame=frame_count,
        dedupe_key="vehicle_state_status",
        value=vehicle_status,
    )
    if frame_aligner.initialized:
        flight_log.record_event(
            "alignment_initialized",
            data=build_alignment_log(frame_aligner),
            frame=frame_count,
            dedupe_key="alignment_initialized",
            value=True,
        )
    flight_log.record_event(
        "control_valid_change",
        data={
            "valid": command_send_valid(mode_snapshot, control_target_valid, tracking_task_ready),
            "target_valid": bool(control_target_valid),
            "task_ready": bool(tracking_task_ready),
        },
        frame=frame_count,
        dedupe_key="control_valid",
        value=command_send_valid(mode_snapshot, control_target_valid, tracking_task_ready),
    )
    flight_log.record_event(
        "tracking_task_gate_change",
        data={
            "ready": bool(tracking_task_ready),
            "wait_reason": tracking_wait_reason,
            "alignment": build_alignment_log(frame_aligner),
        },
        frame=frame_count,
        dedupe_key="tracking_task_gate",
        value=(bool(tracking_task_ready), tracking_wait_reason),
    )
    flight_log.record_event(
        "mode_change",
        data=mode_snapshot,
        frame=frame_count,
        dedupe_key="control_mode",
        value=mode_snapshot["id"],
    )
    flight_log.record_event(
        "loss_phase_change",
        data={
            "phase": loss_phase_name(vision_log["source"], tracking_source, control_target_valid),
            "vision_source": vision_log["source"],
            "tracking_source": tracking_source,
        },
        frame=frame_count,
        dedupe_key="loss_phase",
        value=loss_phase_name(vision_log["source"], tracking_source, control_target_valid),
    )


def build_flight_sample(vision_log,
                        tracking_result,
                        tracking_frame,
                        control_target_valid,
                        cmd_dx,
                        cmd_dy,
                        cmd_dz,
                        cmd_dyaw,
                        data_link,
                        vehicle_state,
                        now,
                        frame_aligner,
                        mode_snapshot,
                        tracking_task_ready,
                        tracking_wait_reason,
                        tag_lost_duration):
    return {
        "vision": vision_log,
        "tracking": build_tracking_log(tracking_result, tracking_frame),
        "command": build_command_log(
            control_target_valid,
            tracking_task_ready,
            tracking_wait_reason,
            mode_snapshot,
            cmd_dx,
            cmd_dy,
            cmd_dz,
            cmd_dyaw,
        ),
        "drone": build_drone_log(data_link),
        "vehicle": build_vehicle_log(vehicle_state, now),
        "alignment": build_alignment_log(frame_aligner),
        "mode": mode_snapshot,
        "tag_lost_duration": log_float(tag_lost_duration, 3),
    }


def main():
    repo_root = Path(__file__).resolve().parent
    flight_log = CodexFlightLogger(
        enabled=CODEX_FLIGHT_LOG_ENABLED,
        log_dir=CODEX_FLIGHT_LOG_DIR,
        sample_interval_s=CODEX_FLIGHT_LOG_SAMPLE_INTERVAL_S,
        event_min_interval_s=CODEX_FLIGHT_LOG_EVENT_MIN_INTERVAL_S,
        flush_interval_s=CODEX_FLIGHT_LOG_FLUSH_INTERVAL_S,
        repo_root=repo_root,
    )
    flight_log.record_session_config(build_session_config_log(flight_log))
    flight_log.record_event(
        "program_start",
        data={"log_path": flight_log.path},
        force=True,
    )

    # ---------- 硬件初始化 ----------
    video_capture = init_camera()
    detector = init_detector()
    data_link = None
    if DATALINK_ENABLED:
        data_link = init_datalink()
    flight_log.record_event(
        "hardware_initialized",
        data={
            "camera": True,
            "apriltag_detector": True,
            "datalink_enabled": DATALINK_ENABLED,
            "datalink_connected": data_link is not None,
        },
        force=True,
    )

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
    flight_log.record_event(
        "vehicle_tcp_config",
        data={"enabled": VEHICLE_TCP_ENABLED},
        force=True,
    )

    # ---------- 调试工具初始化 ----------
    image_dumper, video_converter, udp_sender = init_debug_tools(
        UDP_SENDER_ENABLED,
        UDP_RECEIVER_IP,
        UDP_SENDER_QUALITY,
    )
    flight_log.record_event(
        "debug_tools_initialized",
        data={
            "debug_mode_enabled": DEBUG_MODE_ENABLED,
            "udp_sender_enabled": UDP_SENDER_ENABLED,
            "udp_receiver_ip": UDP_RECEIVER_IP,
            "udp_quality": UDP_SENDER_QUALITY,
        },
        force=True,
    )

    # ---------- 主循环 ----------
    frame_count = 0
    exit_reason = "normal"
    last_tracking_state = {
        "vision_source": "NONE",
        "tracking_source": "NONE",
        "mode": get_control_mode_snapshot(),
        "control_valid": False,
        "tracking_task_ready": True,
        "tracking_wait_reason": None,
    }
    last_loop_time = time.time()
    last_visual_detected_time = time.time()  # 记录最后一次视觉观测成功的时间戳，用于计算视觉丢失持续时长
    last_tracking_frame = None
    target_estimator = TargetEstimator(
        alpha=TARGET_ESTIMATOR_ALPHA,
        beta=TARGET_ESTIMATOR_BETA,
    )
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
            vision_log = {
                "source": "NONE",
                "tag_id": None,
                "pnp_xyz": None,
                "yaw_cmd_deg": None,
            }

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
                    cmd_dyaw, yaw_error_deg, _ = compute_forward_yaw_cmd(pnp_rvec,
                                                                         tag_forward_axis=TAG_FORWARD_AXIS,
                                                                         kp_yaw=FORWARD_YAW_KP,
                                                                         yaw_deadband=FORWARD_YAW_DEADBAND_RAD,
                                                                         max_dyaw=MAX_FORWARD_DYAW_RAD)
                    vision_log = {
                        "source": "TAG",
                        "tag_id": int(target_tag.tag_id),
                        "pnp_xyz": [log_float(pnp_x), log_float(pnp_y), log_float(pnp_z)],
                        "yaw_cmd_deg": log_float(math.degrees(cmd_dyaw), 2),
                        "yaw_error_deg": log_float(yaw_error_deg, 2),
                    }

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
                    cmd_dyaw, yaw_error_deg, _ = compute_forward_yaw_cmd(pnp_rvec,
                                                                         tag_forward_axis=TAG_FORWARD_AXIS,
                                                                         kp_yaw=FORWARD_YAW_KP,
                                                                         yaw_deadband=FORWARD_YAW_DEADBAND_RAD,
                                                                         max_dyaw=MAX_FORWARD_DYAW_RAD)
                    vision_log = {
                        "source": "COLOR",
                        "tag_id": None,
                        "pnp_xyz": [log_float(pnp_x), log_float(pnp_y), log_float(pnp_z)],
                        "yaw_cmd_deg": log_float(math.degrees(cmd_dyaw), 2),
                        "yaw_error_deg": log_float(yaw_error_deg, 2),
                    }

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

                if frame_count % TRACKING_DEBUG_LOG_INTERVAL_FRAMES == 0:
                    logger.debug(
                        f"[RefTrack] {source_text} {frame_text} {veh_vel_text} {veh_pose_text} {residual_info} | "
                        f"cmd=({cmd_dx:+.2f}, {cmd_dy:+.2f}) m | "
                        f"vel=({tracking_result.fused_vel[0]:+.2f}, {tracking_result.fused_vel[1]:+.2f}) m/s"
                    )

            mode_snapshot = get_control_mode_snapshot()
            tracking_task_ready, tracking_wait_reason = get_tracking_task_gate(mode_snapshot, frame_aligner)
            if tracking_wait_reason is not None:
                tracking_debug_lines.append(f"Task gate {tracking_wait_reason}")

            draw_tracking_debug(color_frame, tracking_debug_lines)

            record_flight_events(
                flight_log,
                frame_count,
                vision_log,
                tracking_result,
                last_tracking_frame,
                vehicle_state_cache,
                vehicle_state,
                frame_aligner,
                control_target_valid,
                tracking_task_ready,
                tracking_wait_reason,
                mode_snapshot,
            )
            if flight_log.should_sample(now):
                flight_log.record_sample(
                    build_flight_sample(
                        vision_log,
                        tracking_result,
                        last_tracking_frame,
                        control_target_valid,
                        cmd_dx,
                        cmd_dy,
                        cmd_dz,
                        cmd_dyaw,
                        data_link,
                        vehicle_state,
                        now,
                        frame_aligner,
                        mode_snapshot,
                        tracking_task_ready,
                        tracking_wait_reason,
                        tag_lost_duration,
                    ),
                    frame=frame_count,
                )
            last_tracking_state = {
                "vision_source": vision_log["source"],
                "tracking_source": tracking_source_name(tracking_result),
                "mode": mode_snapshot,
                "control_valid": bool(control_target_valid),
                "tracking_task_ready": bool(tracking_task_ready),
                "tracking_wait_reason": tracking_wait_reason,
                "tracking_frame": last_tracking_frame,
                "tag_lost_duration": log_float(tag_lost_duration, 3),
            }

            if DATALINK_ENABLED:
                handle_control_mode(
                    data_link,
                    control_target_valid,
                    cmd_dx,
                    cmd_dy,
                    cmd_dz,
                    cmd_dyaw,
                    tag_lost_duration,
                    tracking_task_ready=tracking_task_ready,
                    tracking_wait_reason=tracking_wait_reason,
                )

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
                    exit_reason = "debug_window_escape"
                    break

    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        logger.info("收到 Ctrl+C，准备退出主循环")
    except Exception as err:
        exit_reason = f"exception:{type(err).__name__}"
        flight_log.record_event(
            "exception",
            data={"type": type(err).__name__, "message": str(err)},
            frame=frame_count,
            force=True,
        )
        raise
    finally:
        flight_log.close(
            data={
                "exit_reason": exit_reason,
                "frames": frame_count,
                "last_tracking_state": last_tracking_state,
            },
            frame=frame_count,
        )
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
