"""
基于 AprilTag PnP 结果的机体系控制量计算。
"""

import math

import cv2
import numpy as np
from loguru import logger


# ---------- 视觉控制默认调参（DEFAULT） ----------
# main.py 会显式传入主链路参数；这里的默认值用于单独调用本模块函数或旧测试脚本。

# Tag/车辆前向偏航比例增益。调大响应更快但更容易抖，推荐 0.3~0.8。
DEFAULT_FORWARD_YAW_KP = 0.5

# Tag/车辆前向偏航死区，单位 rad。调大更稳但允许残余误差，推荐 0.04~0.12 rad。
DEFAULT_FORWARD_YAW_DEADBAND_RAD = 0.08

# Tag/车辆前向偏航指令限幅，单位 rad。调大修正更快但更激进，推荐 0.3~0.8 rad。
DEFAULT_MAX_FORWARD_DYAW_RAD = 0.6

# compute_control_cmd() 旧接口中的平移比例增益；当前主链路主要使用参考轨迹，不直接用它。
# 调大响应更快但更容易超调，调小更平滑但跟踪偏差收敛慢，推荐 0.4~0.8。
DEFAULT_TRANSLATION_KP_X = 0.6
DEFAULT_TRANSLATION_KP_Y = 0.6
DEFAULT_TRANSLATION_KP_Z = 0.6


def pnp_to_body_xy(pnp_x, pnp_y):
    """
    将 PnP 相机系水平平移转换为机体系水平位置。

    相机系 +X  →  机体系 +Y（右）
    相机系 −Y  →  机体系 +X（前）
    """
    return np.array([-pnp_y, pnp_x], dtype=float)


def camera_xy_to_body_xy(cam_xy):
    """相机系 XY 方向向量转换到机体系水平向量。"""
    cam_xy = np.asarray(cam_xy, dtype=float)
    return np.array([-cam_xy[1], cam_xy[0]], dtype=float)


def estimate_tag_forward_yaw_body(pnp_rvec, tag_forward_axis="+Y"):
    """
    根据 rvec 估计 Tag/车辆前向在机体系下的角度。

    第一版默认 Tag 的 +Y 轴与无人车车头方向一致；若实物安装方向不同，
    后续可把 tag_forward_axis 改成 +X、-X、+Y、-Y。
    """
    R, _ = cv2.Rodrigues(pnp_rvec)
    axis_map = {
        "+X": R[:, 0],
        "-X": -R[:, 0],
        "+Y": R[:, 1],
        "-Y": -R[:, 1],
    }
    tag_axis = axis_map.get(tag_forward_axis)
    if tag_axis is None:
        raise ValueError(f"未知 tag_forward_axis: {tag_forward_axis}")

    body_xy = camera_xy_to_body_xy(tag_axis[:2])
    norm = np.linalg.norm(body_xy)
    if norm < 1e-6:
        return 0.0
    body_xy = body_xy / norm
    return math.atan2(float(body_xy[1]), float(body_xy[0]))


def compute_forward_yaw_cmd(pnp_rvec,
                            tag_forward_axis="+Y",
                            kp_yaw=DEFAULT_FORWARD_YAW_KP,
                            yaw_deadband=DEFAULT_FORWARD_YAW_DEADBAND_RAD,
                            max_dyaw=DEFAULT_MAX_FORWARD_DYAW_RAD):
    """
    利用 solvePnP 解算出的旋转向量 rvec，让无人机机头对齐 Tag/车辆前向轴。

    当前工程约定 TAG_FORWARD_AXIS="+Y"，即 Tag +Y 方向就是无人车车头方向。
    这里不再选择“最近垂直边”，而是固定使用该前向轴作为偏航目标，保证偏航控制、
    车端速度前馈和 FrameAligner 坐标系 yaw 对齐使用同一套方向语义。

    坐标系约定（与 compute_control_cmd 保持一致）
    -----------------------------------------
        相机系 +X  →  机体系 +Y（右）    body_dy =  pnp_x
        相机系 −Y  →  机体系 +X（前）    body_dx = −pnp_y
        机体系 dyaw > 0  →  从上往下看顺时针旋转

    参数
    ----
    tag_forward_axis: Tag 坐标系中代表车辆前向的轴，默认 +Y
    kp_yaw          : 比例增益
    yaw_deadband    : 偏航死区（弧度）；误差小于此值时输出 0，消除小幅抖动
    max_dyaw        : 单帧最大偏航指令（弧度）

    返回
    ----
    cmd_dyaw        : 偏航角控制指令（弧度），正值=顺时针
    yaw_error_deg   : 原始偏航误差（度），用于日志
    tag_forward_axis: 本帧使用的 Tag 前向轴
    """
    yaw_error = estimate_tag_forward_yaw_body(pnp_rvec, tag_forward_axis)

    if abs(yaw_error) < yaw_deadband:
        cmd_dyaw = 0.0
    else:
        cmd_dyaw = kp_yaw * yaw_error
        cmd_dyaw = float(np.clip(cmd_dyaw, -max_dyaw, max_dyaw))

    yaw_error_deg = math.degrees(yaw_error)
    logger.debug(
        f"[Yaw] axis={tag_forward_axis} | err={yaw_error_deg:+.1f}° | cmd={cmd_dyaw:+.3f} rad"
    )
    return cmd_dyaw, yaw_error_deg, tag_forward_axis


def compute_control_cmd(pnp_x, pnp_y, pnp_z, pnp_rvec):
    """
    将相机坐标系下的位移误差和 Tag 姿态转换为机体系控制指令。
    返回 (cmd_dx, cmd_dy, cmd_dz, cmd_dyaw)。

    偏航控制固定对齐 Tag/车辆前向轴，默认 Tag +Y 方向代表无人车车头方向。
    """
    # =========================
    # 相机坐标 → 机体系（平移）
    # =========================
    body_dx, body_dy = pnp_to_body_xy(pnp_x, pnp_y)
    body_dz = -pnp_z

    # =========================
    # 控制器参数（平移）
    # =========================
    kp_x = DEFAULT_TRANSLATION_KP_X
    kp_y = DEFAULT_TRANSLATION_KP_Y
    kp_z = DEFAULT_TRANSLATION_KP_Z
    kp_yaw = DEFAULT_FORWARD_YAW_KP

    cmd_dx = kp_x * body_dx
    cmd_dy = kp_y * body_dy
    cmd_dz = kp_z * body_dz

    # =========================
    # 偏航控制：对齐 Tag/车辆前向轴
    # =========================
    cmd_dyaw, _, _ = compute_forward_yaw_cmd(pnp_rvec,
                                             tag_forward_axis="+Y",
                                             kp_yaw=kp_yaw,
                                             yaw_deadband=DEFAULT_FORWARD_YAW_DEADBAND_RAD,
                                             max_dyaw=DEFAULT_MAX_FORWARD_DYAW_RAD)

    return cmd_dx, cmd_dy, cmd_dz, cmd_dyaw
