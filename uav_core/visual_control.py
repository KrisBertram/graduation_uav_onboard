"""
基于 AprilTag PnP 结果的机体系控制量计算。
"""

import math

import cv2
import numpy as np
from loguru import logger


# 上一帧选定的边法向编号（0=+R[:,0], 1=−R[:,0], 2=+R[:,1], 3=−R[:,1]）
# 用于帧间滞后：候选切换需超过 HYSTERESIS_BONUS 的点积优势，防止选边抖动
_yaw_locked_edge_idx = None


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


def compute_yaw_cmd(pnp_rvec,
                    kp_yaw=0.5,
                    yaw_deadband=0.04,
                    max_dyaw=0.35,
                    hysteresis_bonus=0.15):
    """
    利用 solvePnP 解算出的旋转向量 rvec，计算让无人机机头与 AprilTag 某条边垂直所需的偏航指令。

    核心思路
    --------
    下视相机能拍到 AprilTag 意味着无人机已足够接近，此时用 atan2(body_dy, body_dx)
    会因近距离的微小平移引起偏航大幅抖动。改用 rvec 描述的 AprilTag 姿态信息更稳定：
        1. 由 rvec 得到旋转矩阵 R（Tag 坐标系 → 相机坐标系）。
        2. 提取 Tag 的 X 轴和 Y 轴在相机系中的方向（即 R 的列向量）。
        3. AprilTag 四条边各有一个法方向：±R[:,0]（垂直于左/右边）和 ±R[:,1]（垂直于上/下边）。
        4. 在相机 XY 平面上投影，找出与机体正前方 (0,−1) 点积最大（夹角最小）的那个。
        5. 计算需要顺时针旋转多少角度 θ 才能让机体正前方对齐该方向。
        6. cmd_dyaw = kp_yaw × θ，加死区 + 限幅。

    坐标系约定（与 compute_control_cmd 保持一致）
    -----------------------------------------
        相机系 +X  →  机体系 +Y（右）    body_dy =  pnp_x
        相机系 −Y  →  机体系 +X（前）    body_dx = −pnp_y
        机体系 dyaw > 0  →  从上往下看顺时针旋转

    选边滞后（hysteresis）
    ----------------------
    当前帧选中的边编号写入全局 _yaw_locked_edge_idx；下一帧中，若某候选与上帧相同，
    在点积比较时加上 hysteresis_bonus 的奖励，避免在两候选接近时来回抖动。

    参数
    ----
    kp_yaw          : 比例增益
    yaw_deadband    : 偏航死区（弧度）；误差小于此值时输出 0，消除小幅抖动
    max_dyaw        : 单帧最大偏航指令（弧度）
    hysteresis_bonus: 对上帧已选方向的点积奖励，建议 0.1~0.2

    返回
    ----
    cmd_dyaw        : 偏航角控制指令（弧度），正值=顺时针
    yaw_error_deg   : 原始偏航误差（度），用于日志
    edge_idx        : 本帧选定的候选编号（0=+R[:,0], 1=−R[:,0], 2=+R[:,1], 3=−R[:,1]）
    """
    global _yaw_locked_edge_idx

    # ── 1. 旋转矩阵：Tag 坐标系 → 相机坐标系 ──
    R, _ = cv2.Rodrigues(pnp_rvec)

    tag_x_in_cam = R[:, 0]   # (3,) Tag X 轴在相机系
    tag_y_in_cam = R[:, 1]   # (3,) Tag Y 轴在相机系

    # ── 2. 4 个候选"垂直于边"的方向，仅取 XY 分量 ──
    # 编号定义：0=+tag_X  1=−tag_X  2=+tag_Y  3=−tag_Y
    candidates_xy = [
         tag_x_in_cam[:2],   # 0: +R[:,0]
        -tag_x_in_cam[:2],   # 1: −R[:,0]
         tag_y_in_cam[:2],   # 2: +R[:,1]
        -tag_y_in_cam[:2],   # 3: −R[:,1]
    ]

    # ── 3. 当前机体正前方在相机 XY 中的方向：(0, −1) ──
    body_fwd = np.array([0.0, -1.0])

    best_dot = -np.inf
    best_idx = 0
    best_dir = body_fwd  # 保底：不旋转

    for i, c in enumerate(candidates_xy):
        n = np.linalg.norm(c)
        if n < 1e-6:
            continue
        c_norm = c / n
        dot = float(np.dot(c_norm, body_fwd))

        # 对上帧选中方向给予滞后奖励
        if i == _yaw_locked_edge_idx:
            dot += hysteresis_bonus

        if dot > best_dot:
            best_dot = dot
            best_idx = i
            best_dir = c_norm

    _yaw_locked_edge_idx = best_idx

    # ── 4. 偏航误差：顺时针角度，使机体正前方转向 best_dir ──
    # 推导：从 (0,−1) 顺时针旋转 θ 后的新方向 = (sin θ, −cos θ)
    # 令 (sin θ, −cos θ) = best_dir  →  θ = atan2(best_dir[0], −best_dir[1])
    yaw_error = math.atan2(float(best_dir[0]), float(-best_dir[1]))

    # ── 5. 死区 + 比例控制 + 限幅 ──
    if abs(yaw_error) < yaw_deadband:
        cmd_dyaw = 0.0
    else:
        cmd_dyaw = kp_yaw * yaw_error
        cmd_dyaw = float(np.clip(cmd_dyaw, -max_dyaw, max_dyaw))

    yaw_error_deg = math.degrees(yaw_error)
    logger.debug(
        f"[Yaw] edge={best_idx} | err={yaw_error_deg:+.1f}° | cmd={cmd_dyaw:+.3f} rad"
    )
    return cmd_dyaw, yaw_error_deg, best_idx


def compute_control_cmd(pnp_x, pnp_y, pnp_z, pnp_rvec):
    """
    将相机坐标系下的位移误差和 Tag 姿态转换为机体系控制指令。
    返回 (cmd_dx, cmd_dy, cmd_dz, cmd_dyaw)。

    v4 新增：通过 compute_yaw_cmd(pnp_rvec) 自动对齐 AprilTag 边的法线方向，
    替代原来 body_dyaw=0 的空实现。
    """
    # =========================
    # 相机坐标 → 机体系（平移）
    # =========================
    body_dx, body_dy = pnp_to_body_xy(pnp_x, pnp_y)
    body_dz = -pnp_z

    # =========================
    # 控制器参数（平移）
    # =========================
    kp_x = 0.6
    kp_y = 0.6
    kp_z = 0.6
    kp_yaw = 0.5  # v4: 新增偏航控制增益

    cmd_dx = kp_x * body_dx
    cmd_dy = kp_y * body_dy
    cmd_dz = kp_z * body_dz

    # =========================
    # 偏航控制：对齐 AprilTag 某条边的法线方向
    # =========================
    cmd_dyaw, _, _ = compute_yaw_cmd(pnp_rvec,
                                     kp_yaw=kp_yaw,
                                     yaw_deadband=0.08,
                                     max_dyaw=0.6,
                                     hysteresis_bonus=0.15)

    return cmd_dx, cmd_dy, cmd_dz, cmd_dyaw
