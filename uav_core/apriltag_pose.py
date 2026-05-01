"""
AprilTag 参数、目标选择与 PnP 位姿解算。
"""

import cv2
import numpy as np
from loguru import logger
from pupil_apriltags import Detector


# =========================================================
# 相机内参（Matlab 标定结果）
# =========================================================
"""
内参矩阵：
    [ fx    0   cx ]
    [  0   fy   cy ]
    [  0    0    1 ]
"""
cameraMatrix = np.array([
    [1288.028590,         0.0, 522.451003],
    [        0.0, 1294.004313, 254.652409],
    [        0.0,         0.0,        1.0]
], dtype=np.float32)

cameraMatrix[0, 2] = 960 / 2
cameraMatrix[1, 2] = 540 / 2

"""
径向畸变：
    [ k1, k2, k3 ]
切向畸变：
    [ p1, p2 ]
畸变参数：
    [ k1, k2, p1, p2, k3 ]
"""
distCoeffs = np.array([
    0.130990,   # k1
    0.305167,   # k2
   -0.003977,   # p1
    0.006610,   # p2
   -2.142455    # k3
], dtype=np.float32)


# =========================================================
# AprilTag 参数
# =========================================================
TAG_FAMILY = "tagCustom48h12"  # 使用哪种 AprilTag 家族

# v3: 改为嵌套 AprilTag，各层 ID 对应的实际边长（单位：米）
# 当前使用 A3 复合标志生成脚本的理想物理尺寸；新打印机输出已与理想尺寸一致。
# 内层 Tag 按嵌套比例等比例缩放。
# id65：最外层（最大），id66：中间层，id67：最内层（最小）
TAG_SIZES = {
    65: 0.200,  # 最外层，边长 0.200 m
    66: 0.040,  # 中间层，边长 0.040 m
    67: 0.008,  # 最内层，边长 0.008 m
}

# v3: 跟踪优先级列表，优先跟踪 id65，其次 id66，最后 id67
TAG_PRIORITY = [65, 66, 67]


def get_object_points(tag_size):
    """
    根据给定的标签边长，生成 AprilTag 四个角点的三维坐标（相对于标签中心）。
    v3: 因各层嵌套 AprilTag 边长不同，改为按需动态生成，不再使用全局固定值。
    """
    half = tag_size * 0.5
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0]
    ], dtype=np.float32)


def init_detector():
    """创建并返回 AprilTag Detector 实例"""
    return Detector(
        families=TAG_FAMILY,    # 使用哪种 AprilTag 家族
        nthreads=4,             # 使用多少 CPU 线程
        quad_decimate=1.0,      # 图像降采样比例，1.0 表示原分辨率，> 1.0 会更快，但远距离精度下降
        quad_sigma=0.0,         # 高斯模糊参数，0.0 表示不模糊
        refine_edges=True,      # 是否细化边缘检测
        # decode_sharpening=0.25  # 解码锐化参数
    )


def select_target_tag(april_tags):
    """
    按照 TAG_PRIORITY 选择本帧优先级最高的目标 Tag。

    跟踪规则：有 id65 跟 id65；没有 id65 跟 id66；连 id66 也没有则跟 id67；
    否则视为丢失 AprilTag 视野。
    """
    detected_tag_map = {t.tag_id: t for t in april_tags if t.tag_id in TAG_SIZES}

    for priority_id in TAG_PRIORITY:
        if priority_id in detected_tag_map:
            return detected_tag_map[priority_id]
    return None


def estimate_pose(tag, tag_size):
    """
    对单个 AprilTag 进行 solvePnP 位姿解算。
    v3: 新增 tag_size 参数，用于根据各层嵌套 AprilTag 的实际边长动态生成 object_points。
    返回 (ok, rvec, tvec, x, y, z, image_points)，失败时 ok=False。
    """
    image_points = tag.corners.astype(np.float32)  # 提取角点

    # v3: 根据当前 tag 的实际边长动态生成三维角点坐标
    object_points = get_object_points(tag_size)

    # 使用 solvePnP 估计位姿
    pnp_ok, pnp_rvec, pnp_tvec = cv2.solvePnP(
        object_points,                  # 三维点
        image_points,                   # 二维点
        cameraMatrix,                   # 相机内参矩阵
        distCoeffs,                     # 畸变系数
        flags=cv2.SOLVEPNP_IPPE_SQUARE  # 使用适合方形标志的算法
    )

    if not pnp_ok:
        logger.warning(f"Tag {tag.tag_id} solvePnP 失败")
        return False, None, None, None, None, None, None

    pnp_x, pnp_y, pnp_z = pnp_tvec.flatten()  # 提取平移向量
    return True, pnp_rvec, pnp_tvec, pnp_x, pnp_y, pnp_z, image_points
