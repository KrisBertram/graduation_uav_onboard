"""
无人车坐标系到无人机局部坐标系的在线对齐。
"""

import math

import numpy as np
from loguru import logger


def wrap_angle(angle):
    """角度规约到 [-pi, pi]。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rotate_xy(vec_xy, yaw):
    """二维向量旋转。"""
    x, y = float(vec_xy[0]), float(vec_xy[1])
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        c * x - s * y,
        s * x + c * y,
    ], dtype=float)


def blend_angle(old, new, alpha):
    """按最短角差低通融合角度。"""
    return wrap_angle(old + alpha * wrap_angle(new - old))


class FrameAligner:
    """
    估计二维变换 T_drone_from_vehicle = R(theta) + t。

    第一帧视觉有效时直接锁定变换；之后每次看到 Tag 都用低通方式修正。
    """

    def __init__(self, yaw_alpha=0.05, pos_alpha=0.05):
        self.yaw_alpha = yaw_alpha
        self.pos_alpha = pos_alpha
        self.theta = 0.0
        self.translation = np.zeros(2, dtype=float)
        self.initialized = False

    def update(self, vehicle_xy, vehicle_yaw_rad, target_xy_drone, tag_yaw_drone):
        """
        用同一时刻的车端位姿和视觉观测更新坐标系对齐。

        vehicle_xy      : 无人车坐标系下车辆参考点位置 [x, y]
        vehicle_yaw_rad : 无人车坐标系下车头朝向，逆时针为正
        target_xy_drone : 视觉观测得到的 Tag/平台中心在无人机局部系的位置
        tag_yaw_drone   : 视觉观测得到的 Tag/车头朝向在无人机局部系的角度
        """
        vehicle_xy = np.asarray(vehicle_xy, dtype=float)
        target_xy_drone = np.asarray(target_xy_drone, dtype=float)

        theta_meas = wrap_angle(tag_yaw_drone - vehicle_yaw_rad)
        translation_meas = target_xy_drone - rotate_xy(vehicle_xy, theta_meas)

        if not self.initialized:
            self.theta = theta_meas
            self.translation = translation_meas
            self.initialized = True
            logger.info(
                f"坐标系首次对齐: theta={math.degrees(self.theta):+.1f} deg, "
                f"t=({self.translation[0]:+.2f}, {self.translation[1]:+.2f}) m"
            )
            return

        self.theta = blend_angle(self.theta, theta_meas, self.yaw_alpha)
        self.translation = (
            (1.0 - self.pos_alpha) * self.translation
            + self.pos_alpha * translation_meas
        )
        logger.debug(
            f"坐标系在线修正: theta={math.degrees(self.theta):+.1f} deg, "
            f"t=({self.translation[0]:+.2f}, {self.translation[1]:+.2f}) m"
        )

    def transform_point(self, vehicle_xy):
        """无人车坐标系点 -> 无人机局部坐标系点。"""
        return rotate_xy(vehicle_xy, self.theta) + self.translation

    def transform_vector(self, vehicle_vec_xy):
        """无人车坐标系向量 -> 无人机局部坐标系向量。"""
        return rotate_xy(vehicle_vec_xy, self.theta)

    def vehicle_velocity_to_drone(self, speed, vehicle_yaw_rad):
        """由车端 speed+yaw 得到无人机局部坐标系下的前馈速度。"""
        vehicle_vel = np.array([
            speed * math.cos(vehicle_yaw_rad),
            speed * math.sin(vehicle_yaw_rad),
        ], dtype=float)
        return self.transform_vector(vehicle_vel)
