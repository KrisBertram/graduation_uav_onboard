"""
目标状态估计、前视预测和无人机参考点生成。
"""

import numpy as np


def clamp_norm(vec, max_norm):
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec * (max_norm / norm)


def body_to_local_xy(body_xy, drone_yaw):
    """
    机体系水平向量 -> 无人机局部坐标系水平向量。

    与 DataLink.set_pose() 中机体系偏移转换公式保持一致。
    """
    body_xy = np.asarray(body_xy, dtype=float)
    dx, dy = float(body_xy[0]), float(body_xy[1])
    c = np.cos(drone_yaw)
    s = np.sin(drone_yaw)
    return np.array([
        dx * c - dy * s,
        dx * s + dy * c,
    ], dtype=float)


def local_to_body_xy(local_xy, drone_yaw):
    """无人机局部坐标系水平向量 -> 机体系水平向量。"""
    local_xy = np.asarray(local_xy, dtype=float)
    gx, gy = float(local_xy[0]), float(local_xy[1])
    c = np.cos(drone_yaw)
    s = np.sin(drone_yaw)
    return np.array([
        gx * c + gy * s,
       -gx * s + gy * c,
    ], dtype=float)


class TargetEstimator:
    """alpha-beta 目标状态估计器。"""

    def __init__(self, alpha=0.35, beta=0.08):
        self.alpha = alpha
        self.beta = beta
        self.pos = np.zeros(2, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.last_timestamp = None
        self.initialized = False
        self.last_fused_vel = np.zeros(2, dtype=float)

    def reset(self):
        self.pos[:] = 0.0
        self.vel[:] = 0.0
        self.last_timestamp = None
        self.initialized = False
        self.last_fused_vel[:] = 0.0

    def update_measurement(self, measurement_xy, timestamp):
        measurement_xy = np.asarray(measurement_xy, dtype=float)

        if not self.initialized:
            self.pos = measurement_xy.copy()
            self.vel = np.zeros(2, dtype=float)
            self.last_timestamp = timestamp
            self.initialized = True
            return self.pos.copy(), self.vel.copy()

        dt = max(1e-3, min(timestamp - self.last_timestamp, 0.5))
        pred_pos = self.pos + self.vel * dt
        residual = measurement_xy - pred_pos

        self.pos = pred_pos + self.alpha * residual
        self.vel = self.vel + self.beta * residual / dt
        self.last_timestamp = timestamp
        return self.pos.copy(), self.vel.copy()

    def predict(self, timestamp):
        if not self.initialized:
            return None, None
        dt = max(0.0, min(timestamp - self.last_timestamp, 0.5))
        return (self.pos + self.vel * dt).copy(), self.vel.copy()

    def make_future_point(
        self,
        timestamp,
        lookahead_time_s,
        vehicle_vel_xy=None,
        vision_weight=0.4,
        vehicle_weight=0.6,
    ):
        pos, vel = self.predict(timestamp)
        if pos is None:
            return None, None, False

        used_vehicle = vehicle_vel_xy is not None
        if used_vehicle:
            vehicle_vel_xy = np.asarray(vehicle_vel_xy, dtype=float)
            fused_vel = vision_weight * vel + vehicle_weight * vehicle_vel_xy
        else:
            fused_vel = vel

        self.last_fused_vel = fused_vel.copy()
        future_pos = pos + fused_vel * lookahead_time_s
        return future_pos, fused_vel, used_vehicle


class ReferenceTrajectory:
    """把预测目标点变成限速平滑的参考点。"""

    def __init__(self, max_speed_mps=0.8):
        self.max_speed_mps = max_speed_mps
        self.ref_pos = None

    def reset(self):
        self.ref_pos = None

    def update(self, target_pos, dt):
        target_pos = np.asarray(target_pos, dtype=float)
        if self.ref_pos is None:
            self.ref_pos = target_pos.copy()
            return self.ref_pos.copy()

        max_step = max(0.0, self.max_speed_mps * dt)
        step = clamp_norm(target_pos - self.ref_pos, max_step)
        self.ref_pos = self.ref_pos + step
        return self.ref_pos.copy()
