"""
无人车状态包解析、状态缓存和可选 TCP 接入。
"""

import math
import struct
import threading
import time
from dataclasses import dataclass

from loguru import logger

from kb_wifi_connect import TCP_HOST, TCP_PORT, TCPServer


VEHICLE_STATE_CMD = 0x01


@dataclass
class VehicleState:
    """无人车上行状态，字段顺序与参考工程 0x01 协议一致。"""

    speed: float
    distance: float
    yaw: float
    pitch: float
    roll: float
    pos_x: float
    pos_y: float
    pos_z: float
    action: int
    timestamp: float = 0.0

    FMT = "<ffffffffH"
    SIZE = struct.calcsize(FMT)

    @property
    def yaw_rad(self):
        """无人车 yaw：车端上行为 deg，逆时针为正。"""
        return math.radians(self.yaw)

    @classmethod
    def from_bytes(cls, payload: bytes, timestamp=None):
        if len(payload) != cls.SIZE:
            raise ValueError(f"无人车 0x01 payload 长度错误: {len(payload)} != {cls.SIZE}")
        values = struct.unpack(cls.FMT, payload)
        if timestamp is None:
            timestamp = time.time()
        return cls(*values, timestamp=timestamp)


class VehicleStateCache:
    """线程安全保存最新无人车状态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = None

    def update(self, state: VehicleState):
        with self._lock:
            self._state = state

    def get_latest(self, now=None, max_age=None):
        with self._lock:
            state = self._state

        if state is None:
            return None

        if max_age is not None:
            if now is None:
                now = time.time()
            if now - state.timestamp > max_age:
                return None

        return state


class VehicleStateReceiver:
    """
    后台 TCP 接收器。

    注意：不会主动连接 WiFi；只有 VEHICLE_TCP_ENABLED=True 时才启动。
    TCPServer.start() 内部会阻塞等待 ESP8266，因此这里放到 daemon 线程中运行。
    """

    def __init__(self, host=TCP_HOST, port=TCP_PORT, cache=None):
        self.cache = cache if cache is not None else VehicleStateCache()
        self.server = TCPServer(
            host=host,
            port=port,
            data_process_callback=self._on_packet,
        )
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info("无人车 TCP 接收线程已启动，等待车辆端连接...")

    def _run_server(self):
        try:
            self.server.start()
        except Exception as err:
            logger.error(f"无人车 TCP 接收器异常退出: {err}")

    def _on_packet(self, cmd, data, length):
        if cmd != VEHICLE_STATE_CMD:
            logger.debug(f"忽略未知车端数据包 cmd=0x{cmd:02X}, len={length}")
            return

        try:
            state = VehicleState.from_bytes(data)
        except Exception as err:
            logger.warning(f"无人车状态包解析失败: {err}")
            return

        self.cache.update(state)
        logger.debug(
            f"[Vehicle] speed={state.speed:+.2f} m/s, yaw={state.yaw:+.1f} deg, "
            f"pos=({state.pos_x:+.2f}, {state.pos_y:+.2f})"
        )
