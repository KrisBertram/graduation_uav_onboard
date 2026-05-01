"""
DataLink 模块
-----------------------------------------
功能：
1. 通过串口与飞控进行 MAVLink 通信
2. 接收飞控状态信息
3. 发送位置 / 姿态 / 起降等控制指令，姿态单位：弧度
"""

import sys
import time
import serial
import math
import mavlink.mavlink as mavlink
from loguru import logger

class DataLink:
    """
    与飞控通信的核心类
    """

    # ==============================
    # 飞控反馈状态结构
    # ==============================
    class DroneState:
        """
        存储从飞控接收到的实时状态
        """

        def __init__(self):
            # 位置（LOCAL_NED 坐标系，单位：米）
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

            # 位置（LOCAL_NED 坐标系，原始数据，单位：厘米，可供精确使用）
            self.x_cm = 0.0
            self.y_cm = 0.0
            self.z_cm = 0.0

            # 姿态（单位：弧度）
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0

            # 相对高度（单位：米 / 原始数据单位：毫米）
            self.relative_alt = 0.0
            self.relative_alt_mm = 0.0

            # 电池信息
            self.battery_voltage = 0.0  # V
            self.battery_current = 0.0  # A

            # 心跳信息
            self.heartbeat_count = 0  # 心跳次数
            self.last_heartbeat_time = 0  # 上次心跳时间
            self.time_since_last_heartbeat = 0  # 距离上次心跳的时间


    # ==============================
    # 控制目标结构
    # ==============================
    class DroneTarget:
        """
        存储准备发送给飞控的控制目标
        """

        def __init__(self):
            # 目标位置（LOCAL_NED 坐标系，单位：米）
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

            # 目标姿态（单位：弧度）
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0


    # ==============================
    # 初始化
    # ==============================
    def __init__(self, port='/dev/ttyTHS0', baudrate=460800):

        # 串口
        self.serial_port = serial.Serial(port, baudrate, timeout=1.0)

        # MAVLink 实例
        self.mavlink = None

        # 当前接收到的 MAVLink 消息
        self.message = None

        # 无人机状态、目标
        self.state = self.DroneState()
        self.target = self.DroneTarget()


    # ==============================
    # MAVLink 初始化
    # ==============================
    class SerialWrapper:
        """
        让 MAVLink 可以写入串口的包装器
        """
        def __init__(self, file):
            self.file = file
            self.buf = []
        def write(self, data):
            self.file.write(data)

    def init_mavlink(self):
        """
        初始化 MAVLink 通信
        """
        wrapper = self.SerialWrapper(self.serial_port)
        self.mavlink = mavlink.MAVLink(wrapper)


    # ==============================
    # 接收线程
    # ==============================
    def receive_loop(self):
        """
        持续读取串口并解析 MAVLink 消息
        """
        while True:
            if self.serial_port.is_open:
                try:
                    byte = self.serial_port.read(1)
                except Exception as err:
                    logger.error(f"读取串口时发生异常: {err}")
                    continue
            else:
                logger.error("串口未打开")
                continue

            if byte == b'': continue
            if ord(byte) < 0: continue

            try:
                msg = self.mavlink.parse_char(byte)
                if msg is None: continue

                self.message = msg
                msg_id = msg.get_msgId()

                # 心跳
                if msg_id == mavlink.MAVLINK_MSG_ID_HEARTBEAT:
                    if self.state.heartbeat_count < 1e6: self.state.heartbeat_count += 1
                    self.state.time_since_last_heartbeat = time.time() - self.state.last_heartbeat_time
                    self.state.last_heartbeat_time = time.time()
                    # logger.debug(f"心跳更新:\n\
                    #             心跳次数 = {self.state.heartbeat_count}\n\
                    #             距离上次心跳的时间 = {self.state.time_since_last_heartbeat:.6f} s")

                # 位置
                elif msg_id == mavlink.MAVLINK_MSG_ID_GLOBAL_VISION_POSITION_ESTIMATE:
                    if isinstance(msg, mavlink.MAVLink_global_vision_position_estimate_message):
                        self.state.x = msg.x * 0.01
                        self.state.y = msg.y * 0.01
                        self.state.z = msg.z * 0.01

                        self.state.x_cm = msg.x
                        self.state.y_cm = msg.y
                        self.state.z_cm = msg.z

                        self.state.roll = msg.roll
                        self.state.pitch = msg.pitch
                        self.state.yaw = msg.yaw

                        # logger.debug(f"位置更新:\n\
                        #             Position = ({self.state.x:.2f}, {self.state.y:.2f}, {self.state.z:.2f}) m\n\
                        #             Attitude = ({math.degrees(self.state.roll):.2f}, {math.degrees(self.state.pitch):.2f}, {math.degrees(self.state.yaw):.2f}) deg")

                # 高度
                elif msg_id == mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
                    if isinstance(msg, mavlink.MAVLink_global_position_int_message):
                        self.state.relative_alt = msg.relative_alt * 0.001
                        self.state.relative_alt_mm = msg.relative_alt
                        # logger.debug(f"高度更新:\nRelative Altitude = {self.state.relative_alt:.2f} m")

                # 电池
                elif msg_id == mavlink.MAVLINK_MSG_ID_BATTERY_STATUS:
                    if isinstance(msg, mavlink.MAVLink_battery_status_message):
                        self.state.battery_voltage = float(msg.voltages[1] * 0.001)
                        self.state.battery_current = float(msg.current_battery * 0.01)
                        # logger.debug(f"电池更新:\n\
                        #             Voltage = {self.state.battery_voltage:.2f} V\n\
                        #             Current = {self.state.battery_current:.2f} A")
            except Exception as err:
                self.message = None
                logger.debug(f"解析 MAVLink 消息时发生异常: {err}")
                continue


    # ==============================
    # 基础控制指令
    # ==============================

    def send_heartbeat(self):
        """发送心跳消息，保持与飞控的连接"""
        while True:
            self.mavlink.heartbeat_send(mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, mavlink.MAV_MODE_FLAG_MANUAL_INPUT_ENABLED, 0, mavlink.MAV_STATE_STANDBY, 1)
            time.sleep(1)


    def set_arm(self):
        """解锁电机"""
        self.mavlink.set_mode_send(0, mavlink.MAV_MODE_AUTO_ARMED, 0)


    def set_disarm(self):
        """锁定电机"""
        self.mavlink.set_mode_send(0, mavlink.MAV_MODE_AUTO_DISARMED, 0)


    def set_takeoff(self, altitude=0):
        """
        起飞
        
        参数：
        altitude : 目标高度（米），默认 0 米
        """
        self.mavlink.command_long_send(
            0, 0,  # target_system, target_component
            mavlink.MAV_CMD_NAV_TAKEOFF,
            0,  # confirmation
            0, 0, 0, 0,  # param1-4: 未使用
            0, 0,  # param5-6: 纬度、经度（未使用）
            altitude  # param7: 目标高度
        )


    def set_land(self):
        """降落"""
        self.mavlink.command_long_send(
            0, 0,  # target_system, target_component
            mavlink.MAV_CMD_NAV_LAND,
            0,  # confirmation
            0, 0, 0, 0,  # param1-4: 未使用
            0, 0, 0  # param5-7: 纬度、经度、高度（未使用）
        )


    # ==============================
    # 位置控制指令
    # ==============================

    def set_pose(self, dx, dy, dz, dyaw, ignore_z=False, direct_z=False):
        """
        在 FRD 机体系坐标系中设置相对目标位置
        以飞行器当前位置为原点，
        X 轴正方向是机体前方, dx > 0: 向前移动；
        Y 轴正方向是集体右侧, dy > 0: 向右移动；
        Z 轴正方向是机体上方, dz > 0: 向上移动；
        偏航角正方向是顺时针, dyaw > 0: 顺时针旋转；
        
        参数：
        dx, dy, dz : 机体系相对位置偏移（米）, dx: 前/后, dy: 右/左, dz: 上/下
        dyaw       : 相对航向角偏移（弧度）
        ignore_z    : 是否忽略高度偏移（默认否）
        direct_z    : 是否直接使用 dz 作为目标高度（默认否）
        """
        
        if (self.state.x != 0) and (self.state.y != 0) and (self.state.yaw != 0):
            # 将机体系偏移转换为全局系
            global_dx = dx * math.cos(self.state.yaw) - dy * math.sin(self.state.yaw)
            global_dy = dx * math.sin(self.state.yaw) + dy * math.cos(self.state.yaw)
            global_dz = dz
        
            self.target.x = self.state.x + global_dx
            self.target.y = self.state.y + global_dy
            self.target.z = self.state.z + global_dz
            self.target.yaw = self.state.yaw + dyaw

            if direct_z: self.target.z = dz  # 直接使用 dz 作为目标高度
            if ignore_z: self.target.z = 0  # 发 0.3 米以下高度都默认不生效，维持原有高度

            logger.debug(f"dx = {dx:.2f} m, dy = {dy:.2f} m, dz = {dz:.2f} m, dyaw = {math.degrees(dyaw):.2f} deg\n\
                        转换为全局偏移: global_dx = {global_dx:.2f} m, global_dy = {global_dy:.2f} m, global_dz = {global_dz:.2f} m\n\
                        目标位置: ({self.target.x:.2f}, {self.target.y:.2f}, {self.target.z:.2f}) m, 目标航向: {math.degrees(self.target.yaw):.2f} deg")
        
            # type_mask: 忽略速度、加速度、yaw_rate
            # 8(VX) | 16(VY) | 32(VZ) | 64(AX) | 128(AY) | 256(AZ) | 2048(YAW_RATE) = 2552
            type_mask = 0

            self.mavlink.set_position_target_local_ned_send(
                0,  # time_boot_ms
                0, 0,  # target_system, target_component
                mavlink.MAV_FRAME_GLOBAL,  # coordinate_frame
                type_mask,  # type_mask
                self.target.x, self.target.y, self.target.z,  # 机体系相对位置偏移
                0, 0, 0,  # 速度 (忽略)
                0, 0, 0,  # 加速度 (忽略)
                self.target.yaw,  # 相对 yaw
                0  # yaw_rate (忽略)
            )


    def set_attitude_altitude(self, roll=0, pitch=0, yaw=0, altitude=0):
        """
        设置目标姿态和高度

        参数：
        roll, pitch, yaw : 目标姿态（弧度）
        altitude         : 目标高度（米）
        """
        self.target.roll = roll
        self.target.pitch = pitch
        self.target.yaw = yaw
        self.target.z = altitude

        self.mavlink.set_position_target_local_ned_send(
            0,  # time_boot_ms
            0, 0,  # target_system, target_component
            mavlink.MAV_FRAME_BODY_NED,  # coordinate_frame
            0,  # type_mask
            0, 0, self.target.z,  # 全局位置 (z轴使用目标高度)
            0, 0, 0,  # 速度 (忽略)
            self.target.roll, self.target.pitch, 0,  # 加速度 (忽略)
            self.target.yaw,  # 相对 yaw
            0  # yaw_rate (忽略)
        )


    # ==============================
    # 便捷控制函数（机体系）
    # ==============================

    def move_forward(self, distance):
        """向前移动指定距离（米）"""
        self.set_pose(distance, 0, 0, 0)


    def move_backward(self, distance):
        """向后移动指定距离（米）"""
        self.set_pose(-distance, 0, 0, 0)


    def move_right(self, distance):
        """向右移动指定距离（米）"""
        self.set_pose(0, distance, 0, 0)


    def move_left(self, distance):
        """向左移动指定距离（米）"""
        self.set_pose(0, -distance, 0, 0)


    def move_up(self, distance):
        """向上移动指定距离（米）"""
        self.set_pose(0, 0, distance, 0)


    def move_down(self, distance):
        """向下移动指定距离（米）"""
        self.set_pose(0, 0, -distance, 0)


    def rotate_left(self, angle):
        """向左（逆时针）旋转指定角度（弧度）"""
        self.set_pose(0, 0, 0, angle)


    def rotate_right(self, angle):
        """向右（顺时针）旋转指定角度（弧度）"""
        self.set_pose(0, 0, 0, -angle)


    # ==============================
    # 状态查询
    # ==============================

    def get_position(self):
        """获取当前位置 (LOCAL_NED 坐标系)"""
        return (self.state.x, self.state.y, self.state.z)


    def get_attitude(self):
        """获取当前姿态 (单位：弧度)"""
        return (self.state.roll, self.state.pitch, self.state.yaw)


    def get_altitude(self):
        """获取相对高度 (单位：米)"""
        return self.state.relative_alt


    def get_battery_info(self):
        """获取电池信息"""
        return {
            'voltage': self.state.battery_voltage,
            'current': self.state.battery_current
        }