"""
键盘控制模式与飞控指令分发。
"""

import time

from loguru import logger


control_mode = 0  # 0=手动待机，1=起飞，2=降落，3=悬停待机，4=AprilTag 跟踪

CONTROL_MODE_NAMES = {
    0: "manual",
    1: "takeoff",
    2: "land",
    3: "hold",
    4: "tracking",
}

# ---------- 飞控模式调参 ----------
# 这些参数会直接影响真实飞控动作，实测前应逐项确认；不要在不了解飞控响应时一次性大幅调整。

# 解锁后等待时间：给飞控/电机留出完成解锁状态切换的时间。
# 调大：更稳妥但起飞流程更慢；调小：流程更快但可能在飞控未准备好时发送起飞。
# 推荐范围：0.5~2.0 s；初始推荐 1.0 s。
ARM_WAIT_S = 1.0

# 起飞指令后等待时间：等待飞控完成起飞初始动作，再发送后续高度/位置目标。
# 调大：更保守；调小：可能过早覆盖起飞目标。
# 推荐范围：3~6 s；初始推荐 4 s。
TAKEOFF_WAIT_S = 4.0

# 起飞后发送的相对高度目标：用于让无人机爬升到安全跟踪高度附近。
# 调大：起飞后高度更高、视野更大；调小：更接近平台但安全余量更小。
# 推荐范围：1.5~3.0 m；初始推荐 2.0 m。
TAKEOFF_HOLD_Z_M = 2.0

# 起飞后高度目标保持时间：发送起飞后 set_pose 目标后等待多久再切到悬停。
# 调大：给飞控更多时间爬升；调小：模式切换更快但可能还没到目标高度。
# 推荐范围：1~3 s；初始推荐 2 s。
TAKEOFF_HOLD_WAIT_S = 2.0

# 降落指令后等待时间：等待飞控执行降落，再发送加锁。
# 调大：避免过早加锁；调小：降落流程更快但风险更高。
# 推荐范围：5~10 s；初始推荐 6 s。
LAND_WAIT_S = 6.0

# 跟踪模式高度目标：control_mode=4 且参考目标有效时传给 set_pose() 的 z 参数。
# 当前大多数情况下 ignore_z=True，因此主要在低高度 direct_z 触发后生效。
# 推荐范围：2.0~4.0 m；初始推荐 3.0 m。
TRACK_TARGET_Z_M = 3.0

# 直接高度控制触发阈值：当前高度低于该值时，关闭 ignore_z 并启用 direct_z。
# 调大：更早接管高度控制；调小：更多时候保持飞控当前高度。
# 推荐范围：1.2~2.5 m；初始推荐 1.8 m。
TRACK_DIRECT_Z_ENABLE_BELOW_M = 1.8

# 跟踪目标丢失保护宽限期：目标无效但未超过该时间时不发新指令，让飞控自行保持。
# 调大：容忍更长视觉/融合短暂丢失；过大则长时间不修正位置。
# 调小：更快进入零位移保护；过小可能因瞬时漏检频繁保护。
# 推荐范围：1~3 s；初始推荐 2 s。
TRACK_LOST_GRACE_S = 2.0


def get_control_mode_snapshot():
    """返回当前控制模式快照，供主循环和飞行日志低频采样使用。"""
    return {
        "id": control_mode,
        "name": CONTROL_MODE_NAMES.get(control_mode, "unknown"),
    }


def keyboard_listener():
    global control_mode

    while True:
        key = input()

        if key == "0":
            control_mode = 0
            logger.info("切换模式 → 手动模式")

        elif key == "1":
            control_mode = 1
            logger.info("切换模式 → 起飞模式")

        elif key == "2":
            control_mode = 2
            logger.info("切换模式 → 降落模式")

        elif key == "3":
            control_mode = 3
            logger.info("切换模式 → 悬停模式")

        elif key == "4":
            control_mode = 4
            logger.info("切换模式 → AprilTag 跟踪模式")


def handle_control_mode(data_link, control_target_valid, cmd_dx=0, cmd_dy=0, cmd_dz=0, cmd_dyaw=0, tag_lost_duration=0.0):
    """
    根据当前控制模式和参考轨迹有效状态，向飞控发送对应指令。
    注意: control_mode 为全局变量，部分模式会在执行后自动切换。

    模式说明：
        0 - 手动待机：不发任何指令，等待用户输入，可随时从模式 4 跳入此模式作为紧急停止
        1 - 起飞：执行一次起飞序列（解锁 → 起飞 → 爬升），完成后自动切换到模式 3
        2 - 降落：执行一次降落序列（降落 → 加锁），完成后自动切换到模式 0
        3 - 悬停待机：不发任何指令，飞控自行保持位置，可随时从模式 4 跳入此模式作为紧急停止
        4 - AprilTag 追踪：参考目标有效则发追踪指令；参考目标无效则进入丢失保护
    """
    global control_mode

    if control_mode == 0:   # 手动模式，等于空闲模式，什么都不做，等待用户指令
        tag_lost_duration = 0.0  # 在手动模式下，重置 Tag 丢失持续时间，避免切换到跟踪模式后误判为丢失状态
        pass

    elif control_mode == 1: # 起飞模式，发送起飞指令，但是只发送一次，起飞成功后立刻变成悬停模式
        logger.info("收到指令：解锁电机并起飞")
        data_link.set_arm()
        time.sleep(ARM_WAIT_S)
        data_link.set_takeoff(altitude=0)
        time.sleep(TAKEOFF_WAIT_S)
        data_link.set_pose(0, 0, TAKEOFF_HOLD_Z_M, 0)  # 起飞后先发送高度目标，保持当前位置附近
        time.sleep(TAKEOFF_HOLD_WAIT_S)
        control_mode = 3  # 切换到悬停模式
        logger.info("起飞完成，自动切换 → 悬停待机模式")

    elif control_mode == 2: # 降落模式，发送降落指令，但是只发送一次，降落成功后立刻变成手动模式
        logger.info("收到指令：执行降落")
        data_link.set_land()
        time.sleep(LAND_WAIT_S)
        data_link.set_disarm()
        logger.info("已退出控制程序")
        control_mode = 0  # 切换到手动模式
        logger.info("降落完成，自动切换 → 手动待机模式")

    elif control_mode == 3: # 悬停模式，发送零速度指令，保持当前位置
        tag_lost_duration = 0.0  # 在手动模式下，重置 Tag 丢失持续时间，避免切换到跟踪模式后误判为丢失状态
        # logger.info(f"z = {data_link.state.z:.6f} m\t| z_cm = {data_link.state.z_cm:.6f} cm")
        # logger.info(f"relative_alt = {data_link.state.relative_alt:.6f} m\t| relative_alt_mm = {data_link.state.relative_alt_mm:.6f} mm")
        pass  # 什么也不发，保持在空中即可

    elif control_mode == 4: # AprilTag 跟踪模式
        if control_target_valid == 1:
            local_ignore_z = True  # 默认忽略高度控制，保持当前高度，除非低于 TRACK_DIRECT_Z_ENABLE_BELOW_M
            local_direct_z = False # 默认不直接使用 dz 作为目标高度，低于阈值后才启用 direct_z

            # 当当前高度低于阈值时，直接控制高度，关闭 ignore_z，开启 direct_z
            if data_link.state.z < TRACK_DIRECT_Z_ENABLE_BELOW_M:
                local_ignore_z = False
                local_direct_z = True

            data_link.set_pose(cmd_dx, cmd_dy, TRACK_TARGET_Z_M, cmd_dyaw,
                               ignore_z=local_ignore_z,
                               direct_z=local_direct_z)  # 发送位姿控制命令
        elif tag_lost_duration < TRACK_LOST_GRACE_S:
            # 丢失目标仍在宽限期内，不发任何指令，飞控自行保持位置
            # 避免因短暂遮挡或检测抖动导致不必要的零速干预
            logger.warning(f"Tag 丢失中（已丢失 {tag_lost_duration:.2f}s / {TRACK_LOST_GRACE_S:.2f}s），宽限期内不发指令...")
            pass
        else:
            # 没有检测到标签，连续丢失超过宽限期，发送零位移指令，保持当前位置
            # 防止飞机因丢失追踪目标而乱飞，等待重新检测到 AprilTag
            data_link.set_pose(0, 0, 0, 0, ignore_z=True)
            logger.warning(f"Tag 丢失超过 {TRACK_LOST_GRACE_S:.2f}s（已丢失 {tag_lost_duration:.2f}s），已发送零位移指令，等待重新检测...")
