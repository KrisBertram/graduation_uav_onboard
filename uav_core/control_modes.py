"""
键盘控制模式与飞控指令分发。
"""

import time

from loguru import logger


control_mode = 0  # 0=手动待机，1=起飞，2=降落，3=悬停待机，4=AprilTag 跟踪


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
        time.sleep(1)
        data_link.set_takeoff(altitude=0)
        time.sleep(4)
        data_link.set_pose(0, 0, 2, 0)  # 起飞后先发送零速度指令，保持当前位置
        time.sleep(2)
        control_mode = 3  # 切换到悬停模式
        logger.info("起飞完成，自动切换 → 悬停待机模式")

    elif control_mode == 2: # 降落模式，发送降落指令，但是只发送一次，降落成功后立刻变成手动模式
        logger.info("收到指令：执行降落")
        data_link.set_land()
        time.sleep(6)
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
            local_ignore_z = True  # 默认忽略高度控制，保持当前高度，除非非常近距离（Z < 2.0 m）时才直接控制高度
            local_direct_z = False # 默认不直接使用 dz 作为目标高度，而是作为相对高度偏移，除非非常近距离（Z < 2.0 m）时才直接控制高度

            # 当目标非常近时（Z < 2.0 m），直接控制高度，关闭 ignore_z，开启 direct_z
            if data_link.state.z < 1.8:
                local_ignore_z = False
                local_direct_z = True

            data_link.set_pose(cmd_dx, cmd_dy, 3.0, cmd_dyaw,
                               ignore_z=local_ignore_z,
                               direct_z=local_direct_z)  # 发送位姿控制命令
        elif tag_lost_duration < 2.0:
            # 丢失 Tag 不足 2 秒，进入宽限期，不发任何指令，飞控自行保持位置
            # 避免因短暂遮挡或检测抖动导致不必要的零速干预
            logger.warning(f"Tag 丢失中（已丢失 {tag_lost_duration:.2f}s / 2.00s），宽限期内不发指令...")
            pass
        else:
            # 没有检测到标签，连续丢失超过 2 秒，发送零速度指令，保持当前位置
            # 防止飞机因丢失追踪目标而乱飞，等待重新检测到 AprilTag
            data_link.set_pose(0, 0, 0, 0, ignore_z=True)
            logger.warning(f"Tag 丢失超过 2 秒（已丢失 {tag_lost_duration:.2f}s），已发送零位移指令，等待重新检测...")
