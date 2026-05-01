"""
DataLink 使用示例
===========================================
测试 DataLink 类各个接口的使用方法和典型场景
"""
import sys
import threading
import time
import math
from loguru import logger

sys.path.append('/home/zxt/zxt_ws')
sys.path.append('/home/zxt/zxt_ws/mavlink')
from mavlink.kb_DataLink import DataLink

# ============================================
# 示例 1: 基础初始化和通信
# ============================================
def example_basic_setup():
    """初始化 DataLink 并启动接收线程"""
    
    logger.info("示例 1: 基础初始化和通信，初始化 DataLink 并启动接收线程")

    # 创建 DataLink 实例
    link = DataLink(port='/dev/ttyTHS0', baudrate=115200)
    
    # 初始化 MAVLink
    link.init_mavlink()
    
    # 启动接收线程
    receive_thread = threading.Thread(target=link.receive_loop, daemon=True)
    receive_thread.start()

    # 心跳线程
    heartbeat_thread = threading.Thread(target=link.send_heartbeat, daemon=True)
    heartbeat_thread.start()

    return link


# ============================================
# 示例 2: 测试机体系控制函数
# ============================================
def example_set_position(link: DataLink):
    """测试机体系控制函数 set_pose(dx, dy, dz, dyaw)"""
    
    # 等待飞控连接
    while link.state.heartbeat_count < 5:
        logger.info("等待飞控连接...")
        time.sleep(1)

    logger.info("飞控已连接，开始飞行流程")

    while True:
        print("\n============ 控制菜单 ============")

        print("1 - 解锁电机、起飞")
        print("2 - 降落并退出")

        print("\n平移控制:")
        print("3 - dx =  0.3m")
        print("4 - dx = -0.3m")
        print("5 - dy =  0.3m")
        print("6 - dy = -0.3m")
        print("7 - dz =  0.3m")
        print("8 - dz = -0.3m")
        
        print("\n旋转控制:")
        print("q - dyaw =   0°")
        print("w - dyaw =  90°")
        print("e - dyaw = -90°")
        print("r - dyaw = 180°")
        print("t - dyaw =  45°")

        print("==================================")
        

        choice = input("请输入指令编号: ")

        if choice == "1":
            print("收到指令：解锁电机并起飞")
            link.set_arm()
            time.sleep(1)
            link.set_takeoff(altitude=0)
            time.sleep(2)

        elif choice == "2":
            print("收到指令：执行降落")
            link.set_land()
            time.sleep(6)
            link.set_disarm()
            print("已退出控制程序")


        elif choice == "3":
            print("执行 dx = 0.3m")
            link.set_pose(0.3, 0, 0, 0)

        elif choice == "4":
            print("执行 dx = -0.3m")
            link.set_pose(-0.3, 0, 0, 0)

        elif choice == "5":
            print("执行 dy = 0.3m")
            link.set_pose(0, 0.3, 0, 0)

        elif choice == "6":
            print("执行 dy = -0.3m")
            link.set_pose(0, -0.3, 0, 0)

        elif choice == "7":
            print("执行 dz = 0.3m")
            link.set_pose(0, 0, 0.3, 0)

        elif choice == "8":
            print("执行 dz = -0.3m")
            link.set_pose(0, 0, -0.3, 0)

        elif choice == "q":
            print("dyaw = 0°")
            link.set_pose(0, 0, 0, 0)

        elif choice == "w":
            print("dyaw = 90°")
            link.set_pose(0, 0, 0, math.radians(90))

        elif choice == "e":
            print("dyaw = -90°")
            link.set_pose(0, 0, 0, math.radians(-90))

        elif choice == "r":
            print("dyaw = 180°")
            link.set_pose(0, 0, 0, math.radians(180))

        elif choice == "t":
            print("dyaw = 45°")
            link.set_pose(0, 0, 0, math.radians(45))
        

        else:
            print("无效输入，请重新输入。")

# ============================================
# 测试飞行高度
# ============================================
def example_set_altitude(link: DataLink):
    """测试飞行高度控制函数 set_altitude(altitude)"""
    
    # 等待飞控连接
    while link.state.heartbeat_count < 5:
        logger.info("等待飞控连接...")
        time.sleep(1)

    logger.info("飞控已连接，开始飞行流程")

    while True:
        print("\n============ 控制菜单 ============")

        print("1 - 解锁电机、起飞")
        print("2 - 降落并退出")

        print("\n增量上升:")
        print("3 - dz =  0.6m")
        print("4 - dz = -0.6m")
        print("5 - dz =  1.0m")
        print("6 - dz = -1.0m")
        print("7 - dz =  2.0m")
        print("8 - dz = -2.0m")
        
        print("\n设置高度:")
        print("q - altitude = 0.6m")
        print("w - altitude = 1.2m")
        print("e - altitude = 1.8m")
        print("r - altitude = 2.4m")
        print("t - altitude = 3.0m")

        print("==================================")
        

        choice = input("请输入指令编号: ")

        if choice == "1":
            print("收到指令：解锁电机并起飞")
            link.set_arm()
            time.sleep(1)
            link.set_takeoff(altitude=0)
            time.sleep(2)

        elif choice == "2":
            print("收到指令：执行降落")
            link.set_land()
            time.sleep(6)
            link.set_disarm()
            print("已退出控制程序")


        elif choice == "3":
            print("执行 dz = 0.6m")
            link.set_pose(0, 0, 0.6, 0)

        elif choice == "4":
            print("执行 dz = -0.6m")
            link.set_pose(0, 0, -0.6, 0)

        elif choice == "5":
            print("执行 dz = 1.0m")
            link.set_pose(0, 0, 1.0, 0)

        elif choice == "6":
            print("执行 dz = -1.0m")
            link.set_pose(0, 0, -1.0, 0)

        elif choice == "7":
            print("执行 dz = 2.0m")
            link.set_pose(0, 0, 2.0, 0)

        elif choice == "8":
            print("执行 dz = -2.0m")
            link.set_pose(0, 0, -2.0, 0)

        elif choice == "q":
            print("altitude = 0.6m")
            link.set_attitude_altitude(0, 0, 0, altitude=0.6)

        elif choice == "w":
            print("altitude = 1.2m")
            link.set_attitude_altitude(0, 0, 0, altitude=1.2)

        elif choice == "e":
            print("altitude = 1.8m")
            link.set_attitude_altitude(0, 0, 0, altitude=1.8)

        elif choice == "r":
            print("altitude = 2.4m")
            link.set_attitude_altitude(0, 0, 0, altitude=2.4)

        elif choice == "t":
            print("altitude = 3.0m")
            link.set_attitude_altitude(0, 0, 0, altitude=3.0)
        

        else:
            print("无效输入，请重新输入。")

if __name__ == '__main__':

    link = example_basic_setup()
    
    try:
        example_set_position(link)
        # example_set_altitude(link)

    except KeyboardInterrupt:
        print("程序已退出")