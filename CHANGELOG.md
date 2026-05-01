# CHANGELOG

本文件用于记录本工程各版本快照对应的主要改动。具体协作约定和当前项目结构见 `AGENTS.md`。

## v6.8_green-yellow-separation

- 将形状编码彩色备用标志的颜色类从 `green/purple/lime` 调整为 `green/purple/yellow`，辅助标记改用纯黄色 `#FFFF00`。
- 将绿色标记从荧光绿调整为更深的 `#00B83F`，提高绿色与黄色之间的颜色可分性。
- 同步更新彩色 PnP 默认 HSV 阈值、HSV 调参工具颜色顺序和复合标志生成脚本。
- 重新生成 A3 复合标志 PNG/PDF/JSON，JSON 中辅助标记颜色类为 `yellow`。

## v6.7_shape-coded-color-markers

- 将彩色备用标志从 6 色圆点改为 3 色 × 2 形状编码标记，保留上下左右主标记和右上/左下辅助标记。
- 更新彩色标志生成脚本，输出 JSON 中新增 `color_class`、`shape`、`size_m` 等字段。
- 重构彩色 PnP 检测逻辑：增加地板色排除、多候选轮廓、圆/方形状分类、组合枚举和 PnP 重投影择优。
- HSV 调参工具改为调试 3 个颜色类，并打印 `COLOR_CLASS_HSV_RANGES` 片段。

## v6.6_color-pose-fallback

- 新增彩色标记点 PnP 备用视觉：AprilTag 失败且 `COLOR_MARKER_ENABLED=True` 时，尝试检测 6 色圆点并解算 `rvec/tvec`。
- 彩色 PnP 成功后复用现有参考轨迹、偏航控制和 `set_pose()` 控制链路；AprilTag 成功时仍保持最高优先级。
- 将嵌套 AprilTag 实际尺寸更新为当前打印后实测值：ID65 `0.194 m`，ID66 `0.0388 m`，ID67 `0.00776 m`。
- 新增彩色点调试可视化，显示质心、颜色名称、坐标轴、重投影误差和置信度；默认 HSV 阈值集中放在 `uav_core/color_marker_pose.py` 文件顶部，便于后续实测替换。

## v6.4_reference-tracking

- 新增预测式参考轨迹跟踪：视觉测量进入目标状态估计器，再生成前视目标点和平滑参考点。
- 新增无人车状态解析与缓存，`0x01` 上行包同步为 `speed, distance, yaw, pitch, roll, pos_x, pos_y, pos_z, action`。
- 新增无人车坐标系到无人机局部坐标系的在线对齐；首次有效观测直接对齐，后续低通修正。
- 新增车端 `speed + yaw` 速度前馈；车端状态超时后自动退回视觉估速。
- 跟踪模式继续使用现有 `DataLink.set_pose()`，不修改飞控端协议约定；第一版不做自动下降。

## v6.3_mavlink-import

- 修复 `python3 main.py` 启动时报错 `ModuleNotFoundError: No module named 'mavcrc'` 的问题。
- 在 `mavlink/__init__.py` 中为本地 MAVLink 生成代码补充同目录导入路径，保持 `mavlink/mavlink.py` 生成代码不变。

## v6.2_root-main

- 将视觉跟踪/飞控联调主入口整理为根目录 `main.py`，运行方式为 `python main.py`。
- 删除旧的 `kb_camera_datalink_debugging_v5.py` 兼容入口，避免入口冗余。
- 保留 `uav_core/` 作为核心功能包，主循环在根目录入口中编排，相机、AprilTag、视觉控制、模式分发和调试工具仍分模块维护。
- 将逐版本变更历史从 `AGENTS.md` 中移出，统一记录在本文件。
- 强化版本恢复脚本：以后由恢复脚本生成的 `emergency_*` 快照也会包含 `restore.py`，可直接一键恢复。

## v6.1_modular-entry

- 备份目录：`backup/v6.1_modular-entry/`
- 这是入口清理前的源码快照。
- 工程已拆分出 `uav_core/` 包，主循环仍位于 `uav_core/app.py`。
- `kb_camera_datalink_debugging_v5.py` 保留为兼容入口，内部调用 `uav_core.app.main()`。

## v6.0_tracking-baseline

- 备份目录：`backup/v6.0_tracking-baseline/`
- 这是大规模工程整理前的可跟踪基线快照。
- 视觉跟踪、AprilTag PnP、偏航对齐、飞控控制模式和 UDP 图传逻辑主要集中在 `kb_camera_datalink_debugging_v5.py`。
- 该版本保留了用户已实际测试可成功跟踪无人车的状态。
