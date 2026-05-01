# 会话交接总结

本文件用于在开启新的 Codex 会话时，快速交接当前工程进度、关键设计决策、已完成改动和后续开发注意事项。

建议新会话优先阅读顺序：

1. `AGENTS.md`
2. `CHANGELOG.md`
3. `SESSION_SUMMARY.md`
4. `main.py`
5. `uav_core/apriltag_pose.py`
6. `uav_core/color_marker_pose.py`
7. `uav_core/visual_control.py`
8. `uav_core/reference_tracking.py`
9. `uav_core/frame_alignment.py`
10. `uav_core/vehicle_state.py`

## 当前总体状态

- 项目是毕业设计《车载无人机的自主跟踪与降落方法研究》中的无人机端代码。
- 机载电脑为 Jetson Orin NX；飞控为现成飞控，飞控端协议暂时不能随意改。
- 当前工程已整理为以根目录 `main.py` 为主入口的无人机端主链路工程。
- 视觉检测、车机 TCP 状态接入、参考轨迹跟踪、飞控控制、UDP 图传和调试工具都围绕当前工程结构继续开发。
- `test/` 目录主要保留独立功能验证、离线调参、标志生成和辅助脚本。
- 当前工作区已初始化为 Git 仓库，默认分支为 `main`，并已连接 GitHub 远程仓库 `origin = git@github.com:KrisBertram/graduation_uav_onboard.git`。

## 硬件安全边界

- 默认不要运行会影响真实硬件的脚本，除非用户明确要求。
- `main.py`、`kb_wifi_connect.py`、`test/kb_datalink_test.py`、`test/kb_front_camera_test.py`、`test/kb_udp_video_sender_test.py`、`test/record_clean_video.py`、`test/color_marker_hsv_tuner.py --source csi` 都可能打开真实相机、飞控、WiFi、串口或网络。
- 代码改动后优先做静态检查，例如：

```bash
python3 -m py_compile main.py uav_core/*.py test/*.py
```

## 工程整理与版本历史

本轮长会话中，工程从一个很长的 `kb_camera_datalink_debugging_v5.py` 脚本逐步整理为当前结构：

- `main.py`：主链路入口。
- `uav_core/`：无人机端核心功能包。
- `mavlink/`：MAVLink 相关代码，其中 `mavlink.py`、`mavcrc.py` 默认不要改。
- `utils/`：图像转储、图片转视频、Tag 可视化、UDP 图传等工具。
- `test/`：调参、生成、离线验证和早期模块测试脚本。
- `CHANGELOG.md`：记录版本变更。
- `AGENTS.md`：记录当前工程约定、结构、安全边界和坐标系，不记录逐版本历史。

重要版本：

- `v6.0_tracking-baseline`：用户已实测可成功跟踪无人车的原始基线。
- `v6.2_root-main`：确定根目录 `main.py` 为主入口，删除旧入口冗余。
- `v6.3_mavlink-import`：修复 `mavcrc` 导入问题。
- `v6.4_reference-tracking`：新增目标状态估计、车端速度前馈、坐标系在线对齐和平滑参考轨迹。
- `v6.6_color-pose-fallback`：新增彩色标记 PnP 备用视觉接入。
- `v6.7_shape-coded-color-markers`：将 6 色圆点升级为 3 色 × 2 形状编码标记，并增强检测逻辑。
- `v6.8_green-yellow-separation`：绿色/纯黄色分离增强，当前彩色标志颜色类为 `green/purple/yellow`。

## Git / GitHub 版本管理

- 后续版本管理以 Git commit、branch、tag 和 GitHub 远程仓库为准；`backup/` 目录仅作为历史快照资料保留，默认不再更新。
- 当前 `main` 已推送到 GitHub，`origin/main` 表示 GitHub 远程仓库上的 `main` 分支在本地的追踪引用。
- 本工作区通过 SSH key 与 GitHub 通信；私钥只保存在本机 `~/.ssh/`，不要提交到仓库或写入文档。
- 日常开发建议小步提交：`git status` -> `git diff` -> `git add -A` -> `git commit -m "..."`。
- 是否 `git push` 由 Codex 根据改动价值判断：重要文档约定、阶段性功能、实测前后状态、稳定修复和需要云端备份的提交应及时推送；零碎试验可先留在本地。
- 重大改动、跨文件重构、飞控控制逻辑修改或协议字段调整前，优先确认当前稳定状态已提交；必要时从 `main` 创建功能分支。

本会话已完成并推送到 GitHub 的提交：

- `4143755 chore: establish local git baseline`
- `5b68fb7 chore: restore hardcoded wifi config`
- `08710d7 docs: switch version management to git`
- `b044d19 docs: document github sync workflow`

另外，本会话中用户确认 WiFi SSID/密码不作为敏感信息处理，`kb_wifi_connect.py` 已恢复为硬编码配置；`.env.example` 已删除。

## 当前主链路关键开关

当前 `main.py` 顶部关键开关：

- `DEBUG_MODE_ENABLED = False`
- `UDP_SENDER_ENABLED = True`
- `DATALINK_ENABLED = False`
- `VEHICLE_TCP_ENABLED = False`
- `COLOR_MARKER_ENABLED = True`
- `TAG_FORWARD_AXIS = "+Y"`

注意：

- `DATALINK_ENABLED=False` 时不会真的向飞控发控制指令。
- `VEHICLE_TCP_ENABLED=False` 时不会启动无人车 TCP 接入，也不会阻塞等待网络连接。
- `COLOR_MARKER_ENABLED=True` 时，AprilTag 失败后会尝试彩色标记 PnP 备用视觉。
- 用户现场测试时可能会临时修改这些开关，新会话不要假设它们一定保持当前值。

## 坐标系关键结论

这些结论非常重要，后续改视觉和控制前必须确认：

- OpenCV 图像坐标：`u` 向右，`v` 向下。
- OpenCV 相机坐标：`+X` 图像右方，`+Y` 图像下方，`+Z` 沿相机光轴指向被拍摄物体。
- `solvePnP` 返回的 `tvec=(pnp_x, pnp_y, pnp_z)` 表示标志原点相对于相机坐标系的位置，单位是米。
- `cv2.drawFrameAxes()`：红色 `+X`，绿色 `+Y`，蓝色 `+Z`。
- 当前嵌套 AprilTag/彩色板坐标：
  - 纸面向右为 Tag `+X`。
  - 纸面向下为 Tag `+Y`。
  - 当前推荐安装：Tag `+Y` 朝无人车车头，Tag `+X` 朝无人车左侧，Tag `+Z` 指向纸内。
- `main.py` 中 `TAG_FORWARD_AXIS = "+Y"` 表示 Tag `+Y` 是无人车车头方向。
- 无人机机体系控制约定：
  - `dx > 0`：机头前方。
  - `dy > 0`：机体右侧。
  - `dz > 0`：向上。
  - `dyaw > 0`：从上往下看顺时针旋转。
- 当前相机坐标到机体系平移映射在 `uav_core/visual_control.py`：
  - `body_dx = -pnp_y`
  - `body_dy = pnp_x`
  - `body_dz = -pnp_z`
- 无人车坐标系：
  - 开机位置为原点。
  - 开机车头方向为 `+X`。
  - 开机车体左侧为 `+Y`。
  - yaw 逆时针为正。
  - TCP 上行 yaw 单位为度，内部计算转弧度。

曾经踩过的关键坑：

- 用户实际测试发现 AprilTag 曾经安装成 `+Y` 指向车尾，导致车端速度前馈方向错误。
- AprilTag 是有方向的，不能把它当成四边完全对称的无方向图案。
- 如果实物安装方向错，参考轨迹和车端前馈效果会明显变差。

## 嵌套 AprilTag 及其检测

模块：`uav_core/apriltag_pose.py`

当前使用：

- `TAG_FAMILY = "tagCustom48h12"`
- 嵌套 Tag ID：`65 / 66 / 67`
- 跟踪优先级：`[65, 66, 67]`
- 当前新打印机输出与理想尺寸一致，因此尺寸已改回理想值：
  - ID 65：`0.200 m`
  - ID 66：`0.040 m`
  - ID 67：`0.008 m`

相机内参仍直接写在 `uav_core/apriltag_pose.py` 中，并且 `cameraMatrix[0,2]`、`cameraMatrix[1,2]` 被覆盖为 `960/2`、`540/2`。

## 参考轨迹跟踪

v6.4 已实现第一版“移动平台预测式参考轨迹跟踪”：

- `uav_core/reference_tracking.py`
  - `TargetEstimator`：视觉目标位置/速度估计。
  - `ReferenceTrajectory`：平滑参考点生成与速度限制。
  - 坐标转换辅助：`body_to_local_xy()`、`local_to_body_xy()`。
- `uav_core/frame_alignment.py`
  - `FrameAligner`：估计无人车坐标系到无人机局部坐标系的二维变换。
  - 形式：`p_drone = R(theta) * p_vehicle + t`。
  - 首次有效观测直接对齐，后续低通修正。实现无人车坐标系到无人机局部坐标系的在线对齐。
- `uav_core/vehicle_state.py`
  - 解析无人车 `0x01` 上行状态包。
  - payload 长度为 34 bytes。
  - 字段顺序：`speed, distance, yaw, pitch, roll, pos_x, pos_y, pos_z, action`。

当前默认参数在 `main.py`：

- `LOOKAHEAD_TIME_S = 0.5`
- `VEHICLE_STATE_TIMEOUT_S = 0.3`
- `TAG_LOST_PREDICT_TIME_S = 1.0`
- `MAX_REF_SPEED_MPS = 0.8`
- `MAX_CMD_OFFSET_M = 1.2`
- `ALIGN_YAW_ALPHA = 0.05`
- `ALIGN_POS_ALPHA = 0.05`
- `VISION_VEL_WEIGHT = 0.4`
- `VEHICLE_VEL_WEIGHT = 0.6`

第一版目标是提高移动平台跟踪稳定性，不做完整自动下降。

## 彩色备用标志

当前彩色备用视觉已经不是早期 6 色圆点方案，而是 v6.8 后的 3 色 × 2 形状编码方案。

生成脚本：

- `test/nested_apriltag_color_board_generator.py`

输出：

- `nested_apriltag_output/nested_apriltag_color_board_a3.png`
- `nested_apriltag_output/nested_apriltag_color_board_a3.pdf`
- `nested_apriltag_output/color_marker_layout.json`

当前 A3 横向理想尺寸：

- 页面：`420 mm × 297 mm`
- DPI：`1200`
- 外层 Tag：`200 mm`
- 彩色标记尺寸：`32 mm`
- 彩色标记中心距 Tag 中心：`118.5 mm`

当前 `uav_core/color_marker_pose.py` 中运行时物理尺寸：

- `OUTER_TAG_SIZE_M = 0.200`
- `COLOR_MARKER_CENTER_DISTANCE_M = 0.1185`
- `COLOR_MARKER_SIZE_M = 0.032`

当前 6 个形状编码标记：

- `positive_x`：紫色圆形，Tag `+X`，安装后车体左侧。
- `negative_x`：紫色正方形，Tag `-X`，安装后车体右侧。
- `positive_y`：绿色圆形，Tag `+Y`，安装后车头。
- `negative_y`：绿色正方形，Tag `-Y`，安装后车尾。
- `redundant_top_right`：纯黄色圆形，Tag `+X/-Y`。
- `redundant_bottom_left`：纯黄色正方形，Tag `-X/+Y`。

当前打印色：

- 深绿色：`#00B83F`
- 紫色：`#7A00FF`
- 纯黄色：`#FFFF00`

## 彩色标记检测

模块：`uav_core/color_marker_pose.py`

当前 HSV 阈值：

```python
COLOR_CLASS_HSV_RANGES = {
    "green": [((50, 55, 45), (86, 255, 255))],
    "purple": [((118, 40, 65), (150, 227, 224))],
    "yellow": [((22, 100, 80), (38, 255, 255))],
}
```

当前地板色排除阈值：

```python
FLOOR_HSV_RANGES = {
    "floor_1": [((164, 69, 110), (179, 149, 195)), ((0, 69, 110), (0, 149, 195))],
    "floor_2": [((6, 19, 138), (22, 90, 213))],
    "floor_3": [((99, 55, 102), (115, 144, 195))],
}
FLOOR_REJECTION_ENABLED = True
```

检测思路：

- 按 3 个颜色类做 HSV 粗筛。
- 从候选 mask 中扣除用户实测的三类地板色。
- 对每个颜色保留多个轮廓候选。
- 计算面积、bbox、质心、长宽比、extent、circularity、近似顶点数。
- 按圆形/正方形进行形状评分。
- 对 4~6 个唯一标记候选组合枚举。
- 使用 `solvePnP` / `solvePnPRansac`，按重投影误差、形状分数、布局分数选择最佳组合。
- 只有 `z > 0`、重投影误差达标、至少 4 个唯一标记时返回 `ColorMarkerPoseObservation`。

主链路策略：

- AprilTag 成功：优先使用 AprilTag 的 `rvec/tvec`。
- AprilTag 失败且 `COLOR_MARKER_ENABLED=True`：尝试彩色标记 PnP。
- 两者都失败：短时间使用参考轨迹预测，超过 `TAG_LOST_PREDICT_TIME_S` 后进入丢失保护/零控制。

## HSV 调参工具

脚本：

- `test/color_marker_hsv_tuner.py`

功能：

- 支持 CSI、USB camera、image、video。
- 支持视频循环播放。
- 支持鼠标 ROI 采样、滑条调 HSV。
- 当前调试颜色类为 `green / purple / yellow`。
- 输出 JSON：`test/color_marker_hsv_thresholds.json`。
- 同时打印可复制到 `COLOR_CLASS_HSV_RANGES` 的 Python 字典片段。

常用命令：

```bash
python3 test/color_marker_hsv_tuner.py --source video --path image_output/video/output.mp4
python3 test/color_marker_hsv_tuner.py --source image --path <image_path>
python3 test/color_marker_hsv_tuner.py --source csi
```

注意：CSI 模式会打开真实相机。

## 运动模糊问题与当前处理方向

用户通过飞行视频发现：

- 无人车移动时，摄像头视角中的 AprilTag 容易拖影。
- 拖影导致 AprilTag 检测器完全识别不到 Tag。
- Tag 检测不连续会让速度估计和轨迹预测变差。

讨论过的优先级：

- 根本治理：加光、缩短曝光、提高帧率、锁曝光、必要时换更合适的相机。
- 工程兜底：用车端状态和历史估计进行短时预测。
- 毕设场景定制：增加彩色形状编码标志，AprilTag 丢失时用彩色 PnP 备用。

当前已先实现第三条，即彩色形状编码备用视觉。

## 已知实现细节和约束

- 飞控输出继续使用 `DataLink.set_pose()`。
- 暂不修改 `mavlink/kb_DataLink.py` 中既有 `MAV_FRAME_GLOBAL`、`type_mask` 和飞控端协议约定。
- 用户说明过：完全不发送指令时飞机会自动悬停；控制量全为 0 也近似等价于悬停。
- `mavlink/mavlink.py` 是生成代码，默认不要改。
- `mavlink/__init__.py` 曾为解决 `ModuleNotFoundError: No module named 'mavcrc'` 做过兼容处理。
- `reference/graduation_ugv_firmware/` 是无人车端参考仓库，默认只读；需要对接协议时优先看其中的 `docs/vehicle_drone_protocol.md` 和 `code/wifi_packet.c/.h`。

## 近期刚完成的小改动

用户换了更好的打印机，新打印的 AprilTag + 彩色标记板与理想尺寸一致，因此刚刚把运行时物理尺寸改回理想值：

- `uav_core/apriltag_pose.py`
  - ID 65：`0.200 m`
  - ID 66：`0.040 m`
  - ID 67：`0.008 m`
- `uav_core/color_marker_pose.py`
  - `OUTER_TAG_SIZE_M = 0.200`
  - `COLOR_MARKER_CENTER_DISTANCE_M = 0.1185`
  - `COLOR_MARKER_SIZE_M = 0.032`
- `AGENTS.md` 已同步当前实现说明。

已验证：

```bash
python3 -m py_compile uav_core/apriltag_pose.py uav_core/color_marker_pose.py main.py
```

## 后续建议任务

可能的下一步开发方向：

- 用现场视频或图片验证当前 `green/purple/yellow` 阈值和地板排除逻辑，观察误检/漏检。
- 为 `uav_core/color_marker_pose.py` 增加离线合成图或真实图片回归测试，验证至少 4 点可 PnP、只有地板色时不返回有效 pose。
- 对彩色 PnP 的组合枚举做性能观察，必要时增加候选数量限制或几何先验。
- 现场打开 `COLOR_MARKER_ENABLED=True` 后，先不接飞控或保持 `DATALINK_ENABLED=False`，只看可视化和日志确认 `COLOR` 来源是否稳定。
- 后续若继续优化跟踪性能，可在不改变飞控协议的前提下调 `LOOKAHEAD_TIME_S`、`MAX_REF_SPEED_MPS`、`MAX_CMD_OFFSET_M`、视觉/车端速度权重。
- 真正进入降落阶段前，建议先把“稳定跟踪”和“下降/降落”拆成明确状态机，避免一次性把跟踪和降落逻辑耦合在一起。

## 给下一个会话的提醒

- 先读 `AGENTS.md`，尤其是坐标系约定和硬件安全边界。
- 版本管理约定已改为 Git/GitHub；不要再要求创建新的 `backup/` 源码快照。
- 不要把 `CHANGELOG.md` 中的历史尺寸误认为当前尺寸；当前尺寸以源码和 `AGENTS.md` 的“当前重要实现细节”为准。
- 不要把早期 6 色圆点方案误认为当前方案；当前是 `green/purple/yellow` 三色 × 圆/方两形状。
- 不要默认运行真实硬件脚本。
- 做跨文件、控制链路或协议改动前，先确认当前稳定状态已提交；必要时创建 Git 分支。
