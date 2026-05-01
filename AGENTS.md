# AGENTS.md

本文件用于帮助后续参与本仓库的 Codex/Agent 快速理解项目背景、代码结构和协作约定。

## 项目背景

这是毕业设计《车载无人机的自主跟踪与降落方法研究》中的无人机端机载电脑代码。机载电脑为 Jetson Orin NX；无人机飞控为现成飞控，不能修改飞控端代码。整体目标是：无人车按照预设轨迹自主行驶，无人机自主跟踪无人车，并降落在无人车平台上。车机之间通过 WiFi + TCP 通信，共享位姿、轨迹等信息；机载电脑与飞控之间通过 MAVLink 串口通信。

当前仓库已逐步整理为以 `main.py` 为主入口的无人机端主链路工程，视觉检测、车机通信、参考轨迹跟踪、飞控控制和调试工具均围绕该结构继续开发；后续开发也将继续围绕现有工程结构推进；`test/` 目录主要保留独立功能验证、离线调参和辅助生成脚本。

## 阅读与编码要求

- 本项目 Python 文件均按 UTF-8 编码保存，并包含大量中文注释。读文件时优先使用 `rg`/`sed` 等能正确保留 UTF-8 的工具。
- 如果读取中文注释时出现乱码，先尝试用正确编码重新读取；若仍无法解决，必须暂停并告知用户，不能忽略注释继续改代码。
- 文件名或库名以 `kb_` 开头的，通常是用户本人编写或二次开发的代码，应优先关注。
- 保留现有中文注释和日志风格；新增注释应简洁，优先解释坐标系、协议、飞控安全相关逻辑。
- 本工作区已初始化为 Git 仓库，默认分支为 `main`。后续版本管理以 Git 提交、分支、标签和 GitHub 远程仓库为准。

## Git 版本管理约定

- 后续不再为常规开发创建 `backup/v<major.minor>_<summary>/` 本地源码快照；`backup/` 目录仅作为历史快照资料保留，默认不再更新。
- 每次开始改动前，先用 `git status` 确认工作区状态；如果已有未提交改动，必须判断这些改动是否属于当前任务，不能误覆盖或回退用户改动。
- 日常开发按“小步提交”管理：完成一个清晰功能点、调参点、修复点或文档更新后，先检查 `git diff`，再 `git add -A`，最后 `git commit -m "<type>: <summary>"`。
- 重大改动、跨文件重构、飞控控制逻辑修改或协议字段调整前，优先确认当前稳定状态已提交；必要时从 `main` 创建功能分支，例如 `git switch -c feature/landing-state-machine`。
- 已实测稳定或需要长期引用的阶段版本，优先使用 Git tag 标记，例如 `git tag v6.9_tracking-tuning`；阶段性版本差异仍可写入 `CHANGELOG.md`，但详细历史以 Git commit 为准。
- 当前 GitHub 远程仓库为 `origin = git@github.com:KrisBertram/graduation_uav_onboard.git`；`origin/main` 表示 GitHub 远程仓库上的 `main` 分支在本地的追踪引用。
- 当前工作区通过 SSH key 与 GitHub 通信；SSH 私钥只应保存在本机 `~/.ssh/` 下，不能提交到仓库、复制到文档或发送给他人。公钥可添加到 GitHub 的 SSH keys 页面。
- 推送到 GitHub 由 Codex 根据改动价值判断：重要文档约定、阶段性功能、实测前后状态、稳定修复和需要云端备份的提交应及时 `git push`；零碎试验或尚未验证的本地改动可先只保留在本地 commit 或工作区。
- 推送前再次检查 `git status`、`git log --oneline --decorate -5` 和当前分支跟踪关系，确认要上传的历史正确；已建立 upstream 后通常直接使用 `git push`。
- `AGENTS.md` 只记录当前项目结构、协作约定和安全边界，不记录逐版本变更历史。

## 重点目录与文件

- `main.py`
  - 当前视觉跟踪/飞控联调主入口。
  - 负责主循环编排、功能开关、DataLink 初始化和资源释放。
  - 运行方式为 `python main.py`。
  - 当前关键功能开关包括 `DEBUG_MODE_ENABLED`、`UDP_SENDER_ENABLED`、`DATALINK_ENABLED`、`VEHICLE_TCP_ENABLED`、`COLOR_MARKER_ENABLED`。
  - `DATALINK_ENABLED=False` 时不应假设飞控指令真的发送。

- `uav_core/`
  - 无人机端核心功能包，承载视觉跟踪、控制模式和调试工具等可复用逻辑。
  - `camera.py`：Jetson CSI 摄像头 GStreamer 管线和相机初始化。
  - `apriltag_pose.py`：相机内参、AprilTag 参数、目标 Tag 选择、PnP 位姿解算。
  - `visual_control.py`：相机坐标到机体系映射、Tag 前向估计、偏航对齐。
  - `color_marker_pose.py`：AprilTag 失效时的彩色标记检测与 PnP 备用位姿解算。
  - `reference_tracking.py`：目标状态估计、车端速度前馈融合、前视点预测、参考点限速平滑。
  - `frame_alignment.py`：无人车坐标系到无人机局部坐标系的旋转/平移在线对齐。
  - `vehicle_state.py`：无人车 TCP 上行状态包解析、最新状态缓存、可选后台接收器。
  - `control_modes.py`：键盘控制模式切换和飞控指令分发。
  - `debug_tools.py`：图像转储、视频转换、UDP 图传初始化和退出收尾。

- `kb_wifi_connect.py`
  - 无人机端 WiFi 连接和 TCP 服务器初版程序。
  - TCP 自定义包格式为 `A5 5A + length + cmd + data + crc16 + FF`。
  - `RecvPacket`/`SendPacket` 使用 `struct` 打包解析；修改字段时必须同步修改 dataclass 字段和 `FMT` 格式字符串，并检查 payload 长度。
  - `cmd=0x01` 为无人车到无人机上行状态，当前字段为 `speed, distance, yaw, pitch, roll, pos_x, pos_y, pos_z, action`，长度 34 字节。
  - 这里包含 `nmcli` WiFi 操作和阻塞式 TCP accept，运行会影响真实网络连接。

- `mavlink/`
  - `mavlink.py`、`mavcrc.py` 是 MAVLink 开源库/生成代码，默认不要改。
  - `kb_DataLink.py` 是本项目基于 MAVLink 的简化接口封装，负责串口初始化、心跳、飞控状态解析、解锁/加锁、起飞/降落、位置/姿态控制等。
  - 重要控制接口包括 `set_arm()`/`set_disarm()`、`set_takeoff()`/`set_land()`、`set_pose()` 等，其中 `set_pose()` 用于按机体系相对位移和相对偏航生成飞控目标。

- `utils/`
  - `kb_ImageDumper.py`：调试图像转储工具，支持覆盖、时间戳目录、递增目录、会话目录等存储模式。
  - `kb_Image2Video.py`：图片序列转视频工具，支持多种编码器和排序方式。
  - `kb_TagVisualizer.py`：AprilTag 可视化工具，包括像素比例尺、图像中心、tvec 向量、Z 距离条、重投影误差等。
  - `udp_video_sender.py`：无人机端 UDP 图传发送模块，使用 JPEG 压缩、分片、后台线程和小队列丢旧帧策略。

- `test/`
  - 存放早期模块测试/示例脚本，不是标准 pytest 测试套件。
  - `kb_datalink_test.py` 会连接飞控并提供交互式起飞/降落/移动菜单。
  - `kb_front_camera_test.py`、`kb_udp_video_sender_test.py` 会打开 Jetson CSI 摄像头，进行 AprilTag 检测，后者还会启动 UDP 图传。
  - `nested_apriltag_color_board_generator.py` 用于生成 A3 形状编码彩色备用标志。
  - `color_marker_hsv_tuner.py` 用于调试 `green/purple/yellow` 三个颜色类的 HSV 阈值；运行 CSI 模式会打开相机。
  - `record_clean_video.py` 用于录制无叠加标记的干净相机视频，默认输出到 `image_output/video/`；运行会打开摄像头。
  - `color_marker_hsv_tuner_windows.py` 是可脱离本工程运行的 Windows 单文件离线调参脚本，主要配合录制视频使用。

- `reference/graduation_ugv_firmware/`
  - 从 GitHub clone 的无人车端 TC387 单片机工程，仓库地址为 `https://github.com/KrisBertram/graduation_ugv_firmware.git`。
  - 作为无人机端开发的只读参考代码使用；除非用户明确要求跨仓库修改并 push，否则不要直接改这里。
  - 注意区分两个 `AGENTS.md`：本文件是无人机端工程说明，`reference/graduation_ugv_firmware/AGENTS.md` 是无人车端工程说明，阅读或执行约定时不要混用。
  - 重点参考 `AGENTS.md`、`docs/vehicle_drone_protocol.md`、`code/wifi_packet.c/.h`、`code/wifi.c/.h`、`code/define.h`、`user/cpu0_main.c`、`user/isr.c`。
  - 平时搜索无人机端主项目时默认排除该目录；需要对接车机通信协议时再专门进入此仓库阅读。

- `image_output/`
  - 调试图像和视频输出目录，通常不要阅读、改动或清理。

- `temp/`
  - 已过期或废弃文件，通常没有参考价值，默认不要阅读或改动。

## 硬件与安全边界

- 不要在未明确获得用户要求和安全确认时运行会影响真实硬件的脚本，尤其是包含解锁、起飞、降落、移动、串口飞控通信、WiFi 切换、摄像头独占访问的程序。
- 运行以下文件通常需要真实 Jetson/相机/飞控/网络环境：
  - `main.py`
  - `kb_wifi_connect.py`
  - `test/kb_datalink_test.py`
  - `test/kb_front_camera_test.py`
  - `test/kb_udp_video_sender_test.py`
  - `test/record_clean_video.py`
  - `test/color_marker_hsv_tuner.py --source csi`
- 对飞控控制逻辑做改动时，优先做静态检查、局部函数检查、仿真/假数据验证；不要直接假设硬件测试可用。
- 依赖可能包括 `opencv-python`/`cv2`、`numpy`、`pupil_apriltags`、`loguru`、`pyserial`、GStreamer、`nmcli` 等。当前仓库没有发现 `requirements.txt` 或 `pyproject.toml`。

## 坐标系约定（非常重要）

后续改视觉、控制、车机融合逻辑时，必须先确认下面坐标系，不要凭直觉临时改正负号。

- OpenCV 图像坐标系 / 相机坐标系：
  - 图像像素坐标：`u` 向右为正，`v` 向下为正。
  - OpenCV 相机坐标：`+X` 对应图像右方，`+Y` 对应图像下方，`+Z` 沿相机光轴指向被拍摄物体。
  - `solvePnP` 返回的 `tvec=(pnp_x, pnp_y, pnp_z)` 表示 Tag 原点相对于相机坐标系的位置，单位是米，不是像素偏差。

- AprilTag / `cv2.drawFrameAxes()` 坐标系：
  - `cv2.drawFrameAxes()` 默认颜色：红色为 `+X`，绿色为 `+Y`，蓝色为 `+Z`。
  - AprilTag 是有方向的；把纸面旋转 90°/180° 后，检测出的 Tag 坐标轴也会随之旋转。
  - AprilTag 坐标系的定义以当前 `get_object_points()` 和实际测试结果为准：当绿色 `+Y` 轴在纸面上朝下时，红色 `+X` 轴朝右，蓝色 `+Z` 轴指向纸内。
  - 当前推荐安装方式：Tag 的绿色 `+Y` 指向无人车车头，Tag 的红色 `+X` 指向无人车左侧，Tag 的蓝色 `+Z` 指向纸内（从空中指向地面）。
  - 当前代码用 `main.py` 中的 `TAG_FORWARD_AXIS = "+Y"` 表示“Tag 的 `+Y` 方向就是无人车车头方向”。如果实际安装方向不同，优先重新安装 Tag；确实无法重装时再修改 `TAG_FORWARD_AXIS`。

- 无人机机体系控制约定：
  - 该约定是本项目传给 `DataLink.set_pose(dx, dy, dz, dyaw)` 的控制接口约定，不要强行套用标准 FRD/FLU 名称。
  - `+X`：无人机机头前方，`dx > 0` 表示向前。
  - `+Y`：无人机机体右侧，`dy > 0` 表示向右。
  - `+Z`：向上，`dz > 0` 表示向上。
  - `dyaw > 0`：从上往下看顺时针旋转。
  - 当前视觉平移映射为：`body_dx = -pnp_y`，`body_dy = pnp_x`，`body_dz = -pnp_z`。

- 无人机局部坐标系：
  - 来源是 `DataLink.state.x/y/z/yaw`，由飞控通过 MAVLink 状态更新得到；`x/y/z` 当前按厘米转米保存，`yaw` 为弧度。
  - `DataLink.set_pose()` 内部把机体系水平偏移转成该局部坐标系偏移：
    - `global_dx = dx * cos(yaw) - dy * sin(yaw)`
    - `global_dy = dx * sin(yaw) + dy * cos(yaw)`
  - 该坐标系是无人机/飞控这一侧的局部坐标，不等同于无人车坐标系。

- 无人车坐标系：
  - 原点：无人车开机并完成 IMU 置零时的位置。
  - `+X`：无人车开机时车头方向。
  - `+Y`：无人车开机时车体左侧方向。
  - `yaw`：逆时针为正；车端内部为弧度，上行 TCP 包中为度。
  - `speed`：无人车车体前向速度，单位 m/s。
  - 车端全局速度计算公式为：`vx = speed * cos(yaw)`，`vy = speed * sin(yaw)`。

- 无人车到无人机局部系的在线对齐：
  - `FrameAligner` 估计二维变换：`p_drone = R(theta) * p_vehicle + t`。
  - 首次同时拿到有效车端状态、无人机状态和 AprilTag 视觉观测时，直接计算 `theta` 和 `t`；后续看到 Tag 时低通在线修正。
  - 当前默认 Tag 中心就是车辆/平台参考点，未配置 Tag 相对车体中心的安装偏置。
  - 如果 `TAG_FORWARD_AXIS` 或实际安装方向错误，车端 `speed + yaw` 前馈方向会错，参考轨迹效果会明显变差。

## 当前重要实现细节

- 下视相机标定参数目前直接写在 `uav_core/apriltag_pose.py` 和测试脚本中；`cameraMatrix[0,2]`、`cameraMatrix[1,2]` 被覆盖为 `960/2`、`540/2`。
- `uav_core/apriltag_pose.py` 使用 `tagCustom48h12`，并配置嵌套 Tag；以下数值都应该使用将图形打印出来之后在纸面上实际测量得到的物理尺寸：
  - ID 65：`0.200 m`
  - ID 66：`0.040 m`
  - ID 67：`0.008 m`
  - 跟踪优先级为 `[65, 66, 67]`
- `uav_core/color_marker_pose.py` 中彩色标记点尺寸也都应该使用将图形打印出来之后在纸面上实际测量得到的物理尺寸：中心距 `0.1185 m`，圆直径/方形边长 `0.032 m`。仅当 `COLOR_MARKER_ENABLED=True` 时，AprilTag 失败后会尝试备用的彩色 PnP 方案，为 `False` 时则只使用 AprilTag 进行 PnP。
- 彩色备用标志当前采用 3 色 × 2 形状编码：
  - `positive_x`：紫色圆形，`negative_x`：紫色正方形。
  - `positive_y`：绿色圆形，`negative_y`：绿色正方形。
  - `redundant_top_right`：纯黄色圆形，`redundant_bottom_left`：纯黄色正方形。
  - 检测逻辑会先按 `COLOR_CLASS_HSV_RANGES` 做颜色粗筛，再扣除 `FLOOR_HSV_RANGES` 中的地板色，随后用轮廓形状和 PnP 重投影误差筛选有效组合。
- `uav_core/visual_control.py` 中，相机坐标到机体系平移的映射为：
  - `body_dx = -pnp_y`
  - `body_dy = pnp_x`
  - `body_dz = -pnp_z`
- 参考轨迹跟踪默认由 `main.py` 中的 `VEHICLE_TCP_ENABLED=False` 关闭车端 TCP 接入；关闭时只使用视觉估计生成平滑参考轨迹。
- 车端 TCP 打开后，无人车 `speed + yaw` 会通过 `FrameAligner` 转到无人机局部坐标系，作为目标速度前馈；车端状态超过 `VEHICLE_STATE_TIMEOUT_S` 后自动退回视觉估速。
- 坐标系在线对齐默认假设 Tag 中心就是车辆/平台参考点，Tag 的 `+Y` 轴与无人车车头方向一致；若实物安装不同，应优先调整 `TAG_FORWARD_AXIS` 或后续增加安装偏置配置。
- 偏航控制使用 `solvePnP` 得到的 `rvec`，从 Tag 坐标轴候选法向中选择最接近机体正前方的方向，并带有滞后项 `_yaw_locked_edge_idx` 防抖。
- `uav_core/control_modes.py` 中的控制模式由键盘输入切换：
  - `0` 手动待机
  - `1` 起飞
  - `2` 降落
  - `3` 悬停待机
  - `4` AprilTag 跟踪

## 后续开发建议

- 优先在 `main.py`、`uav_core/` 和 `kb_` 文件中寻找用户正在开发的逻辑。
- 如果要把视觉追踪和 WiFi/TCP 通信合并，注意线程、阻塞调用、共享状态和数据包频率；`TCPServer.start()` 当前会阻塞等待连接。
- 无人车端协议参考仓库已放在 `reference/graduation_ugv_firmware/`；对接车机 TCP 数据包时，以该仓库的 `docs/vehicle_drone_protocol.md` 和 `code/wifi_packet.c/.h` 为准。
- 修改 MAVLink 发送逻辑前，先确认坐标系、单位、`type_mask`、飞控期望的 frame，以及当前飞控状态来源是否有效。
- 修改图传逻辑时，保持 UDP 包头格式和接收端一致；`FRAG_DATA_SIZE` 变更会影响链路丢包和延迟。
- 不要把 `image_output/` 中的图片/视频作为代码上下文，除非用户明确要求分析某次飞行记录。
- 不要基于 `temp/` 中内容推断当前实现，除非用户明确要求追溯历史方案。
- 不要把历史版本变更写进 `AGENTS.md`；日常变更以 Git commit 记录，阶段性版本差异可更新 `CHANGELOG.md`。

## 常用检查方式

- 查看无人机端主项目文件列表：`rg --files -g '!image_output/**' -g '!temp/**' -g '!reference/**' -g '!__pycache__/**' -g '!*.pyc'`
- 查看无人机端主项目结构：`rg -n "^(class|def|async def)|^if __name__" . -g '*.py' -g '!image_output/**' -g '!temp/**' -g '!reference/**'`
- 查看无人车端协议相关文件：`rg -n "serial_datapacket|RecvPacket|SendPacket|send_desc|recv_desc|0x01|0x02|CRC|WIFI_|TCP_" reference/graduation_ugv_firmware/code reference/graduation_ugv_firmware/user reference/graduation_ugv_firmware/docs`
- 语法检查可优先尝试：`python -m py_compile <file.py>`
- 查看 Git 状态：`git status --short`
- 查看未提交改动：`git diff`
- 查看最近提交：`git log --oneline --decorate -5`
- 查看远程仓库：`git remote -v`
- 推送当前分支：`git push`
- 若缺少硬件或依赖导致无法运行，应在最终回复中明确说明未进行真实硬件验证。
