# 0623 精简 PRD 功能排查

对照文档：`example/0623_重构_精简.md`。后续排查和开发只参考该精简文档，不再参考 `example/0623_重构.md`。

状态说明：

- 已实现：代码中已有可运行实现。
- 本次补齐：本次补上了可运行交互或控制面。
- 部分实现：有底座，但尚未达到 PRD 的完整逻辑。
- 未实现：当前仓库没有对应生产逻辑。

## 标定系统

| PRD 要求 | 当前状态 | 说明 |
|---|---|---|
| 相机内参 ChArUco 精标 | 部分实现 | `bas.calibration.charuco` 可生成/检测 ChArUco；`CameraCalibration` 可读取 OpenCV YAML。缺少完整多姿态采集、内参求解和保存 UI。 |
| 投影器-相机联合标定 | 部分实现 | `ProjectionCalibration` 支持加载/保存投影校正 JSON、homography、对应点和 residual field。缺少结构光采集、projector-as-camera 建模和 bundle adjustment 求解器。 |
| 台面局部精修层 | 部分实现 | 已支持 `residual_cam_points/residual_proj_offsets`，运行时会叠加局部残差。当前插值为邻域加权，不是 PRD 建议的薄板样条或 B-spline。 |
| 校正码/编码图案投影 | 本次补齐 | UI 向导可显示 ChArUco 校正码；若 OpenCV aruco 不可用，会 fallback 到编码网格/定位十字。 |
| 校正结果可视化 | 本次补齐 | UI 可显示桌面多边形、对应点编号、误差统计和局部残差箭头。 |
| 校准步骤文档 | 本次补齐 | README 增加了主标定、局部精修、holdout 验收和结果判读步骤。 |
| 图像重投影误差 mean/P95 | 部分实现 | `ProjectionCalibration.calibration_error_stats()` 统计 mean/median/P95/max 像素误差。 |
| 台面毫米误差 median/P95 | 未实现 | 当前没有 holdout 世界点数据结构和自动验证脚本。 |
| 分区误差 | 未实现 | 尚未按近端、中段、远端、六袋口区域汇总 P95。 |
| 误差随距离梯度 | 未实现 | 尚未拟合 `e_mm = alpha + beta d`。 |
| 温漂/振动验证 | 未实现 | README 已写验收要求，代码未自动调度 0/15/30 分钟验证。 |

## 感知与跟踪

| PRD 要求 | 当前状态 | 说明 |
|---|---|---|
| YOLO 球/球杆检测入口 | 已实现 | `UltralyticsDetector` 支持模型、类别、tile 推理和 NMS；另有 `debug_color` 用于 synthetic/smoke。 |
| 检测框到底面圆心几何修正 | 部分实现 | `Detection.center` 使用 bbox 中心，尚未按球高度、相机姿态和底面接触点做完整修正。 |
| 高分框优先、低分框补轨迹 | 部分实现 | tracker 使用 `low_conf` 保留低分检测，新轨迹只由 `high_conf` 创建；尚未明确拆成 ByteTrack 式两轮 Hungarian。 |
| 遮挡短时保留 | 已实现 | `max_lost_frames` 内保留轨迹，visibility 标为 `occluded`。 |
| 短窗速度估计 | 部分实现 | 使用指数平滑即时速度；尚未做 OC-SORT 式观测中心修正和多帧一致性滤波。 |
| 连续置信度 `q_t` | 部分实现 | `TrackObservation.quality` 随检测置信和 lost frame 衰减；尚未融合检测、运动、类别、几何四项置信。 |
| 类别投票 | 已实现 | tracker 维护 `vote_window`，输出稳定类别。 |

## 状态机与事件检测

| PRD 要求 | 当前状态 | 说明 |
|---|---|---|
| 六个宏状态 | 已实现 | `STABLE_IDLE/PRE_SHOT_ARMED/SHOT_ACTIVE/SETTLING/TURN_RESOLVE/ANOMALY_RECOVERY` 已在 schema 和状态机中实现。 |
| 双阈值运动/停止去抖 | 部分实现 | 配置有 `moving_speed_px_s/still_speed_px_s` 和帧计数；当前单位是像素/秒，不是台面 mm/s。 |
| 击球开始需满足三项中至少两项 | 部分实现 | 当前以球杆可见、cue 运动、全局 moving 触发；尚未实现 cue tip 进入邻域 + 速度跃迁 + 加速度峰值的 2/3 投票。 |
| 球-球碰撞事件 | 未实现 | 当前状态机未输出 ball collision 事件。 |
| 库边碰撞事件 | 未实现 | 当前状态机未输出 rail collision 事件。 |
| 口袋区域 + 轨迹趋势 + 消失确认 | 部分实现 | 当前有 `BALL_DISAPPEARED`，但未结合 pocket funnel、throat 方向和替代轨迹验证。 |
| 异常恢复宏状态 | 部分实现 | 已检测多 cue、球数 >16、中心重叠等异常并进入 `ANOMALY_RECOVERY`；尚未做 Hungarian 回滚重配和 `unknown_track` 延迟决策。 |
| 人工介入控制模块 | 本次补齐 | 状态机新增 `force_phase/set_operator_hold/snapshot_layout/clear_review_flags`；UI 新增强制状态、冻结、确认稳定、重置、清除复核。 |
| 人工事件进入回放 | 本次补齐 | 操作员动作会写入 `OPERATOR_*` 事件，并随下一帧状态进入 replay JSONL。 |

## 规划与投影

| PRD 要求 | 当前状态 | 说明 |
|---|---|---|
| 几何候选路线 | 已实现 | `GeometryPhysicsPlanner` 生成目标球、袋口、ghost ball、aim line 和 object line。 |
| 快速物理筛选 | 部分实现 | 有切角、路径 clearance、距离和风险评分；摩擦、旋转、能量损失仍是简化启发式。 |
| 学习排序/真实结果闭环 | 未实现 | `DisabledLearningRanker` 是显式占位，README 已说明当前版本不采集学习数据。 |
| 投影 overlay | 已实现 | `OverlayBuilder` 把路线、ghost ball、目标球、袋口投影到 projector px。 |
| 颗星公式 | 已实现 | 设置页支持位置、缩放、旋转和标签偏移。 |
| 投影校正模式不被实时 overlay 覆盖 | 本次补齐 | UI 进入校正模式后会保持校正码/残差图，点“恢复实时投影”才回到实时路线。 |

## 回放、接口与部署

| PRD 要求 | 当前状态 | 说明 |
|---|---|---|
| 模块边界 `Capture -> Detect -> Track -> State -> Plan -> Projection -> Replay` | 已实现 | `RuntimePipeline` 按该链路组织。 |
| 结构化数据对象 | 已实现 | `FramePacket/DetectionsFrame/TracksFrame/MatchStateFrame/ShotPlan/ProjectionOverlay` 已定义。 |
| JSONL 回放 | 已实现 | `ReplayRecorder` 写 frame/detections/tracks/state/plan/overlay。 |
| Parquet/SQLite | 未实现 | 当前仅 JSONL，可后续扩展。 |
| TensorRT/DeepStream/GStreamer | 未实现 | 当前在线栈是 Python + OpenCV + optional Ultralytics。 |
| UI 模块交互 | 本次补齐 | 主界面新增模块状态表，展示采集、检测、跟踪、状态机、规划、投影、回放、标定状态。 |

## 优先级建议

1. 先补投影校准验证脚本：读取 holdout world/camera/projector 点，输出像素误差、毫米误差、分区 P95、距离梯度。
2. 再补状态机事件层：cue tip 2/3 击球投票、ball collision、rail collision、pocket funnel 判定。
3. 然后把 tracker 关联拆成显式 high-pass + low-pass 两阶段 Hungarian，质量分改为 `q_det/q_motion/q_cls/q_geom` 融合。
4. 最后补联合标定求解器和 B-spline/TPS 局部残差拟合，替换当前邻域加权 residual field。
