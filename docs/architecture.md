# Architecture

## Runtime Chain

```mermaid
flowchart LR
    A["CaptureService"] --> B["DetectService"]
    B --> C["TemporalTracker"]
    C --> D["MatchStateMachine"]
    D --> E["PlanningPositionStabilizer"]
    E --> F["GeometryPhysicsPlanner"]
    F --> J["PocketEntryCorridor"]
    F --> K["TargetShotPlanner (explicit rebounds)"]
    F --> H["RouteTopologyContinuity"]
    H --> I["OverlayBuilder"]
    A --> G["ReplayRecorder"]
    B --> G
    C --> G
    D --> G
    F --> G
    H --> G
    I --> G
```

## Design Notes

- 每个模块输出结构化对象，方便 JSONL 回放。
- 采集层和推理层都是可替换边界。
- 标定层支持旧 OpenCV YAML 和旧投影标定 JSON。
- 状态机按时间窗口推进，不直接信任单帧检测结果。
- 路线稳定只修改规划用球心；原始跟踪和状态机坐标保持不变。
- 每帧重新计算完整物理路线，拓扑连续模块只能选择本帧有效候选，不保存旧坐标。
- 阻挡球碰撞净空始终使用本帧原始测量位置，避免滤波延迟降低安全性。
- `PocketEntryCorridor` 在有限袋口内搜索完整球可通过的球心入口；拟合袋心只保留为袋号和洞心语义，不再强迫物体球线穿过单一点。
- 非撞库目标球路线必须是从目标球到袋口走廊的单一直线；需要撞库的路线只能由目标击球模块给出包含真实碰库点的显式折线，禁止在袋口人工插入无碰撞折点。
- 进球通知位于检测事件 seam：状态机发出 `POCKET_DETECTED` 时立即提示，后续同一决定的 `POCKET_CONFIRMED / POT_PROBABLE` 由桌面GUI与Web共享的 `PocketNoticeTracker` 去重。
- 学习系统不在本程序内实现，不做训练数据采集。
