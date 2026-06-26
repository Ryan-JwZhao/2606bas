# 投影校准双路线方案

## 1. 背景

当前现场现象已经比较明确：

- 调试模式的球圈不能稳定压中球心。
- 离投影仪较远区域的球圈明显比真实球大一圈。
- 相机去畸变已经可靠完成。
- 投影机内置梯形、缩放、自动几何增强已经关闭。

这说明问题不再是简单的 `X/Y` 平移，也不建议继续只靠人工拖 `X/Y`、`scale` 或四点透视去逼近。

核心判断是：

1. 台布平面的投影映射，属于“平面几何校准”问题。
2. 球圈中心和球圈半径，额外受到球体高度的影响，属于“离平面目标补偿”问题。

这两层必须拆开处理。只要把它们混成一个变换，无论四点透视还是更高阶平面网格，最后都会出现“角点对了，但球圈还是不准”。

## 2. 总体目标

保留当前方案，同时新增一套工程化成熟方案，并允许用户在启动后或设置中二选一，且选择结果写入 `local_settings/user_settings.json` 自动记忆。

建议名称：

- `传统模式`
- `工程模式`

建议模式含义：

- `传统模式`：保留现有 `projection_calibration.json + residual field + 球心拉回补偿` 的工作流，不破坏现在能用的路径。
- `工程模式`：新增“平面密集校准 + 球体中心/半径补偿”的双层模型，作为后续默认推荐方案。

## 3. 为什么四点透视不是最佳答案

你现在已经观察到“4 个角位置和桌面对齐，但局部还是不完美”。这恰好说明：

- 四点透视最多只能解决一个全局平面单应问题。
- 如果误差来自投影镜头边缘畸变、局部非线性、布面轻微不平，四点透视不够。
- 如果误差来自球心不在台布平面上，四点透视永远不可能把所有球都调到最优。

因此，最成熟路线不是“更用力地调四点”，而是：

1. 先把台布平面校到准。
2. 再单独校球体相关的中心偏差和半径偏差。

## 4. 双路线设计

### 4.1 传统模式

沿用当前项目已有链路：

- `camera_px -> projector_px` 使用 `homography + residual field`
- `table_mm <-> projector_px` 使用 `table_polygon_proj`
- 球心使用现有 `ball_center_compensation`
- 投影调试球圈继续沿用当前实现

优点：

- 成本低，和当前代码兼容最好。
- 现场可以继续马上使用。

缺点：

- 球圈半径仍然容易受球体高度和平面映射混用影响。
- 远端区域“圈偏大、圈偏中心”问题较难根治。

### 4.2 工程模式

工程模式拆成两层：

1. 平面层：只负责台布上的几何映射。
2. 球体层：只负责球心和球圈半径。

#### 平面层目标

得到稳定的 `camera_px -> table_mm -> projector_px` 高精度平面映射，不依赖人工盯画面微调。

建议方法：

- 使用结构光或编码图案做密集采样。
- 最低可用版本：多组 ChArUco / 编码网格联合采样。
- 正式推荐版本：Gray code 或 phase-shift + 辅助角点定位。
- 求解结果使用“全局单应 + 局部网格/TPS/B-spline 残差”。

平面层只验证这些对象：

- 台边
- 袋口
- 台布上的参考点
- 台布上的辅助线

#### 球体层目标

不要再把“相机里看到的球轮廓半径”直接拿去通过平面映射投到投影上。

工程模式下，球相关投影改为：

1. 先估计球心在台面坐标中的落点。
2. 再用已知真实球直径 `57.15 mm` 计算该位置应投出的物理半径。
3. 球圈半径由 `table_mm -> projector_px` 的局部尺度决定，而不是由图像 `radius_px` 决定。

这样做的直接收益是：

- 远端球圈不会因为球高和透视叠加而越来越大。
- 全桌任何位置的球圈半径都回到“真实物理球径”定义。

球体层还需要一个“球心落点补偿模型”：

- 输入：检测到的球心 `camera_px`
- 输出：该球中心在台布平面的校正落点 `table_mm`

推荐模型：

- 第一阶段：规则网格残差场
- 第二阶段：TPS 或 B-spline 位移场

## 5. 工程模式的成熟校准流程

### 5.1 阶段 A：平面密集校准

目标：生成 `plane_projection_calibration.json`

步骤：

1. 投影结构光/编码图案到台布平面。
2. 相机采集多帧。
3. 自动恢复大量 `camera_px <-> projector_px` 对应点。
4. 拟合全局 `homography`。
5. 再拟合局部残差网格或 TPS。
6. 生成独立 holdout 报告。

建议采样覆盖：

- 四角
- 四边中点
- 六个袋口附近
- 台面中心
- 投影仪远端一整条长边

验收仍然保留现有 holdout 体系，但工程模式下建议重点看：

- `table_error_mm median`
- `table_error_mm p95`
- `zone_p95_mm`
- `distance_slope_mm_per_cm`

### 5.2 阶段 B：球心补偿校准

目标：生成 `ball_center_compensation_engineered.json`

步骤：

1. 把单颗真实球依次放到若干标准位置。
2. 每个位置记录：
   - 相机检测中心 `camera_px`
   - 该球真实落点 `table_mm`
3. 用这些样本拟合球心位移场。

推荐采样点数：

- MVP：15 到 20 点
- 正式版本：25 到 40 点

推荐覆盖：

- 近端、中部、远端
- 左中右三列
- 六个袋口附近

建议输出：

- `delta_table_mm_x`
- `delta_table_mm_y`
- 质量统计

### 5.3 阶段 C：球圈半径模型

目标：不再使用“图像半径直接投影”的旧逻辑。

工程模式下的球圈半径计算建议固定为：

1. 用球心补偿后的 `table_mm` 位置作为中心。
2. 半径固定取 `ball_diameter_mm / 2`。
3. 通过局部 `table_mm -> projector_px` 变换换算成投影半径。

如果后续仍要显示“相机观测轮廓圈”，那也应作为单独的“观测圈”，不能和“物理球圈”混用。

建议在调试模式中分成两个可选层：

- `物理球圈`
- `观测球圈`

默认只显示 `物理球圈`。

## 6. 模式切换与记忆

现有工程已经具备设置持久化基础，因此建议新增以下配置项。

### 6.1 AppConfig 建议新增字段

放在 `CalibrationConfig`：

```python
projection_mode: str = "legacy"  # legacy | engineered
legacy_projection_file: Optional[str] = None
engineered_plane_projection_file: Optional[str] = None
engineered_ball_compensation_file: Optional[str] = None
```

### 6.2 UserSettings 建议新增字段

写入 `local_settings/user_settings.json`：

```json
{
  "projection_mode": "engineered",
  "legacy_projection_calibration_file": "local_settings/calibrations/projection_legacy.json",
  "engineered_plane_projection_file": "local_settings/calibrations/projection_plane_engineered.json",
  "engineered_ball_compensation_file": "local_settings/calibrations/ball_comp_engineered.json"
}
```

### 6.3 启动加载规则

建议规则：

1. 程序启动时先读 `projection_mode`。
2. `legacy` 模式加载传统投影标定文件。
3. `engineered` 模式加载平面标定文件和球体补偿文件。
4. 模式切换后立即保存到 `user_settings.json`。

这样就能满足“每次开启二选一，且设置需记忆”。

## 7. UI 建议

### 7.1 设置页

在“投影修正”或“标定”区域增加：

- 单选框：`传统模式`
- 单选框：`工程模式`
- 路径输入：传统标定文件
- 路径输入：工程模式平面标定文件
- 路径输入：工程模式球体补偿文件

### 7.2 投影校准对话框

保留当前入口，但在内部拆成两个向导：

- `传统校准`
- `工程校准`

工程校准下再分两步：

1. `平面自动校准`
2. `球体补偿校准`

### 7.3 调试模式

增加调试显示选项：

- `显示物理球圈`
- `显示观测球圈`
- `显示球心补偿箭头`
- `显示平面残差热区`

## 8. 数据结构建议

### 8.1 工程模式平面文件

示例：

```json
{
  "mode": "engineered_plane_v1",
  "projector_size": [1280, 800],
  "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "residual_model": {
    "type": "tps",
    "control_cam_points": [],
    "offsets_proj": []
  },
  "table_polygon_cam": [],
  "table_polygon_proj": [],
  "quality_report": {}
}
```

### 8.2 工程模式球体补偿文件

示例：

```json
{
  "mode": "engineered_ball_comp_v1",
  "ball_diameter_mm": 57.15,
  "center_bias_model": {
    "type": "grid",
    "control_camera_points": [],
    "delta_table_mm": []
  },
  "quality_report": {}
}
```

## 9. 与当前代码的衔接方式

当前项目已经有几个很适合复用的基础：

- `ProjectionCalibration`：可以继续作为平面层主体。
- `ResidualField`：可先作为工程模式 MVP 的局部残差实现。
- `ball_center_compensation`：可作为工程模式球体层的过渡入口。
- `verify_holdout_samples()`：可继续作为验收框架。

建议不要推倒重来，而是按下面方式演进：

1. 保留 `ProjectionCalibration` 作为平面层。
2. 新增 `BallProjectionCompensation` 或 `BallCompensationModel`。
3. 在 `CalibrationService` 中按模式分流：
   - `legacy`
   - `engineered`
4. 在调试球圈构建时，工程模式改用“物理半径”而不是“图像半径映射”。

## 10. 推荐实施顺序

### 第一阶段

先做“框架切换 + 配置记忆”，不改变传统模式行为：

1. 新增 `projection_mode`
2. 新增双路线文件路径
3. 设置页可切换并记忆
4. 启动按模式加载

### 第二阶段

做工程模式 MVP：

1. 平面层继续使用现有 `homography + residual field`
2. 调试球圈改成“补偿球心 + 固定物理半径”
3. 增加工程模式球体补偿文件

这一步通常就能先解决你现在最明显的“远端圈偏大”问题。

### 第三阶段

升级到正式工程模式：

1. 引入密集编码图案采样
2. 把局部残差从邻域加权升级到 TPS / B-spline
3. 增加球心补偿自动采样流程
4. 增加工程模式专用验收报表

## 11. 验收标准

建议把“看起来顺眼”改成以下机器验收：

- 平面 holdout `median < 1.0 mm`
- 平面 holdout `p95 < 2.0 mm`
- 袋口分区 `p95 < 2.5 mm`
- 距离梯度绝对值 `< 0.03 mm/cm`
- 球心补偿后，全台面球圈中心误差：
  - `median < 1.5 mm`
  - `p95 < 3.0 mm`
- 工程模式球圈半径误差：
  - `median < 1.0 mm`
  - `p95 < 2.0 mm`

## 12. 最终建议

如果目标是“少依赖人眼、能长期稳定复用、远端不再明显偏大”，推荐结论是：

- 保留当前传统模式，作为快速回退路径。
- 新增工程模式，并把它定义为“平面校准”和“球体补偿”两套文件联合生效。
- 工程模式下，球圈半径必须改为按真实球径投影，不能再沿用图像半径直接变换。

这条路线是当前项目最稳妥、也最符合工程经验的升级方向。

