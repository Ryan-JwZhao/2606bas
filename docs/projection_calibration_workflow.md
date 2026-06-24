# 投影校正详细工作流

本文用于现场完成 BAS 的投影校正、结果验证和常见问题排查。当前程序已经支持校正码显示、投影窗口管理、校正结果可视化、残差箭头显示和 Holdout 验证；完整的结构光采集、投影仪建模、bundle adjustment 或局部残差拟合求解，仍需要外部标定脚本生成 `projection_calibration.json`。

## 1. 先弄清三个坐标系

投影校正的核心是把相机看到的点，稳定映射到投影仪要打出的点：

```text
camera_px  ->  projector_px  ->  table_mm
```

- `camera_px`: 相机图像像素坐标，原点在图像左上角，单位是像素。
- `projector_px`: 投影画面像素坐标，原点在投影画面左上角，单位是像素。
- `table_mm`: 台面毫米坐标，默认用台面内框矩形映射到 `0..table_width_mm` 和 `0..table_height_mm`。

不要把投影校正和颗星公式混在一起：

- 投影校正文件 `projection_calibration.json` 决定路线 overlay 从相机坐标到投影坐标是否准确。
- 颗星公式只是辅助显示网格、标签、位置、缩放和旋转，不替代相机-投影仪几何校正。
- 相机畸变矫正负责先把镜头畸变拉直；如果相机畸变文件错了，后面的投影校正会一起偏。

## 2. 当前程序能做什么

主界面点击“校正投影仪”会打开投影仪校正向导，向导内可以：

- 打开指定屏幕上的全屏投影窗口。
- 显示全屏 ChArUco 校正码。
- 在 OpenCV ArUco 不可用时，显示编码网格和定位十字。
- 加载当前 `projection_calibration.json` 并显示台面多边形、对应点编号。
- 显示局部残差箭头。
- 选择 Holdout JSON 并计算图像误差、台面毫米误差、分区 P95 和误差随距离梯度。
- 恢复运行时实时投影 overlay。

当前程序不会自动完成：

- 多帧结构光采集。
- 投影仪作为反向相机的完整模型估计。
- bundle adjustment。
- 为多个小 ChArUco 板分配唯一 marker ID。
- 自动产出正式的 `projection_calibration.json`。

因此现场流程是：程序负责显示与验证，外部标定脚本负责采集和求解。

## 3. 需要重新校正的情况

只要以下任一项发生变化，就应重新做投影校正：

- 相机位置、角度、高度、镜头焦距、对焦环、分辨率或裁切区域变化。
- 投影仪位置、角度、焦距、变焦、梯形校正、分辨率或系统缩放变化。
- 台球桌移动，或者投影仪/相机支架发生明显振动。
- 更换相机内参文件或开启/关闭实时去畸变。
- 更换投影屏幕编号，或者 Windows 显示设置里的主副屏布局变化。

通常不需要重新校正：

- 只调整 UI 主题、检测模型、规划参数。
- 只改变曝光或灯光，但画面几何没有变化。
- 只调整颗星公式的显示开关、标签偏移或轻微视觉样式。

## 4. 文件和命令速查

常用文件：

- 默认配置：`configs/default.yaml`
- 本机 UI 设置：`local_settings/user_settings.json`
- 相机内参文件：`calibration.camera_file`
- 投影校正文件：`calibration.projection_file`
- 投影屏幕与分辨率：`projection.screen_index`, `projection.projector_width`, `projection.projector_height`
- A4 打印 ChArUco 主板：`docs/calibration_assets/charuco_a4_landscape_10x7_25mm.svg`

常用命令：

```powershell
.\.venv\Scripts\python.exe -m bas doctor
.\.venv\Scripts\python.exe -m bas inspect-calib
.\.venv\Scripts\python.exe -m bas verify-calib C:\path\to\holdout.json
.\.venv\Scripts\python.exe -m bas ui
```

建议每次正式校正前先运行：

```powershell
.\.venv\Scripts\python.exe -m bas doctor
.\.venv\Scripts\python.exe -m bas inspect-calib
```

`doctor` 用于检查依赖和路径，`inspect-calib` 用于确认当前加载的相机和投影校正是否有效。

## 5. 现场准备

硬件准备：

- 固定相机和投影仪，确认支架不会被碰动。
- 关闭投影仪自动梯形校正、自动画面增强、自动缩放等会改变几何的功能。
- Windows 显示设置中确认投影仪分辨率与程序设置一致。
- 投影仪尽量使用原生分辨率，避免系统层面缩放或镜像模式带来的二次缩放。
- 相机对焦、投影对焦都要在真实台面距离上完成。
- 台面清洁，尽量减少高反光物体。

软件准备：

1. 双击 `Start_BAS.cmd` 打开 UI。
2. 进入“设置”，确认：
   - `相机标定文件` 指向有效 OpenCV YAML。
   - `投影校正文件` 指向准备写入或替换的 JSON 路径。
   - `默认投影设备` 指向真实投影仪屏幕。
   - `默认投影分辨率` 与系统显示设置一致。
   - 若启用相机畸变矫正，`工业相机畸变矫正` 文件必须和当前相机分辨率匹配。
3. 保存设置后，点击“初始化图形图像模块”。
4. 点击“探测相机”，确认当前工业相机能稳定出图。

## 6. 相机内参先行

投影校正依赖相机画面几何稳定。建议先完成相机 ChArUco 内参标定，并保存为 OpenCV YAML。

内参采集建议：

- 使用平整、哑光、尺寸准确的 ChArUco 板。
- 采集不同位置、不同倾斜角度的图片，覆盖画面中心、四角、近边和远边。
- 保证角点清晰，不要过曝、虚焦或运动模糊。
- 如果真实运行时会启用 1920x1080，就用同样分辨率做内参标定。

完成后检查：

```powershell
.\.venv\Scripts\python.exe -m bas inspect-calib
```

如果 `camera_valid` 为 `false`，不要继续做投影校正，先修复相机内参路径或文件格式。

## 7. 打开投影校正向导

UI 内操作：

1. 点击“校正投影仪”。
2. 点击“打开投影窗口”。
3. 若投影画面出现在错误屏幕，回到“设置”调整“默认投影设备”。
4. 点击“显示 ChArUco 校正码”。
5. 若 ChArUco 生成失败，程序会自动退回“编码网格/定位十字”；这只能用于人工对齐和排查，正式自动角点采集仍建议使用真正的 ChArUco 或外部结构光图案。

此时不要点击“恢复实时投影”，否则实时路线 overlay 会重新接管投影窗口。

## 8. 主校正采集

主校正目标是建立大范围的 `camera_px -> projector_px` 单应矩阵或更高阶模型。

采集建议：

- 至少采集 20 组有效姿态，正式交付建议 30 组以上。
- 每组都要记录相机看到的角点和投影仪对应点。
- 覆盖靠近投影仪的一端、远离投影仪的一端、两条长边、六个袋口附近和台面中心。
- 不要只在中心区域采集，否则边角和袋口容易漂。
- 每采一组先检查角点数量和重投影误差，坏帧直接剔除。

质量要求：

- ChArUco 角点要清晰，棋盘不要弯曲。
- 板面不要被球、袋口阴影或灯具反光遮挡。
- 外部脚本应保存原始图、检测结果和求解日志，方便回溯。

## 9. 局部精修与 Holdout

主校正完成后，还需要在局部区域补控制点，尤其是袋口和远端区域。

推荐区域：

- 六个袋口：`pocket_lt`, `pocket_mt`, `pocket_rt`, `pocket_rb`, `pocket_mb`, `pocket_lb`
- 两条长边中段：`long_rail_top_mid`, `long_rail_bottom_mid`
- 靠近投影仪短边：`near_short_rail`
- 远离投影仪短边：`far_short_rail`
- 台面中心：`center`

如果使用多张小 ChArUco 纸板：

- 多张同时入镜时，每张板必须使用唯一 marker ID，否则外部脚本可能把角点归错板。
- 当前仓库提供的 A4 主板适合整板采集或单板移动采集；不要把多张相同 ID 的板同时放在画面里做自动匹配。
- 如果暂时没有唯一 ID 多板套件，可以逐区域单板采集：每次只放一张板，采完移动到下一个区域。

Holdout 数据必须独立于求解数据：

- 不要把参与求解的点再拿来做验收。
- 每个关键区域至少保留 3-5 个 holdout 点。
- Holdout 样本建议同时包含 `camera_px`、`projector_px`、`world_mm`、`zone` 和 `distance_cm`。

## 10. 生成 projection_calibration.json

外部脚本最终应生成类似结构：

```json
{
  "mode": "external_bundle_adjustment",
  "saved_at": "2026-06-24 15:00:00",
  "homography": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "cam_points": [[100.0, 100.0], [1800.0, 100.0], [1800.0, 900.0], [100.0, 900.0]],
  "proj_points": [[80.0, 60.0], [1200.0, 70.0], [1210.0, 740.0], [70.0, 730.0]],
  "residual_cam_points": [[500.0, 300.0]],
  "residual_proj_offsets": [[0.4, -0.2]],
  "table_polygon_cam": [[100.0, 100.0], [1800.0, 100.0], [1800.0, 900.0], [100.0, 900.0]],
  "table_polygon_proj": [[80.0, 60.0], [1200.0, 70.0], [1210.0, 740.0], [70.0, 730.0]],
  "projector_size": [1280, 800],
  "quality_report": {
    "mean_px": 0.12,
    "median_px": 0.09,
    "p95_px": 0.28,
    "max_px": 0.42
  }
}
```

字段说明：

- `homography`: 基础相机到投影仪单应矩阵，至少需要 4 对有效点。
- `cam_points` 和 `proj_points`: 参与预检查的相机/投影仪对应点。
- `residual_cam_points` 和 `residual_proj_offsets`: 局部残差控制点。运行时会在基础单应矩阵后叠加邻域加权偏移。
- `table_polygon_proj`: 台面内框在投影坐标中的多边形。建议一定写入；缺失时程序只能按投影画面留边估算。
- `projector_size`: 必须和实际运行分辨率一致。
- `quality_report`: 外部脚本的质量报告，供人工追踪；程序会重新计算 `cam_points/proj_points` 的像素误差。

## 11. 加载与预览

生成 JSON 后：

1. 在 UI “设置”中把“投影校正文件”指向新 JSON。
2. 保存设置。
3. 点击“校正投影仪”。
4. 点击“显示当前校正结果”。
5. 检查台面多边形是否压在真实台面内框上。
6. 检查对应点编号是否落在采集位置附近。
7. 点击“显示残差箭头”，查看局部修正方向是否合理。

命令行检查：

```powershell
.\.venv\Scripts\python.exe -m bas inspect-calib
```

重点看：

- `projection_valid` 是否为 `true`
- `projection_source` 是否是新文件路径
- `projector_size` 是否匹配当前投影分辨率
- `projection_error.mean_px/p95_px` 是否在预期范围

## 12. Holdout 验收

Holdout JSON 可以是数组，也可以是带 `samples` 字段的对象：

```json
{
  "samples": [
    {
      "camera_px": [123.4, 567.8],
      "projector_px": [456.7, 321.0],
      "world_mm": [1000.0, 500.0],
      "zone": "pocket_lt",
      "distance_cm": 120.0
    }
  ]
}
```

运行：

```powershell
.\.venv\Scripts\python.exe -m bas verify-calib C:\path\to\holdout.json
```

验收阈值：

| 指标 | MVP 可用 | 正式交付建议 |
|---|---:|---:|
| 图像重投影 mean | `< 0.20 px` | `< 0.20 px` |
| 图像重投影 P95 | `< 0.35 px` | `< 0.35 px` |
| 台面毫米 median | `< 1.5 mm` | `< 1.0 mm` |
| 台面毫米 P95 | `< 3.0 mm` | `< 2.0 mm` |
| 任一袋口区域 P95 | `< 2.5 mm` | `< 2.5 mm` |
| 误差随距离梯度绝对值 | `< 0.03 mm/cm` | `< 0.03 mm/cm` |

如果像素误差很好但毫米误差差，优先检查 `table_polygon_proj`、台面尺寸和 `world_mm` 定义是否一致。

## 13. 结果判读

正常结果：

- 台面多边形压住真实内框，不明显旋转或平移。
- 对应点编号和采集位置一致。
- 残差箭头短而方向分散，没有某一区域整体同向漂移。
- Holdout 的分区 P95 没有单独爆掉的袋口。

需要补采：

- 某一端的残差箭头整体朝同一方向。
- 远离投影仪一端误差显著大于近端。
- 某个袋口区域 P95 大于 `2.5 mm`。
- 距离梯度绝对值超过 `0.03 mm/cm`。

需要重做主校正：

- 台面整体旋转、缩放或错位。
- `projector_size` 和真实分辨率不一致。
- 改了投影仪梯形校正或 Windows 缩放。
- 相机内参无效，或运行时重复做了去畸变。

## 14. 常见问题排查

| 现象 | 优先检查 |
|---|---|
| 投影窗口不在投影仪上 | “设置”里的默认投影设备，Windows 屏幕顺序，是否用了复制屏而不是扩展屏。 |
| 投影画面被系统缩放 | Windows 显示缩放设为 100%，投影仪使用原生分辨率，程序分辨率与系统一致。 |
| ChArUco 检测不到 | 打印比例、焦距、曝光、反光、OpenCV ArUco 字典、棋盘规格是否匹配。 |
| 中心准、边缘偏 | 主采集覆盖不足，镜头畸变文件不准，局部残差控制点太少。 |
| 近端准、远端偏 | 投影仪角度太斜，远端采样不足，投影画面经过梯形校正。 |
| 袋口附近偏 | 袋口区域缺少局部控制点，袋口阴影影响角点，holdout 分区样本太少。 |
| 运行路线和校正预览不一致 | 是否加载了旧 `projection_calibration.json`，是否保存了 UI 设置，是否重启/重新初始化。 |
| 验收像素误差好但路线仍偏 | 检查台面毫米坐标、球心检测、`table_polygon_proj` 和台面内框定义。 |
| 多张小板同时入镜后点乱跳 | 多张板使用了重复 marker ID；改成唯一 ID 套件，或一次只采一张板。 |

## 15. A4 打印校正码

仓库内提供一张可打印 A4 横向 ChArUco 主板：

```text
docs/calibration_assets/charuco_a4_landscape_10x7_25mm.svg
```

规格：

- 页面：A4 横向，`297 mm x 210 mm`
- ChArUco：`10 x 7` squares
- 单格边长：`25 mm`
- ArUco marker 边长：`18 mm`
- 字典：OpenCV `DICT_4X4_50`，对应当前代码里的 `dictionary_id = 0`

打印要求：

- 横向打印。
- 缩放设为 `100%` 或“实际大小”。
- 关闭“适合页面”“填满页面”等自动缩放。
- 打印后用尺检查棋盘宽度应为 `250 mm`，高度应为 `175 mm`，单格应为 `25 mm`。
- 建议贴在硬质平板或泡沫板上，保持平整。

外部脚本使用这张纸板时，ChArUco 参数应设置为：

```python
squares_x = 10
squares_y = 7
square_length_m = 0.025
marker_length_m = 0.018
dictionary_id = 0
```

## 16. 现场记录模板

建议每次校正留一份记录，方便之后复盘：

```text
日期:
操作者:
相机型号/序列号:
相机分辨率:
相机内参文件:
投影仪型号:
投影分辨率:
Windows 缩放:
投影屏幕编号:
投影校正文件:
主采集样本数:
局部精修样本数:
Holdout 样本数:
图像 mean/P95:
台面 median/P95:
最差分区:
距离梯度:
是否通过:
备注:
```

