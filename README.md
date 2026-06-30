# BAS 本地说明

## 简介
`2606BAS` 是一套台球辅助程序，包含实时采集、检测追踪、状态机、路线规划、前端预览与投影输出。

当前支持三类主要路线模式：
- `规则模式`：按当前回合目标花色规划推荐进球路线。
- `自由模式`：按球杆方向模拟白球自由碰撞路线。
- `目标锁定模式`：当球杆持续指向某颗目标球时，锁定目标并尝试单独规划理论进球路线。

## 启动方式
在项目根目录运行：

```powershell
python -m bas
```

推荐使用流程：
1. 先确认相机、标定文件、检测模型和投影设备路径都已配置。
2. 打开程序后先开始采集，确认前端预览正常。
3. 如需落地投影，再打开投影窗口。
4. 根据现场需求切换 `规则模式` / `自由模式`。
5. 如需刷新几何或图像模块，使用界面里的初始化入口。

## 本次新增：投影交互系统
这次新增了一层独立的投影交互控制器，负责：
- 3 秒文字提示
- 袋口进球 RGBA 序列动画
- 全局胜利 RGBA 序列动画
- 调试用的手动触发入口

这层逻辑直接叠加在投影输出上，特点是：
- 不参与相机侧几何校正
- 不对 RGBA 动画做 OpenCV 畸变矫正
- 按整张 `1280x800` 画面直接覆盖到投影输出
- 动画默认按 `12~16 fps` 播放，优先保证投影端平稳

## 文字提示
所有文字提示默认显示 3 秒，样式为：
- 白色小字
- 黑色描边
- 显示在 `pocket + inline` 连线对应的球心可达范围左上角

当前已接入的提示场景：
- 系统开机
- 图形图像模块初始化成功
- 星公式开启 / 关闭
- 规则模式 / 自由模式切换
- 下一杆自由
- 下一杆黑球
- 精彩时刻已触发
- 精彩时刻已导出
- 目标锁定模式成功锁定目标球

说明：
- 只要目标球已经锁定，就会提示；即使这次没有算出可行击球路线，也会提示锁定成功。

## RGBA 动画素材
程序会从本地目录读取 PNG 序列：
- 袋口动画：`example/motion/Goal/pocket0` 到 `example/motion/Goal/pocket5`
- 胜利动画：`example/motion/Win`

袋口映射关系与 `pocket.json` 保持一致：
- `pocket0`：左上
- `pocket1`：中上
- `pocket2`：右上
- `pocket3`：右下
- `pocket4`：中下
- `pocket5`：左下

运行时行为：
- 状态机产生 `POCKET_CONFIRMED` 后，自动播放对应袋口动画。
- 状态机产生 `GAME_OVER_CANDIDATE` 后，自动播放全局胜利动画。

## 手动调试方式
主界面侧边栏新增了 `交互调试` 区块：
- `P0 ~ P5`：手动触发 6 个袋口动画
- `结算`：手动触发全局胜利动画

如果投影窗口还没打开，点击这些按钮会自动打开投影窗口，方便直接调样式、调节素材节奏和观察覆盖效果。

## 主要实现文件
- `bas/projection/interaction.py`
- `bas/ui/main_window.py`
- `tests/test_projection_interaction.py`
- `tests/test_ui_runtime.py`

## 本地验证
建议至少执行：

```powershell
.\.venv\Scripts\python.exe -m py_compile bas\projection\interaction.py bas\ui\main_window.py tests\test_projection_interaction.py tests\test_ui_runtime.py
.\.venv\Scripts\python.exe -m pytest tests\test_projection_interaction.py tests\test_ui_runtime.py -q --basetemp=pytest_tmp_codex\projection_interaction
```

## .gitignore 约定
项目默认不提交本地运行产物、大文件和素材目录，例如：
- `logs/`
- `outputs/`
- `replays/`
- `local_settings/`
- `example/`

说明：
- `example/motion/` 下的 RGBA 序列会被程序直接读取，但默认作为本地素材保留，不纳入版本控制。
- 如果后续需要共享动画素材，建议通过单独压缩包或专门素材仓库管理，不直接把整套 PNG 序列提交到主仓库。
