# BAS 学习排序工具说明

这个目录按 `example/0623_重构_rl.md` 里最推荐的路线实现：主程序继续用几何/快速物理生成候选 shot，`rl` 只负责学习真实样本里的残差和候选排序。当前不把纯 RL 作为主算法。

## 目录用途

- `dataset.py`：读取主程序采集的 `shot_samples.jsonl`，并转换为训练矩阵。
- `model.py`：轻量 MLP 排序模型和 JSON 导出逻辑。
- `train.py`：训练多任务排序器，输出主程序可直接加载的 `ranker.json`。
- `inspect_samples.py`：统计样本数、候选数、进球/犯规标签。
- `make_demo_samples.py`：生成少量演示样本，用来检查工具链是否跑通。
- `data/`：本地样本目录，默认被 `.gitignore` 忽略。
- `models/`：本地训练结果目录，默认被 `.gitignore` 忽略。

## 1. 依赖状态

当前工作区的 `.venv` 已经安装运行、YOLO 和训练依赖，包括 PyTorch/CUDA。正常使用不需要再手动安装依赖。

如果以后目录被移动或 `.venv` 损坏，再运行根目录的：

```powershell
.\Setup_Environment.cmd
```

## 2. 在主程序中采集学习样本

打开 BAS 设置窗口：

- “学习样本目录”：默认 `rl/data/samples`
- 勾选“启用学习样本采集”
- “学习排序模型”可以先留空

主程序运行时会在样本目录下创建：

```text
rl/data/samples/learning_xxx/shot_samples.jsonl
```

每条样本包含击球前布局、候选 shot、事件序列、结束布局和基础标签。关闭采集开关后，主程序不会写训练样本。

## 3. 检查样本

```powershell
.\.venv\Scripts\python.exe -m rl.inspect_samples --samples rl\data\samples
```

如果暂时没有真实样本，可以先生成演示样本检查流程：

```powershell
.\.venv\Scripts\python.exe -m rl.make_demo_samples
.\.venv\Scripts\python.exe -m rl.inspect_samples --samples rl\data\samples
```

## 4. 训练排序模型

推荐直接双击根目录：

```text
Train_RL_One_Click.cmd
```

等价命令是：

```powershell
.\.venv\Scripts\python.exe -m rl.train --samples rl\data\samples --out rl\models\ranker.json --epochs 50
```

训练输出是轻量 JSON 模型，不要求主程序安装 PyTorch。训练目标包含：

- 进洞概率
- 母球摔袋概率
- 犯规概率
- 留位收益
- 候选排序相关性

## 5. 让主程序使用训练结果

在 BAS 设置窗口中：

- 勾选“启用学习排序”
- “学习排序模型”指向 `C:\CodeProject\2606BAS\rl\models\ranker.json`
- “权重”控制学习分数和原几何分数的混合比例，默认 `0.65`

模型路径为空、文件不存在或格式不兼容时，主程序自动退回原来的几何/物理排序，仍可独立运行。

也可以直接改配置：

```yaml
learning:
  ranker_enabled: true
  ranker_model_path: rl/models/ranker.json
  score_blend: 0.65
  collect_enabled: false
  samples_directory: rl/data/samples
```

## 6. 建议工作流

1. 先关闭学习排序，只开启学习样本采集。
2. 采集一批真实击球样本。
3. 用 `inspect_samples.py` 检查样本数量和标签分布。
4. 用 `train.py` 训练并导出 `ranker.json`。
5. 在主程序设置里指向模型路径，开启学习排序。
6. 先保留回放和样本采集，观察学习排序和原几何排序的差异。
