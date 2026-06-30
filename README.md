# BAS 使用说明

## 简介

这是一个台球辅助系统，包含采集、检测、跟踪、状态机、路线规划和投影叠加等模块。

## 基本使用

1. 安装依赖：运行 `Setup_Environment.cmd`
2. 打开桌面控制台：运行 `Start_BAS.cmd`
3. 命令行启动界面：`.\.venv\Scripts\python.exe -m bas ui`
4. 命令行无界面运行主流程：`.\.venv\Scripts\python.exe -m bas run --headless --max-frames 90`

## 常用排查项

- 帧率较低时，优先检查检测后端、`target_shot`、`cue_sector`、录像回放和学习采样是否开启。
- 摄像头异常时，可先执行 `.\.venv\Scripts\python.exe -m bas probe-cameras`
- 标定问题可执行 `.\.venv\Scripts\python.exe -m bas inspect-calib`

## 投影动效设置

- 在“设置 -> 投影输出”中可以单独勾选“开启击球特效”和“开启胜利动画”。
- 取消勾选后，系统不会再自动触发对应动效，但交互调试按钮仍可手动测试播放。

## 目录说明

- `bas/`：核心代码
- `configs/`：默认配置
- `tests/`：回归测试
- `docs/`：设计与说明文档
