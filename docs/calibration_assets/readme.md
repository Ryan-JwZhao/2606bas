# 投影校正打印素材说明

本目录保存可打印的校正辅助文件。当前提供 A4 横向 ChArUco 主板：

```text
charuco_a4_landscape_10x7_25mm.svg
```

使用方式：

1. 用浏览器、Inkscape 或矢量图查看器打开 SVG。
2. 选择 A4 横向打印。
3. 缩放选择 `100%` 或“实际大小”，不要选择“适合页面”。
4. 打印后用尺检查棋盘宽度为 `250 mm`，高度为 `175 mm`，单格为 `25 mm`。
5. 将纸张贴到硬质平板上，避免纸面弯曲。

外部标定脚本参数：

```python
squares_x = 10
squares_y = 7
square_length_m = 0.025
marker_length_m = 0.018
dictionary_id = 0  # OpenCV DICT_4X4_50
```

注意事项：

- 这张主板适合整板移动采集或单板局部采集。
- 如果多张 ChArUco 小板同时进入相机画面，每张板应使用唯一 marker ID。
- 不建议把多张相同 ID 的打印板同时摆在台面上做自动匹配。
- 更完整的流程见 `docs/projection_calibration_workflow.md`。

