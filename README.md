# 火焰 / 烟雾检测

保留项目最后选定的双类别模型和一套官方 Ultralytics 训练、推理入口。

- `models/best.pt`：2026-10-25 项目最终选定的 protected_dual 权重，约 38 MB；类别 `0=smoke`、`1=fire`，输入尺寸默认 640。
- `train.py`、`predict.py`：共用官方 `ultralytics==8.4.190`，不重复收录框架源码。

## 安装

需要 Python 3.10+、Git 和 Git LFS。

```bash
git lfs install
git clone https://github.com/suiseye/fire-smoke-research-archive-20261026.git
cd fire-smoke-research-archive-20261026
git lfs pull --include="models/best.pt" --exclude=""
pip install -r requirements.txt
```

模型由 Git LFS 保存；克隆后需要下载实际权重。

## 推理

```bash
python predict.py --source image.jpg
python predict.py --source video.mp4 --device 0
```

结果保存在 `runs/predict/`。置信度默认 0.25，可用 `--conf` 调整。

## 训练

自行准备完整标注的数据集 YAML，类别顺序必须为 `0: smoke`、`1: fire`：

```bash
python train.py --data /path/to/data.yaml --device 0
```

这是基于最终权重的官方标准微调入口，不复现历史的部分标签监督和教师保护训练流程。本精简版不包含原始数据集、旧实验、各轮检查点或电控代码。

“最终”指项目最后选定版本，不表示已证明其精度最高或通过完整实机验收。安装兼容性和推理冒烟检查不代表检测效果验证。

采用 AGPL-3.0；官方框架源码和许可证见 [Ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.4.190)。
