# M2P：DINOv2-B/14 弱监督点跟踪复现

本目录独立实现 [M2P 论文](https://arxiv.org/abs/2603.17813)中的 DINOv2-B/14 弱监督预训练，以及 TAP-Vid DAVIS、RGB-Stacking 评测。使用 YouTube-VOS 2019 和 DAVIS 2017 的图像与对象掩膜训练；推理仅输入视频和查询点。

## 实现与配置

主要配置为 `configs/m2p_v2b14.json`，论文未披露细节的实现假设保存在其中。

- `model.py`：冻结 DINOv2 骨干、LoRA、匹配细化头、教师匹配与三项损失。
- `geometry.py`：K-means 与 FPS 分组采点、Procrustes 相似变换、边界距离与坐标采样。
- `data.py`、`training.py`：自然视频采样、增强、梯度累积与验证集检查点选择。
- `evaluation.py`：TAP-Vid queried-first 评测；`pipeline.py` 为统一运行入口。
- `runtime.py`、`recovery.py`：必要运行记录、检查点保存及中断训练恢复。

训练每帧采样 12 组、每组 12 点，以 3 个高置信教师匹配估计局部运动。LSC、MLC、MBC 权重分别为 0.02、0.5、10；LoRA rank 为 64。训练分辨率 336×336，有效批次 16、微批次 2，AdamW 学习率 2e-5，150,000 次更新，在 110,000 步衰减至 0.1 倍。

验证使用官方 YouTube-VOS 验证图像及可用掩膜；只有初始化掩膜时使用明确记录的合成仿射配对。每 5,000 步按验证目标选择最佳检查点，TAP-Vid 不参与选择。

模型查询格式为 `N×(t,x,y)`，输出轨迹为 `T×N×(x,y)`。评测分辨率 448×448，报告可见点定位准确率；匹配分数未校准为遮挡概率，不报告 AJ/OA 的复现结果。本实现覆盖上述分支，未实现 DINOv3、M2P-Chrono、M2P-Tracker 或 TAP-Vid Kinetics。

## 数据与依赖

共享原始数据位于同级 `../data/`：

| 目录 | 用途 |
|---|---|
| `../data/youtubevos2019/` | 训练集与验证集图像、掩膜及元数据 |
| `../data/natural/DAVIS/` | DAVIS 2017 训练图像、掩膜与官方划分 |
| `../data/benchmarks/` | TAP-Vid DAVIS、RGB-Stacking 官方 pickle 数据 |
| `assets/` | 官方 DINOv2-B/14 初始化权重、TAP-Vid 指标源码及许可证 |
| `vendor/` | 固定版本 DINOv2 源码与许可证 |

YouTube-VOS 镜像固定为 `yinloonga/YouTube_VOS_2019@7a0eb80c729b0dbbceaea15950e86ed606ab21ed`；DINOv2 源码为 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；TAP-Vid 指标源码为 `730cda1c730877cfedbe01bf87fb1cadb78a565d`。

现有数据和依赖可直接使用。全新资源目录可使用 `.venv-train/bin/python -B scripts/prepare_m2p_assets.py` 准备上述资源；该入口不覆盖已有解压文件，不应对已准备的数据重复运行。训练入口直接检查实际资源，不自动下载。

## 运行

工作目录为 `/root/ultrasound/m2p`，使用 Python 3.11 训练环境：

```bash
cd /root/ultrasound/m2p
# 基线评测、训练、最佳检查点评测。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --config configs/m2p_v2b14.json --stage all

# 单独训练，不读取 TAP-Vid 基准。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --stage train

# 使用相同配置评测原始骨干与已保存的最佳检查点。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --stage evaluate

# 中断后从最佳适配权重继续，重新建立优化器。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --stage train --resume-best
```

默认输出为 `runs/m2p_reproduction/`，在实际运行时创建。保存实际配置、命令、代码与权重哈希、数据清单、训练日志、验证指标、最佳适配权重和评测预测。已有完整训练结果不会被新训练覆盖。

`--resume-best` 不恢复 AdamW 动量或随机状态，因此不等同于连续未中断训练。恢复时核对配置与验证目标，保留中断前日志并记录恢复步数。独立评测要求配置与检查点匹配。

当前 GPU 隔离固定为物理 GPU 1，UUID 为 `GPU-3d4c8194-7991-0a63-c517-075c83c0b393`、PCI 为 `0000:07:00.0`，进程内对应 `cuda:0`。

## 环境与验证

在可用的 Python 3.11 环境中重建训练环境：

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv-train/bin/python -m pip install -r requirements-train.txt
```

既有测试覆盖坐标、采样、损失梯度、梯度累积、检查点与恢复；模型相关检查需要上述 GPU 和初始化权重。测试输出使用系统临时目录：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-train/bin/python -B -m pytest -q -p no:cacheprovider tests
```
