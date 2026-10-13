# SemiEchoTracker：EchoXFlow A4C 三点实验

本目录沿用 Wu 等人的检测器、双向跟踪器和 PAST，实现一次 EchoXFlow A4C 训练与独立测试。实验入口为 `train.py`、`test.py`；本轮结果目录为 [`results/echoxflow/seed33_full/`](results/echoxflow/seed33_full/)。

论文：Han Wu 等，*Semi-Supervised Landmark Tracking in Echocardiography Video via Spatial-Temporal Co-Training and Perception-Aware Attention*，IEEE TMI，2026，[DOI](https://doi.org/10.1109/TMI.2026.3651389)。作者未公开 guided-DINOv2 的预训练流程，本实验使用公开 DINOv2 ViT-S/14 并冻结骨干，其余模块随机初始化，因此属于现有数据上的方法实现实验，不等同于论文原始数据集和预训练条件下的数值复现。

## 论文与实现

| 部分 | 实现 |
|---|---|
| 冻结 DINOv2 特征 | `models/dinov2.py`，显式指定源码和权重，严格加载 |
| PAST 空间与时间注意力 | `models/agent_attention.py`，`models/SemiEchoTracker.py`，3 层、Top-K 32、8 个头 |
| 逐帧检测与双向跟踪 | `models/GCN.py`；沿用 `conv_mlp_keypoint_cnn_gcn_noxy` 和分离图参数 |
| 端点监督、前后向一致性、速度及协同损失 | `train.py`；坐标项为逐点欧氏距离，前后向与协同项只使用中间帧，速度对应同一中间帧两侧的到达步 |
| 推理与评测 | `test.py`；仅运行检测器，不输入真实关键点、不平滑 |

检测器与两个跟踪方向分别计算协同距离后相加。训练只用端点标签初始化和监督跟踪器；验证依据检测器在全部 10 帧上的毫米 MRE 选择最佳检查点。

## 数据协议

原始数据位于 `/root/ultrasound/data/echoxflow/raw/`，版本为 `d907c536cae16750b9e46f269a27befd4979ebdd`。派生数据位于 `/root/ultrasound/data/semiechotracker/echoxflow/`，`manifest.json` 保存来源、内容哈希、折号、相位选择、时间关联、物理视野与排除项。

- 官方 fold 0–2 / 3 / 4 分别用于训练 / 验证 / 测试，实际保留 **160 / 64 / 64 段视频**，对应 **159 / 62 / 63 个检查**。检查 ID 和视频内容均不跨集合。
- 官方 A4C 关联中，77 条多平面关联无法解析；按视频内容去重后固定选最小标注记录 ID，去除 78 条重复标注；再排除 2 段 ED–ES 区间不足 10 个真实视频帧的记录，以及 1 段标注与视频检查 ID 不一致的记录。
- 由三帧中值滤波后的心内膜面积曲线估计 ES，取其前的面积最大值为 ED。在对应视频区间内均匀选择 10 个真实帧，每帧匹配最近轮廓，时间误差不超过半个轮廓采样间隔。端点为估计相位附近的实际视频帧。
- 点顺序为心尖代理点、ED 图像左侧基底端点、ED 图像右侧基底端点。心尖代理点是在 ED 时距两端中点最远的心内膜点，整段固定轮廓索引。
- 使用 EchoXFlow 官方几何工具转换至 224×224 正方形物理视野。每段保存像素到毫米的仿射变换；图像与标签共享几何增强，误差按每段自身的物理坐标计算。
- **参考来自软件轮廓，ED/ES 为估计相位，并非专家手工关键点真值。**训练缓存及读取器只有第 0、9 帧坐标；验证和测试包含全部 10 帧坐标。

数据准备使用本项目的 Python 3.13 环境 `.venv-data/`。官方扫描转换源码固定为 `b782e0a4e3dbaadf3c88fa572710bf4254d86143`，保存在 `vendor/echoxflow-b782e0a4e3dbaadf3c88fa572710bf4254d86143/`，含原始许可证；源码缺失时从[该固定版本](https://github.com/Ahus-AIM/EchoXFlow/tree/b782e0a4e3dbaadf3c88fa572710bf4254d86143)恢复到此目录。数据环境可在本项目目录重建：

```bash
python3.13 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements-data.txt
```

重新准备数据须保留原始 Zarr、元数据、`splits.csv`、许可证。准备入口不会覆盖已有派生目录：

```bash
cd /root/ultrasound/SemiEchoTracker
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-data/bin/python -B dataset/prepare_echoxflow.py --workers 4
```

## 运行

当前环境为 `/root/ultrasound/SemiEchoTracker/.venv-train/`。本轮使用物理 GPU 0（以下 UUID）、batch 4、FP32、种子 33、300 轮；Adam 学习率 0.001、weight decay 0，余弦调度，无 warmup；λ₁/λ₂/λ₃/λ₄ 为 1/0.1/0.1/1。标准增强沿用旋转、噪声、亮度对比度和模糊，整段共享增强参数。

```bash
cd /root/ultrasound/SemiEchoTracker
CUDA_VISIBLE_DEVICES=GPU-38e464b3-e728-fc20-01b4-81898424591b \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv-train/bin/python -B -u train.py \
  --data_dir /root/ultrasound/data/semiechotracker --data_type echoxflow \
  --backbone_source public \
  --backbone_repo /root/ultrasound/m2p/vendor/dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
  --backbone_checkpoint /root/ultrasound/SemiEchoTracker/weights/dinov2_vits14_pretrain.pth \
  --training_mode semi --optimizer adam --lr 0.001 --weight_decay 0 \
  --warmup_epochs 0 --epochs 300 --seed 33 --precision fp32 \
  --batch_size 4 --grad_accum_steps 1 --num_workers 4 \
  --tracking_supervised 1 --tracking 0.1 --velocity 0.1 --mutual_consistency 1 \
  --agent_num 32 --attention_layers 3 --best_metric err_init --run_tag seed33_full
```

已有运行目录不会被新训练覆盖。恢复时使用相同参数，并追加 `--resume /root/ultrasound/SemiEchoTracker/results/echoxflow/seed33_full/latest_model.pth`；恢复会核对配置、代码、骨干和数据哈希，加载优化器、调度器与随机状态。检查点编号内部从 0 开始，日志和测试指标的轮次从 1 开始。

完成训练后进行独立测试，配置从最佳检查点读取：

```bash
cd /root/ultrasound/SemiEchoTracker
CUDA_VISIBLE_DEVICES=GPU-38e464b3-e728-fc20-01b4-81898424591b \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv-train/bin/python -B test.py \
  --checkpoint results/echoxflow/seed33_full/best_model.pth \
  --split test --batch_size 4 --no_visuals
```

损失、时间索引、毫米转换、增强对齐及训练标签边界的回归检查：

```bash
cd /root/ultrasound/SemiEchoTracker
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv-train/bin/python -B -m unittest discover -s tests -v
```

## 结果入口

`seed33_full/` 保存 `config.json`、`provenance.json`、`data_manifest.json`、环境版本、必要源码快照、训练日志、逐轮指标，以及 `best_model.pth` 和 `latest_model.pth`。检查点包含继续训练所需状态。

独立测试输出位于 `outputs/test/metrics.json` 和 `outputs/test/predictions.npz`。指标按端点、中间帧、全部帧分别统计 MRE、逐点误差的总体标准差（ddof=0）及 SDR@2/3/4 mm，并保存三个点各自的指标。预测文件包含样本 ID、预测、参考、逐点毫米误差、帧索引、时间戳及坐标变换；配置、数据和预测通过 SHA-256 关联。

本轮完成 300 轮训练；验证集全部帧 MRE 最优的是第 **246 轮（5.2811 mm）**。该检查点在官方 fold 4 的 **64 段视频、63 个检查、640 帧、1920 次点观测**上，全部帧 MRE 为 **4.7130 ± 3.9643 mm**，SDR@2/3/4 mm 为 **18.70% / 34.58% / 52.29%**。标准差按单个点的误差统计，不代表不同随机种子的波动。

[实验结果讲解报告](/root/ultrasound/SemiEchoTracker/results/echoxflow/seed33_full/experiment_report.md)完整介绍数据处理与筛选、官方划分、论文评估体系、训练收敛、端点/中间帧/全部帧及三个关键点的结果，并说明与论文原始条件的差异。完整精度指标见 [metrics.json](/root/ultrasound/SemiEchoTracker/results/echoxflow/seed33_full/outputs/test/metrics.json)，对应预测见 [predictions.npz](/root/ultrasound/SemiEchoTracker/results/echoxflow/seed33_full/outputs/test/predictions.npz)。

## 引用

```bibtex
@article{wu2026semiechotracker,
  title={Semi-Supervised Landmark Tracking in Echocardiography Video via Spatial-Temporal Co-Training and Perception-Aware Attention},
  author={Wu, Han and Chen, Haoyuan and Zhou, Lin and Xu, Qi and Cui, Zhiming and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  doi={10.1109/TMI.2026.3651389}
}
```
