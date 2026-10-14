# M2P：DINOv2-B/14 弱监督点跟踪复现

本目录独立实现 [M2P 论文](https://arxiv.org/abs/2603.17813)中的 DINOv2-B/14 弱监督训练，支持自然视频预训练与 TAP-Vid 评测，以及 EchoXFlow A4C 心肌点跟踪实验。训练通过掩膜约束点匹配，推理仅输入视频和查询点。

## 实现与配置

自然视频配置为 `configs/m2p_v2b14.json`；A4C 配置为 `configs/m2p_echoxflow_a4c.json`。论文未披露细节的实现假设保存在配置中。

- `model.py`：冻结 DINOv2 骨干、LoRA、匹配细化头、教师匹配与三项损失。
- `geometry.py`：K-means 与 FPS 分组采点、Procrustes 相似变换、边界距离与坐标采样。
- `data.py`、`training.py`：自然视频采样、增强、梯度累积与验证集检查点选择。
- `echoxflow.py`：独立读取 EchoXFlow、扫描转换、心肌掩膜缓存与帧对采样。
- `evaluation.py`：TAP-Vid 和 EchoXFlow 点跟踪评测；`pipeline.py` 为统一运行入口。
- `runtime.py`、`recovery.py`：必要运行记录、检查点保存及中断训练恢复。

训练每帧采样 12 组、每组 12 点，以 3 个高置信教师匹配估计局部运动。LSC、MLC、MBC 权重分别为 0.02、0.5、10；LoRA rank 为 64。自然视频训练分辨率 336×336，有效批次 16、微批次 2，AdamW 学习率 2e-5，150,000 次更新，在 110,000 步衰减至 0.1 倍。

验证使用官方 YouTube-VOS 验证图像及可用掩膜；只有初始化掩膜时使用明确记录的合成仿射配对。每 5,000 步按验证目标选择最佳检查点，TAP-Vid 不参与选择。

模型查询格式为 `N×(t,x,y)`，输出轨迹为 `T×N×(x,y)`。评测分辨率 448×448，报告可见点定位准确率；匹配分数未校准为遮挡概率，不报告 AJ/OA 的复现结果。本实现覆盖上述分支，未实现 DINOv3、M2P-Chrono、M2P-Tracker 或 TAP-Vid Kinetics。

## 数据与依赖

共享原始数据位于同级 `../data/`：

| 目录 | 用途 |
|---|---|
| `../data/youtubevos2019/` | 训练集与验证集图像、掩膜及元数据 |
| `../data/natural/DAVIS/` | DAVIS 2017 训练图像、掩膜与官方划分 |
| `../data/benchmarks/` | TAP-Vid DAVIS、RGB-Stacking 官方 pickle 数据 |
| `../data/echoxflow/raw/` | EchoXFlow 原始 Zarr、Croissant 关联和官方五折划分 |
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

配置中的 `gpu` 选择 `nvidia-smi` 显示的物理 GPU 编号，运行时解析其 UUID 和 PCI，在进程内映射为 `cuda:0`。自然视频配置为 GPU 1，A4C 配置为 GPU 0；修改该字段即可更换设备。GPU 和运行路径不参与检查点的实验参数兼容性比较。

## EchoXFlow A4C 实验

训练、验证和测试均限定 `views: ["4ch"]`，直接从官方 DINOv2-B/14 权重开始，不依赖自然视频训练检查点。原始数据固定为 `d907c536cae16750b9e46f269a27befd4979ebdd`，几何及轮廓定义参照官方工具版本 `b782e0a4e3dbaadf3c88fa572710bf4254d86143`。

完成视频内容去重、时间对齐与掩膜检查后的划分如下。检查 ID、源数组校验和、时间匹配和排除原因保存在运行目录中的 `echoxflow_manifest.json`。

| 集合 | 官方折号 | A4C 序列 | 检查数 |
|---|---|---:|---:|
| 训练 | 2、3、4 | 186 | 183 |
| 验证 | 1 | 49 | 48 |
| 测试 | 0 | 56 | 56 |

仅使用明确指向视频数组的关联。同内容重复标注选择有效时间对齐帧最多者，相同时按标注记录 ID 排序。标注与视频记录必须属于同一官方折，视频内容不得跨折。时间对齐采用最近邻，容差为中位帧间隔的一半加 1e-6 秒；多个标注映射到同一帧时保留误差最小者，不向未标注时间外推。

由成像几何建立覆盖完整扇区的物理等比例正方形网格，将原始深度×波束数组转换为 448×448 灰度图，扇区外为零。五层轮廓间的四条壁厚带构成心肌掩膜，去除心腔并裁至有效成像区域。缓存按帧分块，保存在本次运行目录的 `cache/`；数据源、处理代码或几何参数不匹配时明确报错，不静默复用。

每个训练样本均匀随机选择一条训练序列，再选取其中两个不同的有效标注帧，帧间隔不固定。训练只读取图像、掩膜与距离图，从第一帧掩膜采样查询点；原始轮廓的跨帧点索引不进入训练。图像缩放至 336×336，应用现有仿射增强及灰度亮度、对比度增强，再复制为三通道。

A4C 保留论文模型及三项损失，采用 20,000 步域内训练预算，在第 15,000 步将学习率衰减至 0.1 倍。初始学习率 2e-5、有效批次 16、微批次 1、种子 20261002；每 500 步评测固定验证帧对，以掩膜加权损失最低者保存 `echoxflow/best.pt`。该预算与论文的自然视频预训练不同。

测试参考为每组五层轮廓中的中层点（索引 2、7、12、…）。每个有至少两个有效位置的点从首次有效帧发起查询，评测后续有效参考位置；查询帧和扇区外参考位置不计分，模型预测越界仍计入误差。模型不读取目标帧掩膜，不进行测试时优化。

主要指标为物理坐标下的 EPE（mm），同时报告将物理正方形映射到 256 标准坐标后的 PCK@1/2/4/8/16 及平均 PCK。先对序列取均值，再对检查取均值；训练前后差值采用按检查配对的 2,000 次 bootstrap 给出 95% 区间。参考轨迹来自软件追踪，结果表征与该参考的一致性；不报告 OA/AJ。

工作目录同上，运行命令为：

```bash
# 只准备 A4C 缓存，不使用 GPU。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --config configs/m2p_echoxflow_a4c.json --stage prepare

# 原始 DINOv2 评测、20,000 步训练、最佳 M2P 评测。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --config configs/m2p_echoxflow_a4c.json --stage all

# 单独评测已保存的最佳检查点。
.venv-train/bin/python -B -u -m m2p_lab.pipeline --config configs/m2p_echoxflow_a4c.json --stage evaluate
```

输出目录由 `run_dir` 指定，默认 `runs/m2p_echoxflow/a4c_fold0_seed20261002/`。评测结果位于 `evaluation/dinov2_baseline/summary.json`、`evaluation/m2p_best/summary.json`，配对差值和区间位于 `evaluation/comparison.json`。清单与缓存需要配套保留；重建缓存时使用新的 `run_dir` 执行 `prepare`。已有完成的实验使用新运行目录，不覆盖历史结果。

### A4C fold 0 结果

数据处理、划分、论文方法与评估指标对应关系、完整结果及解释见本运行目录中的 [实验结果说明](runs/m2p_echoxflow/a4c_fold0_seed20261002/experiment_report.md)。

`a4c_fold0_seed20261002` 已在 GPU 0 完成 20,000 次更新及 40 次验证，训练耗时 4 小时 15 分 8 秒（含验证）。验证损失最低的检查点位于第 18,500 步，加权损失为 0.891973。

测试覆盖 56 条序列、56 个检查、2,016 个查询点；两种模型使用相同查询及参考位置。排除查询帧后，有效位置覆盖率为 99.73%（105,414／105,699）。下表 EPE 单位为 mm，PCK 单位为 %，按序列、检查依次取均值。

| 模型 | EPE ↓ | PCK@1 | PCK@2 | PCK@4 | PCK@8 | PCK@16 | 平均 PCK ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 DINOv2-B/14 | 7.022 | 1.64 | 6.16 | 20.40 | 48.94 | 81.41 | 31.71 |
| M2P，最佳验证检查点 | 2.242 | 13.14 | 36.83 | 71.32 | 93.95 | 99.61 | 62.97 |

按检查配对的 2,000 次 bootstrap 显示，训练后减训练前的 EPE 差值为 −4.780 mm，95% 区间为 [−5.006, −4.572] mm；平均 PCK 差值为 +31.26 个百分点，95% 区间为 [29.76, 32.64]。结果仅描述这一 A4C 划分下与软件参考轨迹的一致性。

完整精度的指标及各阈值差值区间见 [comparison.json](runs/m2p_echoxflow/a4c_fold0_seed20261002/evaluation/comparison.json)，最佳权重见 [best.pt](runs/m2p_echoxflow/a4c_fold0_seed20261002/echoxflow/best.pt)。该运行保存了 `environment.json`、`provenance.json` 和 `source_code.tar.gz`，分别记录实际环境、代码与输入指纹、执行源码。

## 环境与验证

在可用的 Python 3.11 环境中重建训练环境：

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv-train/bin/python -m pip install -r requirements-train.txt
```

既有测试覆盖坐标、采样、损失梯度、梯度累积、检查点与恢复，以及 EchoXFlow 的时间对齐、参考轨迹隔离、毫米指标和 GPU 配置；模型相关检查需要 GPU 和初始化权重。测试输出使用系统临时目录，可通过 `M2P_TEST_CONFIG` 指定模型检查所用配置：

```bash
M2P_TEST_CONFIG=configs/m2p_echoxflow_a4c.json PYTHONDONTWRITEBYTECODE=1 .venv-train/bin/python -B -m pytest -q -p no:cacheprovider tests
```
