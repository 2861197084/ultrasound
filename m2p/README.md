# M2P 复现代码与原始数据

## EchoXFlow 全周期 mask 弱监督实验

既有实验使用 `configs/echoxflow_mask_pairs.json` 和仓库自身模型/训练实现，使用 291 段连续图像及固定划分：119 训练、46 验证、64 测试，另 62 段 review 不用于训练/验证。实验成果和配置保留；旧 ED/ES、连续图像与 mask 缓存均需从新原始库重建后才能重新训练。

历史配置中的 `prepared`、`pairs` 和 `raw` 路径当前不存在。样本清单及关联保存在 `../data/echoxflow/legacy_provenance/`，固定划分位于 `../data/echoxflow/legacy_provenance/EchoTracker2/outputs/echoxflow_ed_es_finetune/split.json`；历史配置的 `split_manifest` 仍保留旧路径，新运行需指定实际位置。`echo_masks` 读取已有连续图像与 ED/ES 缓存，不负责从原始库生成这些输入。

连续评测和报告还依赖 EchoTracker2 的本地扩展模块 `evaluate`、`evaluation_data`、`analyze_echoxflow_sequences` 和 `analyze_finetuning`，上游原始版本不包含这些模块。需恢复或替换这些依赖，再按历史时相、几何、样本 ID 和划分重建缓存。输入、依赖和独立输出均准备好后，在 `/root/ultrasound/m2p` 使用以下入口：

```bash
.venv-data/bin/python -B -m m2p_lab.echo_masks --config configs/echoxflow_mask_pairs.json
.venv-train/bin/python -u -m m2p_lab.pipeline --config configs/echoxflow_mask_pairs.json --stage all
# 中断后恢复优化器、调度器、随机状态及抽样位置：在上一命令追加 --resume
# 缓存齐备时，从保存预测重算汇总：将 --stage all 改为 --stage report
```

训练均匀抽视频及其全周期中两张有效 mask 帧，两个时间方向均可；不读取跨帧点对应。每 1,000 步用验证集 ES 点 EPE 选择检查点。微批次 8、累积 2 次、有效批次 16，20,000 次更新，物理 GPU 1。推理仅输入图像与 ED 查询点，逐目标帧匹配，`chunk_size=1` 保证不同输入窗口的共同帧数值一致；既有实验未使用时序模块。

结果、源代码快照、恢复状态与完整预测保存到 `runs/echoxflow_mask_pairs/`；`status.json` 表示历史运行的实际阶段。`--stage report` 仍读取连续图像缓存核对参考、生成图表，当前也需先恢复该输入。无配置参数的旧入口还依赖已移除的历史预检与训练缓存，新实验使用明确的 `--config` 和独立输出，不覆盖历史运行。

本目录保留 M2P 独立复现代码、环境、权重和实验成果。各项目共用的原始数据位于 `/root/ultrasound/data/`；原始库的更新不改变既有实验的标签定义和样本划分。

这是对 [M2P, arXiv:2603.17813v1](https://arxiv.org/abs/2603.17813) 的独立实现，不是作者发布的代码或训练权重。论文未披露细节的实现假设保留在 `configs/m2p_v2b14.json` 中。

## 保留内容

| 路径 | 内容 |
|---|---|
| `../data/echoxflow/raw/` | 左心室 A4C、A2C、ALAX 标注及关联原始视频，1,323 个 Zarr、13,085 个数组 |
| `../data/echoxflow/source/` | 官方完整目录、五折划分、许可证 |
| `../data/echoxflow/legacy_provenance/` | 旧实验实际使用的样本清单和划分 |
| `../data/youtubevos2019/` | YouTube-VOS 2019 原始训练、验证图像、掩膜和元数据 |
| `../data/natural/DAVIS/` | DAVIS 2017 原始图像、掩膜、划分和说明 |
| `../data/benchmarks/` | 官方 TAP-Vid DAVIS 和 RGB-Stacking 数据 |
| `m2p_lab/` | 模型、教师匹配、三项损失、数据采样、训练及评估实现 |
| `configs/` | 复现配置和实现假设，作为后续方案的起点 |
| `scripts/` | 原始数据下载、旧 ED/ES 提取算法及依赖资源准备 |
| `tests/` | 复现实现的测试源码 |
| `vendor/` | 固定版本的 DINOv2、EchoXFlow 几何/扫描转换/轮廓源码及许可证 |
| `assets/` | 官方 DINOv2-B/14 初始化权重、官方 TAP-Vid 指标源码及许可证 |
| `requirements-train.txt` | Python 3.11 训练环境的主要依赖版本 |
| `requirements-data.txt` | 原数据处理环境的依赖版本，使用 Python 3.13；与训练环境分开安装 |

`assets/dinov2_vitb14_pretrain.pth` 是官方初始化模型；实验训练得到的适配权重位于 `runs/`。

## 原始数据说明

EchoXFlow 固定来源为 `Ahus-AIM/EchoXFlow@d907c536cae16750b9e46f269a27befd4979ebdd`。本地保留完整左心室三视图标注及关联视频，数据结构和读取字段见 [数据说明](/root/ultrasound/data/echoxflow/README_zh.md)。`raw/croissant.json` 提供记录和数组关联，`raw/splits.csv` 保留官方折号。模型专用缓存需按实验配置另行生成。

YouTube-VOS 来源镜像为 `yinloonga/YouTube_VOS_2019@7a0eb80c729b0dbbceaea15950e86ed606ab21ed`。DINOv2 源码版本为 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；TAP-Vid 指标源码版本为 `730cda1c730877cfedbe01bf87fb1cadb78a565d`。

## 代码使用边界

模型接口接收视频和查询点 `(t,x,y)`，输出轨迹 `(x,y)` 与未校准匹配分数。训练通过掩膜采点，并使用局部结构、掩膜标签、掩膜边界三项约束；预测网络不接收目标帧掩膜。

GPU 隔离配置仍固定为物理 GPU 1（UUID `GPU-3d4c8194-7991-0a63-c517-075c83c0b393`，PCI `0000:07:00.0`）。进程内的逻辑 `cuda:0` 对应这张 GPU。

`.venv-data` 用于 CPU 数据处理，`.venv-train` 用于模型训练。EchoXFlow 原始数据下载入口为 `scripts/download_echoxflow_raw.py`；mask 准备入口为 `python -m m2p_lab.echo_masks --config ...`，输入前提见上文。`download_echoxflow_edes.py` 保留旧实验的时相估计与提取实现，其主入口仍依赖旧目录，不作为新原始库的下载或重建入口。

自然视频读取与 TAP-Vid 查找使用同级共享 `../data/`。`scripts/prepare_m2p_assets.py` 将自然视频和基准数据写入该共享目录，模型权重及固定源码仍分别位于 `assets/`、`vendor/`；解压遇到已有文件会停止。现有数据已在共享目录时，无需重新下载或解压。

新的实验应明确采样、划分、监督来源和评测口径。EchoXFlow 软件轮廓同索引一致性不能代替独立材料点轨迹准确率。

## 重建训练环境

在可用的 Python 3.11 环境中创建独立环境，并安装指定 CUDA 版本的 PyTorch，再安装其余依赖。下面的命令只安装环境，不启动训练：

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv-train/bin/python -m pip install -r requirements-train.txt
```

不应复用已删除实验的检查状态或结果来声称新实验已通过验证。
