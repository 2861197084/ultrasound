# 超声跟踪实验项目

本仓库管理团队协作规则、m2p 自维护代码，以及 EchoTracker2 和 SemiEchoTracker 的上游版本引用。原始数据、模型权重、Python 环境和实验结果保存在本地，不随 Git 提交。协作要求见 [AGENTS.md](AGENTS.md)。

## 目录与版本管理

| 目录 | 管理方式与用途 |
|---|---|
| `EchoTracker2/` | Git 子模块，来源为 `https://github.com/riponazad/ptecho.git`；项目入口见其 `README.md` |
| `SemiEchoTracker/` | Git 子模块，来源为 `https://github.com/Fitz-Fitz/SemiEchoTracker.git`；项目入口见其 `README.md` |
| `m2p/` | 根仓库直接跟踪源码、配置、测试、依赖声明和文档；运行条件见 [m2p/README.md](m2p/README.md) |
| `data/` | 共享数据；Git 仅跟踪 EchoXFlow 数据说明、许可证及保留的固定划分 |

子模块的准确提交由根仓库记录，`.gitmodules` 保存其获取地址。下载的 `m2p/vendor/`、模型文件、虚拟环境和运行目录由忽略规则排除；官方 TAP-Vid 指标源码与对应许可证随 m2p 代码一起跟踪。

EchoXFlow 原始数据入口见 [data/echoxflow/README_zh.md](data/echoxflow/README_zh.md)。历史固定划分保存在 [split.json](data/echoxflow/legacy_provenance/EchoTracker2/outputs/echoxflow_ed_es_finetune/split.json)，用于辨认既有样本分组，不代表对应的派生缓存已准备好。

## 日常 Git 操作

工作目录为 `/root/ultrasound`。首次克隆根仓库时使用 `git clone --recurse-submodules <仓库地址>`；已有工作副本可初始化根仓库记录的子模块版本：

```bash
git submodule update --init --recursive
git status
git submodule status
```

修改 m2p 或根目录文档时，在根仓库中选择实际变更并提交：

```bash
git add m2p README.md
git diff --cached --stat
git commit -m "说明本次改动"
```

修改子模块时，先在对应目录的工作分支中提交，再由根仓库记录新的提交引用。例如修改 EchoTracker2：

```bash
git -C EchoTracker2 switch -c experiment/new-run
git -C EchoTracker2 add <变更文件>
git -C EchoTracker2 commit -m "说明模型或实验改动"
git add EchoTracker2
git commit -m "更新 EchoTracker2 版本引用"
```

提交前使用 `git status` 和 `git diff --cached` 核对范围，不强制添加被忽略的数据、权重或运行产物。现有子模块的本地排除规则位于各自 Git 目录的 `info/exclude`；上游源码保持原样。

## 远程协作与复现

根仓库需配置团队自己的远程地址后才能推送。子模块当前指向公开上游；共享本地修改时，应先将子模块提交推送到团队有写权限的仓库，并更新 `.gitmodules` 中的地址，再推送根仓库的版本引用。

Git 克隆只恢复受版本控制的材料，数据、权重和环境需要单独准备。训练前以各项目文档和实际输入为准；m2p 当前还存在派生缓存和跨项目评测依赖缺失，不能将 Git 工作区干净视为训练入口已可运行。
