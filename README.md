# 超声跟踪实验项目

本仓库统一管理团队协作规则，以及 EchoTracker2、SemiEchoTracker 和 m2p 的源码、配置与文档。共享数据集、下载的模型权重、Python 环境和实验结果保存在本地，不随 Git 提交。协作要求见 [AGENTS.md](AGENTS.md)。

## 目录与版本管理

| 目录 | 管理方式与用途 |
|---|---|
| `EchoTracker2/` | 普通目录，根仓库直接跟踪；源码基于 [ptecho](https://github.com/riponazad/ptecho) 的 `e6e6965a5cd52445597b51352d2addb748fb50c4`；入口见 [README](EchoTracker2/README.md) |
| `SemiEchoTracker/` | 普通目录，根仓库直接跟踪；源码基于 [SemiEchoTracker](https://github.com/Fitz-Fitz/SemiEchoTracker) 的 `ee145d07334e784c518a787dcd41c0efd0fe4209`；入口见 [README](SemiEchoTracker/README.md) |
| `m2p/` | 根仓库直接跟踪源码、配置、测试、依赖声明和文档；运行条件见 [m2p/README.md](m2p/README.md) |
| `data/` | 共享数据；Git 仅跟踪数据说明、许可证等必要元信息，具体范围见 `.gitignore` |

三个项目均由根仓库直接跟踪。下载的 `m2p/vendor/`、模型文件、虚拟环境和运行目录由忽略规则排除。EchoTracker2 原仓库自带的演示样本、演示图和许可证，以及 m2p 使用的官方 TAP-Vid 指标源码与对应许可证随代码一起跟踪。

EchoXFlow 原始数据入口见 [data/echoxflow/README_zh.md](data/echoxflow/README_zh.md)。各实验的数据准备与划分要求见对应项目文档。

## 日常 Git 操作

首次获取项目使用 `git clone <仓库地址>`。日常操作的工作目录为 `/root/ultrasound`，三个项目的改动均在此选择文件并提交。例如一次提交两个模型项目的改动：

```bash
git status
git add EchoTracker2 SemiEchoTracker
git diff --cached
git commit -m "说明模型或实验改动"
```

修改 m2p 或根目录文档时，同样添加实际变更路径，如 `git add m2p README.md`。提交前使用 `git status` 和 `git diff --cached` 核对范围，不强制添加被忽略的数据、权重或运行产物。忽略规则见根目录 [.gitignore](.gitignore)；EchoTracker2 自带的 `.gitignore` 也在其目录内生效。

## 远程协作与复现

三个项目共用根仓库的提交历史与远程配置。根仓库目前尚未配置远程，后续应使用自己或团队的仓库地址；表中的上游地址仅说明源码来源。`git commit` 只保存本地版本，执行 `git push` 才会上传到配置的远程仓库。

Git 克隆只恢复受版本控制的材料，数据、权重和环境需要单独准备。训练前以各项目文档和实际输入为准；m2p 当前还存在派生缓存和跨项目评测依赖缺失，不能将 Git 工作区干净视为训练入口已可运行。
