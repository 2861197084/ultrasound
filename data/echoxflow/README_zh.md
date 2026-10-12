# EchoXFlow 左心室三视图原始数据

原始数据根目录：`/root/ultrasound/data/echoxflow/raw`。
固定官方版本：`d907c536cae16750b9e46f269a27befd4979ebdd`。

本地包含该版本全部左心室标注及关联视频：336 个检查、446 个标注记录、877 个视频记录，共 1,323 个 Zarr、13,085 个数组。原始 Zarr 文件合计 15.692 GB（14.614 GiB），保留完整原始视频帧、轮廓层、时间戳、ECG 和几何信息。

## 目录

```text
echoxflow/
  raw/
    exams/<exam_id>/<recording_id>.zarr/  原始记录，未修改
    croissant.json                     按本地下载范围筛选的官方格式目录
    splits.csv                         对应检查的官方五折编号
    LICENSE                            官方许可证
  source/
    croissant.json                     未修改的官方完整目录
    splits.csv                         未修改的官方完整划分
    LICENSE                            官方许可证
  legacy_provenance/                    旧实验实际使用的样本清单、划分及配置依据
  README_zh.md                          本说明
```

`raw/` 可直接供官方读取工具使用。`source/` 用于确认原始来源及重新下载；`legacy_provenance/` 只用于理解历史实验，不参与新数据读取。

## 标注与视频关联

| 视图 | 原始轮廓链接 | 按视频内容去重后明确配对、非空的数量 | 待解析链接 | 空轮廓链接 |
| --- | ---: | ---: | ---: | ---: |
| A4C (`4ch`) | 446 | 291 | 77 | 0 |
| A2C (`2ch`) | 446 | 293 | 77 | 1 |
| ALAX (`3ch`) | 278 | 238 | 39 | 0 |
| 合计 | 1,170 | 822 | 193 | 1 |

这些数量不代表独立患者数。关联由 `croissant.json` 中的 `recording_links` 提供，记录信息和数组信息分别位于 `recordings`、`arrays`。重复标注全部保留。193 条待解析链接的所有关联平面均已下载，但不能假定平面编号就是视图编号；它们和空轮廓不自动进入监督训练。

## 读取

从 `raw/croissant.json` 的 `recording_links` 取得标注和视频关联，从 `raw/splits.csv` 取得官方折号，再用 `zarr.open_group(记录路径, mode="r")` 读取对应记录。视频位于关联指定的帧数组，轮廓位于 `data/4ch_contour`、`data/2ch_contour` 或 `data/3ch_contour`，各自的时间戳位于 `timestamps/`，成像几何见记录属性中的 `recording_manifest`。

轮廓和视频的帧数可能不同，训练前需按时间戳对齐。轮廓坐标单位为米，需要结合成像几何转换为图像坐标。每个沿壁位置包含 5 个壁厚方向采样点，中层索引为 `2, 7, 12, …`。这些是软件追踪产生的参考坐标；明确配对不等于逐点组织对应关系已经获得人工认证。

官方工具可设置 `ECHOXFLOW_DATA_ROOT=/root/ultrasound/data/echoxflow/raw`，项目保留的官方工具版本为 `b782e0a4e3dbaadf3c88fa572710bf4254d86143`。

## 下载与校验

```bash
# 查看下载范围；结果输出到终端。
/root/ultrasound/m2p/.venv-data/bin/python -B /root/ultrasound/m2p/scripts/download_echoxflow_raw.py plan
# 补齐缺失或损坏的记录。
/root/ultrasound/m2p/.venv-data/bin/python -B /root/ultrasound/m2p/scripts/download_echoxflow_raw.py fetch --workers 4
# 解码全部数组，检查官方形状、类型、内容校验和及划分。
/root/ultrasound/m2p/.venv-data/bin/python -B /root/ultrasound/m2p/scripts/download_echoxflow_raw.py verify --workers 4
```

下载器直接使用 `source/` 的官方元数据；已存在的数据通过官方数组校验后复用。缺失记录通过 HTTP Range 获取全部原始成员，TAR 索引仅在内存中使用。默认 4 个并发，保留至少 35 GiB 空闲空间，不下载整检查 TAR。下载时暂存到 `.staging/`，校验成功后提交，空暂存目录自动移除。进度和校验结果仅输出到终端。

来源：[官方数据](https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/tree/d907c536cae16750b9e46f269a27befd4979ebdd) · [官方工具](https://github.com/Ahus-AIM/EchoXFlow/tree/b782e0a4e3dbaadf3c88fa572710bf4254d86143)。
