"""Recalculate the complete comparison from persisted predictions; CPU only."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from . import runtime as rt
from .echo_evaluation import ECHO, WINDOWS, metrics, sample, write_csv
from analyze_echoxflow_sequences import aggregate, configure_plots
from analyze_finetuning import paired_change

LABELS = dict(static_points="固定 ED 点", dinov2="原始 DINOv2", m2p="M2P mask 弱监督",
              echotracker2_before="EchoTracker2 原始权重", echotracker2_after="EchoTracker2 ED/ES 点监督后训练")
WINDOW_LABELS = dict(full_cycle="完整标注周期", ed_es_sequence="连续 ED→ES", two_frames="ED/ES 两帧")


def prediction_path(label, window, sid):
    if label.startswith("echotracker2_"):
        return ECHO / "outputs/echoxflow_ed_es_finetune/evaluation" / label.rsplit("_", 1)[1] / window / "predictions" / (sid + ".npz")
    return rt.RUN / "evaluation" / label / window / "predictions" / (sid + ".npz")


def recompute(cfg, records):
    rows, checks, frame_rows = [], [], []
    for window in WINDOWS:
        for label in ("dinov2", "m2p", "echotracker2_before", "echotracker2_after"):
            folder = prediction_path(label, window, records[0]["sample_id"]).parent
            if {p.stem for p in folder.glob("*.npz")} != {r["sample_id"] for r in records}:
                raise ValueError(f"Incomplete or contaminated predictions: {folder}")
            old_rows = None
            if label.startswith("echotracker2_"):
                with (folder.parent / "per_sample.csv").open() as f:
                    old_rows = {r["sample_id"]: r for r in csv.DictReader(f) if r["model"] == "echotracker2"}
            for item in records:
                images, ref, valid, meta, ids = sample(cfg, item, window)
                path = prediction_path(label, window, item["sample_id"])
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(ref, z["reference_xy_normalized"])
                    np.testing.assert_array_equal(valid, z["reference_valid"])
                    np.testing.assert_array_equal(ids, z["source_point_indices"])
                    q = meta["ed_frame"]
                    query = np.column_stack((np.full(len(ref), q), ref[:, q])).astype(np.float32)
                    np.testing.assert_array_equal(query, z["query_txy"])
                    pred = z["prediction_xy_normalized"].astype(np.float64)
                    if old_rows is None:
                        timing = json.loads(str(z["timing_json"]))
                    else:
                        timing = {k: float(old_rows[item["sample_id"]][k]) for k in ("inference_seconds", "peak_gpu_memory_gib")}
                row = metrics(pred, ref, valid, meta, label, timing)
                if old_rows is not None:
                    for k, v in row.items():
                        if k.endswith(("_px", "_mm", "_pct")) and k in old_rows[item["sample_id"]]:
                            np.testing.assert_allclose(v, float(old_rows[item["sample_id"]][k]), atol=1e-12, rtol=0)
                rows.append(row)
                checks.append(dict(model=label, window=window, sample_id=item["sample_id"], sha256=rt.file_sha(path),
                    source_sha256=item["sha256"], reference_and_query_exact=True))
                if label == "dinov2":
                    static = np.broadcast_to(ref[:, q:q+1], ref.shape).copy()
                    rows.append(metrics(static, ref, valid, meta, "static_points"))
                if window == "full_cycle":
                    predictions = [(label, pred)] + ([("static_points", static)] if label == "dinov2" else [])
                    for name, points in predictions:
                        distances = np.linalg.norm((points-ref)*256, axis=-1)
                        for frame in range(len(images)):
                            active = valid[:, frame] & (frame != q)
                            frame_rows.append(dict(sample_id=item["sample_id"], model=name, frame=frame,
                                original_frame_index=meta["original_frame_indices"][frame],
                                timestamp_seconds=meta["image_timestamps_seconds"][frame],
                                valid_points=int(active.sum()),
                                mean_error_px=float(distances[active, frame].mean()) if active.any() else None))
    return rows, checks, frame_rows


def curves(cfg, frame_rows, visuals):
    plt = configure_plots()
    with (rt.RUN / "training/training.csv").open() as f:
        train = list(csv.DictReader(f))
    vals = [json.loads(p.read_text()) for p in sorted((rt.RUN / "training/validation").glob("*.json"))]
    init = json.loads((rt.RUN / "training/initial_validation.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = np.array([int(r["step"]) for r in train])
    loss = np.array([float(r["total"]) for r in train])
    axes[0].plot(steps[99::100], np.convolve(loss, np.ones(100)/100, mode="valid")[::100])
    axes[0].set(xlabel="优化器更新次数", ylabel="弱监督总损失（100 步均值）")
    axes[1].plot([0]+[v["step"] for v in vals], [init["es_l2_epe_px"]]+[v["es_l2_epe_px"] for v in vals], "o-")
    axes[1].set(xlabel="优化器更新次数", ylabel="验证 ES EPE（256 尺度像素）")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(visuals / "training_curve.png", dpi=160); plt.close(fig)
    return vals


def video_and_curve(cfg, item, reason, frame_rows, visuals):
    import cv2
    plt = configure_plots()
    images, ref, valid, meta, _ = sample(cfg, item)
    labels = ("dinov2", "m2p", "echotracker2_after")
    predictions = {}
    for name in labels:
        with np.load(prediction_path(name, "full_cycle", item["sample_id"])) as z:
            predictions[name] = z["prediction_xy_normalized"]
    path = visuals / (reason + "_" + item["sample_id"] + ".mp4")
    fps = min(60., 1/float(np.median(np.diff(meta["image_timestamps_seconds"]))))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1008, 390))
    if not writer.isOpened():
        raise RuntimeError("Cannot encode trajectory video")
    for frame in range(len(images)):
        panels = []
        for name in labels:
            panel = np.zeros((390, 336, 3), np.uint8)
            panel[40:376] = cv2.cvtColor(images[frame], cv2.COLOR_GRAY2BGR)
            for i in range(len(ref)):
                if valid[i, frame]:
                    xy = tuple(np.rint(ref[i, frame]*335).astype(int) + [0, 40])
                    cv2.circle(panel, xy, 2, (0, 220, 0), 1)
                xy = tuple(np.rint(predictions[name][i, frame]*335).astype(int) + [0, 40])
                cv2.circle(panel, xy, 2, (20, 80, 255), -1)
            cv2.putText(panel, name, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
            phase = " ED" if frame == meta["ed_frame"] else " ES" if frame == meta["es_frame"] else ""
            cv2.putText(panel, f"frame {frame}{phase}; green=reference", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, .4, (230,230,230), 1)
            panels.append(panel)
        writer.write(np.concatenate(panels, axis=1))
    writer.release()
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        import imageio_ffmpeg
        encoder = imageio_ffmpeg.get_ffmpeg_exe()
    if encoder:
        temp = path.with_name(path.stem+"_h264.mp4")
        subprocess.run([encoder, "-v", "error", "-y", "-i", str(path), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp)], check=True)
        temp.replace(path)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label in LABELS:
        rows = [r for r in frame_rows if r["sample_id"] == item["sample_id"] and r["model"] == label]
        ax.plot([r["frame"] for r in rows], [r["mean_error_px"] for r in rows], label=LABELS[label])
    for f, label in ((meta["ed_frame"], "ED"), (meta["es_frame"], "ES")):
        ax.axvline(f, linestyle=":", color="#777", label=label)
    ax.set(xlabel="周期内原始连续帧", ylabel="平均位置误差（256 尺度像素）", title=item["sample_id"])
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=140); plt.close(fig)
    return dict(sample_id=item["sample_id"], reason=reason, path=str(path))


def table(group, norm="l2"):
    lines = ["| 模型 | P@1 | P@2 | P@4 | δavg | MTE px | 平均误差 mm | ES EPE px |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        v = group[label]["video_weighted"]
        keys = [f"{norm}_p{t}_pct" for t in (1,2,4)] + [f"{norm}_delta_avg_pct", f"{norm}_mte_px", f"{norm}_mean_error_mm", f"es_{norm}_epe_px"]
        lines.append("| " + " | ".join([LABELS[label]] + [f"{v[k]:.3f}" for k in keys]) + " |")
    return "\n".join(lines)


def report(cfg):
    records = json.loads(Path(cfg["split_manifest"]).read_text())["groups"]["test"]
    rows, checks, frame_rows = recompute(cfg, records)
    summary = {w: aggregate([r for r in rows if r["window"] == w], "model") for w in WINDOWS}
    comparisons = {}
    for window in WINDOWS:
        comparisons[window] = {}
        for status in ("all", "accepted", "review"):
            group = [r for r in rows if r["window"] == window and (status == "all" or r["status"] == status)]
            comparisons[window][status] = {}
            for baseline in ("dinov2", "static_points", "echotracker2_before", "echotracker2_after"):
                paired = [dict(r, condition="after" if r["model"] == "m2p" else "before") for r in group if r["model"] in ("m2p", baseline)]
                keys = ["es_l2_epe_px", "es_l2_mean_error_mm", "es_l2_delta_avg_pct"]
                if window == "full_cycle":
                    keys += ["l2_mte_px", "l2_mean_error_mm", "l2_delta_avg_pct", "other_frames_l2_mte_px"]
                comparisons[window][status]["m2p_vs_"+baseline] = {
                    key: paired_change(paired, key, cfg["bootstrap_resamples"], cfg["bootstrap_seed"]) for key in keys}
    rt.atomic_json(rt.RUN / "summary.json", summary)
    rt.atomic_json(rt.RUN / "paired_changes.json", comparisons)
    rt.atomic_json(rt.RUN / "prediction_checks.json", checks)
    write_csv(rt.RUN / "per_video.csv", rows)
    write_csv(rt.RUN / "frame_errors.csv", frame_rows)
    visuals = rt.RUN / "visualizations"
    visuals.mkdir(exist_ok=True)
    vals = curves(cfg, frame_rows, visuals)
    best = min(vals, key=lambda v: (v["es_l2_epe_px"], v["step"]))
    saved = torch.load(rt.RUN / "training/best.pt", map_location="cpu", weights_only=True)
    if saved["step"] != best["step"]:
        raise ValueError("Best checkpoint does not match the prespecified validation selection")
    # For this pairwise method, common predictions must agree exactly across windows.
    for label in ("m2p", "dinov2"):
        for item in records:
            with np.load(prediction_path(label, "full_cycle", item["sample_id"])) as z:
                full = z["prediction_xy_normalized"]
            for window in WINDOWS[1:]:
                with np.load(prediction_path(label, window, item["sample_id"])) as z:
                    meta = json.loads(str(z["metadata_json"]))
                    np.testing.assert_array_equal(z["prediction_xy_normalized"], full[:, meta["selected_cycle_frames"]])
    chosen = sorted(records, key=lambda r: (r["frames"], r["sample_id"]))
    picks = [(chosen[0], "short"), (chosen[len(chosen)//2], "middle"), (chosen[-1], "long")]
    worst = sorted([r for r in rows if r["model"] == "m2p" and r["window"] == "full_cycle"], key=lambda r:r["l2_mte_px"], reverse=True)[:2]
    lookup = {r["sample_id"]: r for r in records}
    picks += [(lookup[r["sample_id"]], "worst") for r in worst if r["sample_id"] not in {i["sample_id"] for i,_ in picks}]
    videos = [video_and_curve(cfg, item, reason, frame_rows, visuals) for item, reason in picks]
    train = json.loads((rt.RUN / "training/summary.json").read_text())
    prep = json.loads((rt.RUN / "preparation/manifest.json").read_text())
    audit = dict(complete=train["complete"] and train["steps"] == cfg["echo_steps"],
        optimizer_updates=train["steps"], sampled_pairs=train["sampled_pairs"], validation_checks=len(vals),
        best_step=best["step"], best_validation_es_l2_epe_px=best["es_l2_epe_px"],
        best_checkpoint_sha256=rt.file_sha(rt.RUN / "training/best.pt"),
        latest_checkpoint_sha256=rt.file_sha(rt.RUN / "training/latest.pt"),
        training_videos=prep["training_videos"], eligible_training_mask_frames=prep["eligible_frames"],
        test_videos=len(records), test_examinations=len({r["exam_id"] for r in records}),
        new_predictions=6*len(records), reused_echotracker2_predictions=6*len(records),
        metrics_recomputed_from_predictions=True, common_window_predictions_identical=True,
        failed_test_cases=[], videos=videos, software_reference_only=True)
    rt.atomic_json(rt.RUN / "completion_audit.json", audit)
    main = summary["full_cycle"]["status/accepted"]
    diff = comparisons["full_cycle"]["accepted"]["m2p_vs_dinov2"]
    intervals = []
    for key, title in (("l2_mte_px", "全周期 MTE"), ("es_l2_epe_px", "ES EPE"), ("l2_delta_avg_pct", "δavg")):
        v = diff[key]
        intervals.append(f"- 相对原始 DINOv2，{title}改善 {v['mean_improvement']:+.3f}；95% 检查级配对区间 [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]。正值为改善，负值为变差。")
    endpoint = ["| 模型 | 完整周期 ES EPE | 连续 ED→ES EPE | 两帧 EPE |", "|---|---:|---:|---:|"]
    for label in LABELS:
        endpoint.append("| " + " | ".join([LABELS[label]]+[f"{summary[w]['status/accepted'][label]['video_weighted']['es_l2_epe_px']:.3f}" for w in WINDOWS])+" |")
    body = f"""# M2P 全周期 mask 弱监督实验：EchoXFlow 软件参考一致性

本轮完成 **{train['steps']:,} 次更新、{train['sampled_pairs']:,} 对训练帧**，物理 GPU 1。使用固定的 119 段训练、46 段验证、64 段测试视频；测试集 48 段 accepted 为主分析，16 段 review 单列。最佳检查点为第 **{best['step']:,} 步**，验证 ES L2 EPE **{best['es_l2_epe_px']:.3f} px**。完整训练耗时 {train['seconds']/3600:.2f} 小时。

## 主结果：48 段 accepted，全周期，L2

{table(main)}

{chr(10).join(intervals)}

以上是 EchoXFlow 导出的 EchoPAC 软件轮廓在固定索引对应假设下的参考一致性，不是人工验证的心肌材料点轨迹精度。不根据本轮测试结果调整训练次数、参数或检查点。

## 同一 ES 参考上的输入长度对照

{chr(10).join(endpoint)}

M2P 当前实现逐帧将 ED 查询特征与目标帧匹配，没有时序信息交互；三个窗口在共同帧上的预测已逐值核对完全一致。完整周期输入可输出全部连续帧轨迹，但额外输入中间帧不会改变这套实现的 ES 预测。EchoTracker2 的连续时序模型与此不同。

## 弱监督定义与实际训练

119 段训练视频共有 **{prep['eligible_frames']:,} 帧有效心肌 mask**，覆盖完整标注周期。均匀抽取视频，再从该视频全部有效 mask 帧中均匀抽两帧，两个时间方向均可。弱监督指训练梯度不使用跨帧点对应；不是只使用 ED/ES 两张 mask。本轮不等价于“仅两张 mask 能否学会追踪”的实验。

心肌 mask 使用固定版本 EchoXFlow 栅格化代码和已有 336×336 物理视野生成；291 段全部缓存哈希、连续帧索引和 ED/ES 图像已核验，训练 ED/ES mask 与既有缓存完全一致。未匹配或小于 144 前景像素的帧不参与训练抽样，测试连续图像不删除。训练读取器只加载 images 和 masks，不读取参考轨迹。

本地 M2P 是论文独立实现，不是作者发布模型。以原始 DINOv2-B/14 初始化，LoRA rank/alpha 64、dropout 0.1，训练 LoRA 和 16 通道预测头，冻结骨干参数；冻结初始 DINOv2 作为教师。12×12 查询点，top-3 教师对应，LSC/MLC/MBC 权重 0.02/0.5/10。AdamW 学习率 2e-5、weight decay 1e-7，第 15,000 步学习率乘 0.1。微批次 8、累积 2 次、有效批次 16；BF16 特征与 FP32 坐标/损失。两帧共享仿射变换，独立亮度/对比度增强。未做自然视频 M2P 预训练或测试时适配。

每 1,000 步在完整验证视频上计算 ES 点 EPE，按视频等权最小值选择检查点；第 0 步只记录，不参与选择。因此训练监督只用 mask，但模型选择使用验证点标签。保存 best.pt，以及含优化器、调度器、随机状态、抽样位置的 latest.pt。

## 指标、统计和比较边界

ED 实际帧作为查询，每组五个轮廓点取第三个，沿用 EchoTracker2 的相同输入图像、点集合、参考有效性和坐标。336 输入像素坐标除以 335 后，以归一化位移乘 256 计像素误差，乘物理视野计毫米误差。排除 ED 查询帧；越界预测不裁剪、不剔除。P@1/2/4 为距离严格小于阈值的百分比；δavg 为三者均值；MTE 为每点跨有效时间误差中位数后平均点、再平均视频；ES EPE 为 ES 端点平均距离。主要 L2 沿用官方代码，补充 L1 贴合论文文字定义；不沿用代码默认五阈值平均，不报告无独立遮挡标注的 Jaccard。

逐视频和检查等权结果均在 summary.json；paired_changes.json 使用检查为簇、2,000 次配对重采样，seed 20261010。区间反映测试检查抽样不确定性，不涵盖训练随机性、参考误差；探索性比较未多重校正。完整原始预测和逐帧误差可重新计算汇总。

划分按来源检查隔离，未核验患者级隔离；测试子集此前已用于冻结模型实验。M2P 使用完整周期 mask，EchoTracker2 后训练使用完整视频但仅 ED/ES 点损失；标注范围、监督形式、预训练和架构不同，不能把分数差异直接归因于 mask 与点监督。

推理时间为预热后同步 CUDA 的原生推理，含模型内传输，不含磁盘读图与输入张量整理。M2P/DINOv2 均 chunk_size=1，全部原始连续帧保留。EchoTracker2 时间来自既有记录，后训练评测使用 GPU 0，与本轮 GPU 1 的时间不作为严格速度对照。DINOv2 原始特征基线绕过预测头，LoRA 初始残差为零，仍含包装计算开销。

## 全部 64 段与 review 16 段

全部 64 段：

{table(summary['full_cycle']['all'])}

review 16 段：

{table(summary['full_cycle']['status/review'])}

## L1 补充：48 段 accepted

{table(main, 'l1')}

## 文件和复现

- configs/echoxflow_mask_pairs.json：固定协议；provenance.json 与 source_snapshot/：代码与输入指纹。
- training/best.pt、training/latest.pt：最佳模型和完整恢复状态；training/validation/：全部验证记录。
- per_video.csv、frame_errors.csv、summary.json、paired_changes.json：指标与配对区间。
- evaluation/：两个新模型×三个窗口×64 段完整预测；prediction_checks.json 核对历史 EchoTracker2 参考和查询一致。
- visualizations/：训练曲线、按长度预选的短/中/长测试视频和 M2P MTE 最高样本的视频与误差曲线。
- completion_audit.json：完成审计；commands.log：实际执行命令。

训练和完整评测：`.venv-train/bin/python -u -m m2p_lab.pipeline --config configs/echoxflow_mask_pairs.json --stage all`。意外中断后追加 `--resume`。仅重新汇总已保存预测：将 stage 改为 report。

论文依据：M2P arXiv:2603.17813 §III-B/C 的 mask 弱监督与损失；EchoTracker2 arXiv:2605.12140 §3.2 的指标定义及本地固定官方 utils/evaluate.py；本实验不是论文原私有数据结果复现。
"""
    (rt.RUN / "REPORT_zh.md").write_text(body)
    rt.guard_disk(cfg)
    rt.event("report_complete", best_step=best["step"], test_videos=len(records), path=str(rt.RUN / "REPORT_zh.md"))
