from __future__ import annotations

import csv
import json
from pathlib import Path

from .runtime import RUN


def make_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = []
    for name in ("natural", "echo_direct", "echo_transfer"):
        path = RUN / name / "training.csv"
        if not path.exists():
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            stride = max(1, len(rows) // 3000)
            chosen = rows[::stride]
            curves.append((name, [int(r["step"]) for r in chosen], [float(r["total"]) for r in chosen]))
    if curves:
        fig, axes = plt.subplots(len(curves), 1, figsize=(9, 3 * len(curves)), squeeze=False)
        for ax, (name, steps, loss) in zip(axes[:, 0], curves):
            ax.plot(steps, loss, linewidth=.6, alpha=.8)
            ax.set(title=name, xlabel="Optimizer updates", ylabel="Weak supervision loss")
            ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(RUN / "training_curves.png", dpi=170); plt.close(fig)
    lines = ["# M2P reproduction and EchoXFlow transfer", "",
             "This is an independent implementation of arXiv:2603.17813v1, not an official M2P release.", "",
             "All GPU work uses physical GPU 1. Each experiment retains only its best adapter/head checkpoint; optimizer states are not saved.", "",
             "## Training", "", "| Experiment | Updates | Hours | Best validation loss |", "|---|---:|---:|---:|"]
    for name in ("natural", "echo_direct", "echo_transfer"):
        path = RUN / name / "summary.json"
        if path.exists():
            r = json.loads(path.read_text())
            lines.append(f"| {name} | {r['steps']} | {r['seconds']/3600:.2f} | {r['best_validation']:.6f} |")
    recovery_path = RUN / "natural/recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        lines += ["", f"**Interrupted-run recovery:** natural-video training stopped after logged update {recovery['previous_logged_step']:,}. "
                  f"Training continued from the retained best adapter at update {recovery['restored_step']:,} with a fresh AdamW optimizer. "
                  "Optimizer moments and RNG state were not restored. These results are not an uninterrupted 150,000-update reproduction. "
                  "Original logs are preserved under natural/segments/. Reported training hours include discarded work; curves follow the retained checkpoint's lineage."]
    lines += ["", "## TAP-Vid localization", "", "All values are percentages. First-visible query protocol, equal weight per video, 256-pixel metric coordinates.", "",
              "| Model | DAVIS delta average | RGB-Stacking delta average |", "|---|---:|---:|"]
    for name in ("dinov2_baseline", "natural_final", "natural_best"):
        path = RUN / "evaluation" / name / "summary.json"
        if path.exists():
            r = json.loads(path.read_text())
            lines.append(f"| {name} | {r['davis']['average_pts_within_thresh']:.2f} | {r['rgb_stacking']['average_pts_within_thresh']:.2f} |")
    lines += ["| Paper DINOv2-B/14 reference | 38.0 | 37.8 |", "| Paper M2Pv2-B/14 reference | 50.8 | 51.5 |", "",
              "The implementation does not reproduce a learned occlusion predictor; AJ and OA are not claimed.", "",
              "## EchoXFlow", "", "Mask retention and cycle consistency are proxy metrics. Synthetic translation error is a geometric check, not real myocardial material-point accuracy.", "",
              "| Model | Mask retention | Cycle error (px) | Synthetic translation EPE (px) |", "|---|---:|---:|---:|"]
    for name in ("dinov2_baseline", "natural_zero_shot", "echo_direct_final", "echo_direct_best", "echo_transfer_final", "echo_transfer_best"):
        path = RUN / "echo_evaluation" / name / "summary.json"
        if path.exists():
            r = json.loads(path.read_text())["summary"]
            lines.append(f"| {name} | {r['mask_retention']:.4f} | {r['cycle_error_pixels']:.2f} | {r['synthetic_translation_epe']:.2f} |")
    lines += ["", "## Reproduction assumptions", "", "See configuration.json, provenance.json, natural_manifest.json and the environment record for exact sources, validation policy and unpublished implementation defaults.", "",
              "The best checkpoint is selected using fixed validation weak-supervision loss, never the TAP-Vid test scores. Final-step and validation-selected results are reported separately.", ""]
    (RUN / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    make_report()
