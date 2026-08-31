"""Turn Trainer loss logs into a CSV and a publication-ready PNG curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config.settings import SETTINGS


def run(input_path: Path, csv_path: Path, figure_path: Path, summary_path: Path) -> None:
    import matplotlib.pyplot as plt
    raw=json.loads(input_path.read_text(encoding="utf-8"))
    points=[row for row in raw if "loss" in row and "step" in row]
    csv_path.parent.mkdir(parents=True,exist_ok=True); figure_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["step","loss"]);w.writeheader();w.writerows(points)
    if not points: raise ValueError("No logged loss points found in the callback history.")
    plt.figure(figsize=(8,5));plt.plot([x["step"] for x in points],[x["loss"] for x in points],marker="o",linewidth=1.8)
    plt.xlabel("Training step");plt.ylabel("Loss");plt.title("CPT Training Loss — Banking Compliance")
    plt.grid(alpha=.25);plt.tight_layout();plt.savefig(figure_path,dpi=180);plt.close()
    initial=points[0]["loss"]; final=points[-1]["loss"]
    # A simple auditable plateau indicator: first step after which the next
    # three logged losses move by less than 1% of the initial loss.
    plateau=None
    for i in range(len(points)-3):
        window=[p["loss"] for p in points[i:i+4]]
        if max(window)-min(window) <= max(abs(initial)*.01,1e-8): plateau=points[i]["step"];break
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    summary_path.write_text(json.dumps({"initial_loss":initial,"final_loss":final,"loss_change_percent":(final-initial)/initial*100,"plateau_step":plateau},indent=2),encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,default=SETTINGS.results_dir/"perplexity"/"loss_history.json");p.add_argument("--csv",type=Path,default=SETTINGS.reports_dir/"tables"/"loss_history.csv");p.add_argument("--figure",type=Path,default=SETTINGS.reports_dir/"figures"/"loss_curve.png");p.add_argument("--summary",type=Path,default=SETTINGS.reports_dir/"tables"/"loss_summary.json")
    a=p.parse_args();run(a.input,a.csv,a.figure,a.summary)
