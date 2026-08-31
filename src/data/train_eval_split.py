"""Create a deterministic 90/10 document-level train/evaluation split."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging


def run(input_dir: Path, train_dir: Path, eval_dir: Path, seed: int, report: Path, logger) -> None:
    files = sorted(input_dir.rglob("*.txt"))
    if len(files) < 2:
        raise ValueError("At least two cleaned documents are required for a 90/10 split.")
    rng = random.Random(seed); rng.shuffle(files)
    eval_count = max(1, round(len(files) * 0.10))
    eval_files, train_files = files[:eval_count], files[eval_count:]
    for directory in (train_dir, eval_dir): directory.mkdir(parents=True, exist_ok=True)
    for src in train_files:
        destination = train_dir / src.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
    for src in eval_files:
        destination = eval_dir / src.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=["split", "documents", "fraction", "seed"]); w.writeheader()
        w.writerows([{"split":"train", "documents":len(train_files), "fraction":"90% target", "seed":seed},
                     {"split":"eval", "documents":len(eval_files), "fraction":"10% target", "seed":seed}])
    logger.info("Split %d documents into %d train and %d eval files", len(files), len(train_files), len(eval_files))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",type=Path,default=SETTINGS.cleaned_txt)
    p.add_argument("--train-dir",type=Path,default=SETTINGS.train_corpus); p.add_argument("--eval-dir",type=Path,default=SETTINGS.eval_corpus)
    p.add_argument("--seed",type=int,default=SETTINGS.seed); p.add_argument("--report",type=Path,default=SETTINGS.reports_dir/"tables"/"split_report.csv")
    a=p.parse_args(); run(a.input_dir,a.train_dir,a.eval_dir,a.seed,a.report,configure_logging("train_eval_split"))
