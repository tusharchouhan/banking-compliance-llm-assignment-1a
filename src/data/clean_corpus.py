"""Apply the required length, repetition, deduplication, and English filters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def is_repetitive(text: str, threshold: float = 0.30) -> bool:
    parts = paragraphs(text)
    if not parts:
        return False
    counts: dict[str, int] = {}
    for part in parts:
        counts[part] = counts.get(part, 0) + 1
    duplicate_paragraphs = sum(count for count in counts.values() if count > 1)
    return duplicate_paragraphs / len(parts) > threshold


def is_english(text: str) -> bool:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text[:10000]) == "en"
    except Exception:
        # Conservative fallback for environments where langdetect cannot identify short text.
        letters = [c for c in text if c.isalpha()]
        ascii_ratio = sum(c.isascii() for c in letters) / max(len(letters), 1)
        return ascii_ratio >= 0.85


def run(input_dir: Path, output_dir: Path, report_path: Path, logger) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.rglob("*.txt"))
    stats = [{"step": "input", "documents": len(files), "removed": 0, "impact": "baseline"}]
    seen: set[str] = set()
    survivors: list[tuple[Path, str]] = []
    removed_length = removed_repeat = removed_duplicate = removed_language = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 50:
            removed_length += 1
            continue
        if is_repetitive(text):
            removed_repeat += 1
            continue
        digest = hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()
        if digest in seen:
            removed_duplicate += 1
            continue
        seen.add(digest)
        if not is_english(text):
            removed_language += 1
            continue
        survivors.append((path, text))
    counts = [len(files), len(files) - removed_length, len(files) - removed_length - removed_repeat,
              len(files) - removed_length - removed_repeat - removed_duplicate, len(survivors)]
    removed = [0, removed_length, removed_repeat, removed_duplicate, removed_language]
    names = ["input", "length_filter", "repetition_filter", "deduplication", "english_filter"]
    for name, count, drop in zip(names, counts, removed):
        stats.append({"step": name, "documents": count, "removed": drop,
                      "impact": "greatest" if drop == max(removed[1:]) and drop > 0 else ""})
    for path, text in survivors:
        relative_path = path.relative_to(input_dir)
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "documents", "removed", "impact"])
        writer.writeheader(); writer.writerows(stats)
    logger.info("Cleaning complete: %d -> %d documents", len(files), len(survivors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SETTINGS.extracted_txt)
    parser.add_argument("--output-dir", type=Path, default=SETTINGS.cleaned_txt)
    parser.add_argument("--report", type=Path, default=SETTINGS.reports_dir / "tables" / "cleaning_report.csv")
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.report, configure_logging("clean_corpus"))
