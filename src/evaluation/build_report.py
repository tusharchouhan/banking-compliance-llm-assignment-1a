"""Assemble the generated experiment artifacts into a concise final report."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import SETTINGS


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else "Not generated yet."


def run(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True,exist_ok=True)
    cleaning=read(root/"reports/tables/cleaning_report.csv");ppl=read(root/"results/perplexity/ppl_results.csv");loss=read(root/"reports/tables/loss_summary.json");forgetting=read(root/"results/forgetting_check/forgetting_comparison.md")
    output.write_text(f"""# Assignment 1A Results — Banking Regulatory Compliance

## Experiment configuration

- Domain: Enterprise Variant V3 — Banking / FinTech / Insurance regulatory compliance
- Model: HuggingFaceTB/SmolLM2-360M
- CPT split: 90% train / 10% held-out evaluation
- QLoRA adapters: A (r=8, alpha=16), B (r=16, alpha=32), C (r=32, alpha=32)
- Variant requirement: deterministic greedy decoding for compliance demonstrations

## Cleaning report

```csv
{cleaning}
```

## CPT loss summary

```json
{loss}
```

## Domain perplexity

```csv
{ppl}
```

## Catastrophic forgetting comparison

{forgetting}

## Adapter comparison

See `results/adapter_comparison/adapter_comparison.csv`. The table includes outputs for all three adapters and a transparent keyword-grounding proxy; review responses against the source corpus before making a final accuracy claim.

## Limitations

All metrics are data-dependent and must be regenerated after the student places the final public PDF corpus in `data/raw_pdfs/`. Regulatory answers require verification against current primary sources and are not legal or compliance advice.
""",encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=SETTINGS.project_root);p.add_argument("--output",type=Path,default=SETTINGS.reports_dir/"final_report"/"results_summary.md");a=p.parse_args();run(a.root,a.output)
