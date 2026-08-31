"""Run Assignment 1A stages in order; each stage is also independently executable."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config.settings import SETTINGS


STAGES={
    "extract":"src.data.pdf_extractor", "clean":"src.data.clean_corpus", "split":"src.data.train_eval_split",
    "pack":"src.data.tokenize_and_pack", "inspect":"src.evaluation.model_inspection", "baseline":"src.evaluation.baseline_inference",
    "cpt":"src.training.cpt_train", "loss":"src.evaluation.loss_analysis", "ppl":"src.evaluation.perplexity",
    "forgetting":"src.evaluation.forgetting_check", "instructions":"src.data.create_instruction_dataset",
    "adapter_A":"src.qlora.train_adapter_A", "adapter_B":"src.qlora.train_adapter_B", "adapter_C":"src.qlora.train_adapter_C",
    "compare":"src.evaluation.adapter_comparison", "report":"src.evaluation.build_report",
}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stages",nargs="+",choices=["all",*STAGES],default=["all"]);p.add_argument("--max-steps",type=int,default=SETTINGS.max_steps);p.add_argument("--model-id",default=SETTINGS.model_id);a=p.parse_args()
    selected=list(STAGES) if "all" in a.stages else a.stages
    SETTINGS.ensure_directories()
    for stage in selected:
        cmd=[sys.executable,"-m",STAGES[stage]]
        if stage in {"inspect","baseline"}: cmd += ["--model-id",a.model_id]
        if stage == "pack": cmd += ["--model-id",a.model_id]
        if stage == "cpt": cmd += ["--model-id",a.model_id,"--max-steps",str(a.max_steps)]
        if stage == "ppl": cmd += ["--base-model",a.model_id,"--tokenizer-model",a.model_id]
        if stage in {"adapter_A","adapter_B","adapter_C"}: cmd += ["--max-steps",str(a.max_steps)]
        print(f"\n=== {stage} ===",flush=True);subprocess.run(cmd,cwd=SETTINGS.project_root,check=True)
    print("\nPipeline completed.")


if __name__ == "__main__": main()
