"""Evaluate adapters A/B/C on the same banking prompts and produce a comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, model_dtype, set_seed
from src.utils.modeling import generate, load_tokenizer

PROMPTS=["What is KYC?","What is Basel III?","What is RBI LTV Ratio?"]
KEYWORDS={"What is KYC?":["kyc","customer","identity"],"What is Basel III?":["basel","capital","risk"],"What is RBI LTV Ratio?":["ltv","loan","value","rbi"]}


def run(base_model: str, adapter_root: Path, output_dir: Path, tokenizer_model: str, seed: int, logger) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    set_seed(seed);tok=load_tokenizer(tokenizer_model);rows=[]
    for prompt in PROMPTS:
        adapter_outputs={}
        for name in ("A","B","C"):
            kwargs={"torch_dtype":model_dtype()}
            if torch.cuda.is_available(): kwargs["device_map"]="auto"
            model=AutoModelForCausalLM.from_pretrained(base_model,**kwargs)
            model=PeftModel.from_pretrained(model,str(adapter_root/f"adapter_{name}"))
            adapter_outputs[name]=generate(model,tok,prompt,max_new_tokens=100,do_sample=False)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        scores={name:sum(word in text.lower() for word in KEYWORDS[prompt]) for name,text in adapter_outputs.items()}
        best=max(scores,key=scores.get)
        rows.append({"prompt":prompt,"adapter_A":adapter_outputs["A"],"adapter_B":adapter_outputs["B"],"adapter_C":adapter_outputs["C"],"keyword_grounding_scores":json.dumps(scores),"most_domain_relevant_by_keyword_proxy":best})
    output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/"adapter_comparison.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output_dir/"adapter_comparison.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
    logger.info("Saved adapter comparison to %s (winner uses transparent keyword-grounding proxy; review for final submission)",output_dir)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-model",default=str(SETTINGS.models_dir/"cpt_model"));p.add_argument("--adapter-root",type=Path,default=SETTINGS.models_dir);p.add_argument("--output-dir",type=Path,default=SETTINGS.results_dir/"adapter_comparison");p.add_argument("--tokenizer-model",default=str(SETTINGS.models_dir/"cpt_model"));p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args();run(a.base_model,a.adapter_root,a.output_dir,a.tokenizer_model,a.seed,configure_logging("adapter_comparison"))
