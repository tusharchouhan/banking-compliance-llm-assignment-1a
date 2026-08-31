"""Compute base and CPT perplexity on the untouched 10% document split."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, resolve_device
from src.utils.modeling import load_causal_lm, load_tokenizer


def corpus_ids(eval_dir: Path, tokenizer) -> list[int]:
    ids=[]
    for path in sorted(eval_dir.glob("*.txt")):
        ids.extend([tokenizer.bos_token_id] + tokenizer(path.read_text(encoding="utf-8",errors="replace"),add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id])
    return ids


def perplexity(model, ids: list[int], seq_len: int = 1024, stride: int = 1024) -> float:
    import torch
    if len(ids) < 2: raise ValueError("Evaluation corpus contains fewer than two tokens.")
    device=next(model.parameters()).device; losses=[]; total_tokens=0
    for start in range(0,len(ids)-1,stride):
        end=min(start+seq_len,len(ids)); chunk=ids[start:end]
        if len(chunk)<2: break
        input_ids=torch.tensor([chunk],dtype=torch.long,device=device)
        with torch.no_grad(): out=model(input_ids=input_ids,labels=input_ids)
        token_count=len(chunk)-1; losses.append(float(out.loss)*token_count); total_tokens+=token_count
        if end==len(ids): break
    return math.exp(sum(losses)/max(total_tokens,1))


def run(base_model: str, cpt_model: str, eval_dir: Path, output: Path, model_id_for_tokenizer: str, seq_len: int, logger) -> None:
    tokenizer=load_tokenizer(model_id_for_tokenizer); ids=corpus_ids(eval_dir,tokenizer)
    base=load_causal_lm(base_model); cpt=load_causal_lm(cpt_model)
    base_ppl=perplexity(base,ids,seq_len); cpt_ppl=perplexity(cpt,ids,seq_len); improvement=(base_ppl-cpt_ppl)/base_ppl*100
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["base_model","cpt_model","evaluation_documents","base_ppl","cpt_ppl","percentage_improvement"]);w.writeheader();w.writerow({"base_model":base_model,"cpt_model":cpt_model,"evaluation_documents":len(list(eval_dir.glob('*.txt'))),"base_ppl":round(base_ppl,4),"cpt_ppl":round(cpt_ppl,4),"percentage_improvement":round(improvement,2)})
    logger.info("Base PPL %.3f | CPT PPL %.3f | improvement %.2f%%",base_ppl,cpt_ppl,improvement)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-model",default=SETTINGS.model_id);p.add_argument("--cpt-model",type=Path,default=SETTINGS.models_dir/"cpt_model");p.add_argument("--eval-dir",type=Path,default=SETTINGS.eval_corpus);p.add_argument("--output",type=Path,default=SETTINGS.results_dir/"perplexity"/"ppl_results.csv");p.add_argument("--tokenizer-model",default=SETTINGS.model_id);p.add_argument("--seq-len",type=int,default=SETTINGS.max_seq_length)
    a=p.parse_args();run(a.base_model,str(a.cpt_model),a.eval_dir,a.output,a.tokenizer_model,a.seq_len,configure_logging("perplexity"))

