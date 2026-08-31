"""Compare general-knowledge generations before and after CPT."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, set_seed
from src.utils.modeling import generate, load_causal_lm, load_tokenizer

PROMPTS=["The capital of France is", "Water boils at", "The speed of light is approximately"]
EXPECTED=[["paris"],["100","212"],["3", "300,000"]]


def run(base_model: str, cpt_model: str, output_dir: Path, tokenizer_model: str, seed: int, logger) -> None:
    set_seed(seed);tok=load_tokenizer(tokenizer_model);base=load_causal_lm(base_model);cpt=load_causal_lm(cpt_model);rows=[]
    for index, prompt in enumerate(PROMPTS):
        b=generate(base,tok,prompt,max_new_tokens=30,do_sample=False);c=generate(cpt,tok,prompt,max_new_tokens=30,do_sample=False)
        base_has=any(term in b.lower() for term in EXPECTED[index]); cpt_has=any(term in c.lower() for term in EXPECTED[index])
        verdict="Retained" if cpt_has or not base_has else "Degraded"
        rows.append({"prompt":prompt,"base_output":b,"cpt_output":c,"verdict":verdict})
    output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/"forgetting_comparison.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (output_dir/"forgetting_comparison.md").open("w",encoding="utf-8") as h:
        h.write("| Prompt | Base model | CPT model | Verdict |\n|---|---|---|---|\n")
        for row in rows: h.write("| {prompt} | {base_output} | {cpt_output} | {verdict} |\n".format(**{k:v.replace('|','\\|').replace('\n',' ') for k,v in row.items()}))
    logger.info("Saved catastrophic forgetting comparison to %s",output_dir)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-model",default=SETTINGS.model_id);p.add_argument("--cpt-model",type=Path,default=SETTINGS.models_dir/"cpt_model");p.add_argument("--output-dir",type=Path,default=SETTINGS.results_dir/"forgetting_check");p.add_argument("--tokenizer-model",default=SETTINGS.model_id);p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args();run(a.base_model,str(a.cpt_model),a.output_dir,a.tokenizer_model,a.seed,configure_logging("forgetting_check"))
