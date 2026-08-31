"""Generate deterministic no-system-prompt outputs from the pretrained base model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, set_seed
from src.utils.modeling import generate, load_causal_lm, load_tokenizer

PROMPTS = ["What is KYC?", "What is LTV Ratio?", "What is Basel III?"]


def run(model_id: str, output_dir: Path, max_new_tokens: int, seed: int, logger) -> None:
    set_seed(seed); tokenizer=load_tokenizer(model_id); model=load_causal_lm(model_id)
    output_dir.mkdir(parents=True, exist_ok=True); rows=[]
    for index, prompt in enumerate(PROMPTS, 1):
        answer=generate(model,tokenizer,prompt,max_new_tokens=max_new_tokens,do_sample=False)
        (output_dir/f"prompt_{index}.txt").write_text(f"Prompt: {prompt}\n\n{answer}\n",encoding="utf-8")
        rows.append({"prompt":prompt,"response":answer}); logger.info("Generated baseline for %s",prompt)
    with (output_dir/"baseline_outputs.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["prompt","response"]);w.writeheader();w.writerows(rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=SETTINGS.model_id);p.add_argument("--output-dir",type=Path,default=SETTINGS.results_dir/"baseline_outputs");p.add_argument("--max-new-tokens",type=int,default=100);p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args();run(a.model_id,a.output_dir,a.max_new_tokens,a.seed,configure_logging("baseline_inference"))

