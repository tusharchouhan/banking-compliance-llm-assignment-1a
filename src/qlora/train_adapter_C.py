"""Train Adapter C: r=32, alpha=32, q_proj/v_proj/o_proj."""
import argparse
from pathlib import Path
from src.config.settings import SETTINGS
from src.qlora.train_adapter import run
from src.utils.common import configure_logging

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base-model",default=str(SETTINGS.models_dir/"cpt_model"));p.add_argument("--dataset",type=Path,default=SETTINGS.instruction_dataset/"train.jsonl");p.add_argument("--output-dir",type=Path,default=SETTINGS.models_dir/"adapter_C");p.add_argument("--max-steps",type=int,default=SETTINGS.max_steps);p.add_argument("--seed",type=int,default=SETTINGS.seed);a=p.parse_args();run("C",a.base_model,a.dataset,a.output_dir,a.max_steps,a.seed,configure_logging("train_adapter_C"))
