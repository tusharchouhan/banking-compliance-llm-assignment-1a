"""Continual pre-training on packed domain text with HuggingFace Trainer."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, set_seed
from src.utils.modeling import load_causal_lm, load_tokenizer


class PackedParquetDataset:
    def __init__(self, parquet_path: Path):
        import pyarrow.parquet as pq
        self.rows = pq.read_table(parquet_path, columns=["input_ids"]).to_pylist()

    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return {"input_ids": self.rows[index]["input_ids"], "labels": self.rows[index]["input_ids"]}


try:
    from transformers import TrainerCallback
except ImportError:  # permits documentation/help commands before installation
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class LossCaptureCallback(TrainerCallback):
    """Capture each Trainer logging event for the later loss-curve report."""
    def __init__(self, path: Path): self.path = path; self.history = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            item={"step":state.global_step,"loss":float(logs["loss"])};self.history.append(item)
            self.path.parent.mkdir(parents=True,exist_ok=True);self.path.write_text(json.dumps(self.history,indent=2),encoding="utf-8")


class PackedCollator:
    def __call__(self, features):
        import torch
        return {key: torch.tensor([item[key] for item in features], dtype=torch.long) for key in ("input_ids", "labels")}


def run(model_id: str, packed_data: Path, output_dir: Path, loss_path: Path, max_steps: int, learning_rate: float, warmup_ratio: float, seed: int, logger) -> None:
    from transformers import Trainer, TrainingArguments
    set_seed(seed); tokenizer=load_tokenizer(model_id); model=load_causal_lm(model_id,training=True); dataset=PackedParquetDataset(packed_data)
    callback=LossCaptureCallback(loss_path)
    use_bf16 = _cuda_available() and _bf16_supported()
    training_kwargs={
        "output_dir":str(output_dir),
        "overwrite_output_dir":True,
        "per_device_train_batch_size":1,
        "gradient_accumulation_steps":8,
        "learning_rate":learning_rate,
        "weight_decay":0.01,
        "max_steps":max_steps,
        "warmup_ratio":warmup_ratio,
        "logging_strategy":"steps",
        "logging_steps":10,
        "save_strategy":"steps",
        "save_steps":100,
        "save_total_limit":2,
        "report_to":[],
        "remove_unused_columns":False,
        "fp16":_cuda_available() and not use_bf16,
        "bf16":use_bf16,
        "optim":"adamw_torch",
        "seed":seed,
    }
    # Transformers has renamed/removed a few TrainingArguments fields across
    # releases. Keep the assignment settings, but pass only fields supported
    # by the installed version so Colab runtime updates do not stop CPT.
    supported=set(inspect.signature(TrainingArguments.__init__).parameters)
    unsupported=sorted(set(training_kwargs)-supported)
    if unsupported:
        logger.warning("Ignoring unsupported TrainingArguments fields: %s", ", ".join(unsupported))
    args=TrainingArguments(**{key:value for key,value in training_kwargs.items() if key in supported})
    trainer=Trainer(model=model,args=args,train_dataset=dataset,data_collator=PackedCollator(),callbacks=[callback])
    trainer.train(); trainer.save_model(output_dir); tokenizer.save_pretrained(output_dir)
    logger.info("Saved CPT model and tokenizer to %s",output_dir)


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError: return False


def _bf16_supported():
    try:
        import torch
        return torch.cuda.is_bf16_supported()
    except (ImportError, AttributeError): return False


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=SETTINGS.model_id);p.add_argument("--packed-data",type=Path,default=SETTINGS.data_dir/"train_packed.parquet");p.add_argument("--output-dir",type=Path,default=SETTINGS.models_dir/"cpt_model");p.add_argument("--loss-path",type=Path,default=SETTINGS.results_dir/"perplexity"/"loss_history.json");p.add_argument("--max-steps",type=int,default=SETTINGS.max_steps);p.add_argument("--learning-rate",type=float,default=SETTINGS.learning_rate);p.add_argument("--warmup-ratio",type=float,default=SETTINGS.warmup_ratio);p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args();run(a.model_id,a.packed_data,a.output_dir,a.loss_path,a.max_steps,a.learning_rate,a.warmup_ratio,a.seed,configure_logging("cpt_train"))
