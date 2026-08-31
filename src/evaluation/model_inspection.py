"""Inspect the selected causal LM and write the required architecture report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging


def first_attr(config, names, default="unknown"):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def run(model_id: str, report: Path, load_model: bool, logger) -> None:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)
    model = None
    if load_model:
        from src.utils.modeling import load_causal_lm
        model = load_causal_lm(model_id)
    hidden = first_attr(config, ["hidden_size", "n_embd"])
    heads = first_attr(config, ["num_attention_heads", "n_head"])
    layers = first_attr(config, ["num_hidden_layers", "n_layer", "num_layers"])
    vocab = first_attr(config, ["vocab_size"])
    head_dim = hidden // heads if isinstance(hidden, int) and isinstance(heads, int) and heads else "unknown"
    total = sum(p.numel() for p in model.parameters()) if model is not None else "not_loaded"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else "not_loaded"
    lm_head_dim = "unknown"
    if model is not None and hasattr(model, "lm_head"):
        lm_head_dim = model.lm_head.out_features
        if lm_head_dim != vocab:
            raise ValueError(f"lm_head output dimension {lm_head_dim} != vocabulary size {vocab}")
    rows = [{"model_id":model_id,"total_parameters":total,"trainable_parameters":trainable,"decoder_layers":layers,"attention_heads":heads,
             "hidden_size":hidden,"vocabulary_size":vocab,"head_dimension":head_dim,
             "lm_head_output_dimension":lm_head_dim,"lm_head_matches_vocab":lm_head_dim == vocab if model is not None else "not_checked"}]
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    logger.info("Architecture report written to %s", report)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default=SETTINGS.model_id)
    p.add_argument("--report",type=Path,default=SETTINGS.reports_dir/"tables"/"model_architecture.csv");p.add_argument("--config-only",action="store_true",help="Skip model loading and only inspect AutoConfig")
    a=p.parse_args();run(a.model_id,a.report,not a.config_only,configure_logging("model_inspection"))
