"""Shared QLoRA/SFTTrainer implementation for adapters A, B, and C."""

from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, set_seed

ADAPTERS={
    "A":{"r":8,"alpha":16,"target_modules":["q_proj","v_proj"]},
    "B":{"r":16,"alpha":32,"target_modules":["q_proj","v_proj"]},
    "C":{"r":32,"alpha":32,"target_modules":["q_proj","v_proj","o_proj"]},
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_chat(row: dict, tokenizer) -> str:
    messages=[{"role":"user","content":row["instruction"]},{"role":"assistant","content":row["response"]}]
    if getattr(tokenizer,"chat_template",None):
        return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    return f"<|user|>\n{row['instruction']}\n<|assistant|>\n{row['response']}"


def run(adapter: str, base_model: str, dataset_path: Path, output_dir: Path, max_steps: int, seed: int, logger) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer
    from src.utils.modeling import load_tokenizer
    from transformers import AutoModelForCausalLM

    cfg=ADAPTERS[adapter];set_seed(seed);tokenizer=load_tokenizer(base_model)
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(base_model,quantization_config=quant,device_map="auto",torch_dtype=torch.bfloat16)
    model.config.use_cache=False
    rows=[{"text":format_chat(row,tokenizer)} for row in load_rows(dataset_path)]
    dataset=Dataset.from_list(rows)
    lora=LoraConfig(r=cfg["r"],lora_alpha=cfg["alpha"],lora_dropout=0.05,bias="none",task_type=TaskType.CAUSAL_LM,target_modules=cfg["target_modules"])
    cuda=torch.cuda.is_available(); bf16=cuda and torch.cuda.is_bf16_supported()
    args=TrainingArguments(output_dir=str(output_dir),per_device_train_batch_size=2,gradient_accumulation_steps=4,learning_rate=2e-4,num_train_epochs=1,max_steps=max_steps,logging_steps=10,save_steps=100,save_total_limit=2,report_to=[],remove_unused_columns=False,fp16=cuda and not bf16,bf16=bf16,optim="paged_adamw_8bit",seed=seed)
    supported=inspect.signature(SFTTrainer.__init__).parameters
    kwargs={"model":model,"args":args,"train_dataset":dataset,"peft_config":lora}
    if "dataset_text_field" in supported: kwargs["dataset_text_field"]="text"
    if "max_seq_length" in supported: kwargs["max_seq_length"]=SETTINGS.max_seq_length
    if "max_length" in supported: kwargs["max_length"]=SETTINGS.max_seq_length
    if "processing_class" in supported: kwargs["processing_class"]=tokenizer
    else: kwargs["tokenizer"]=tokenizer
    try:
        trainer=SFTTrainer(**kwargs)
    except TypeError as exc:
        # Newer TRL releases move dataset formatting into SFTConfig. Tokenize
        # explicitly so the same script remains usable across TRL versions.
        logger.warning("SFTTrainer text-field API mismatch (%s); using explicit tokenization", exc)
        tokenized=dataset.map(lambda batch: tokenizer(batch["text"],truncation=True,max_length=SETTINGS.max_seq_length),batched=True)
        kwargs.pop("dataset_text_field",None);kwargs.pop("max_seq_length",None);kwargs.pop("max_length",None);kwargs["train_dataset"]=tokenized
        trainer=SFTTrainer(**kwargs)
    trainer.train();trainer.save_model(output_dir);tokenizer.save_pretrained(output_dir)
    (output_dir/"adapter_config_assignment.json").write_text(json.dumps({"adapter":adapter,**cfg},indent=2),encoding="utf-8")
    logger.info("Saved Adapter %s to %s",adapter,output_dir)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--adapter",choices=sorted(ADAPTERS),required=True);p.add_argument("--base-model",default=str(SETTINGS.models_dir/"cpt_model"));p.add_argument("--dataset",type=Path,default=SETTINGS.instruction_dataset/"train.jsonl");p.add_argument("--output-dir",type=Path);p.add_argument("--max-steps",type=int,default=SETTINGS.max_steps);p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args();a.output_dir=a.output_dir or SETTINGS.models_dir/f"adapter_{a.adapter}";run(a.adapter,a.base_model,a.dataset,a.output_dir,a.max_steps,a.seed,configure_logging(f"train_adapter_{a.adapter}"))
