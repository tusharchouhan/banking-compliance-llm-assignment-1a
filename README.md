# Banking & Compliance Domain LLM — Assignment 1A
## Group 14
<ul> 
<li>Tushar Chouhan	2025cs05043
<li>Kelli L Prasanna Kumar	2025cs05010
<li>Jyoti Chugh	2025cs05045
<li>Madhan M	2025cs05047
</ul>
Raw PDF Location - https://drive.google.com/drive/folders/1l9pcE_NOCMpHxio1ZnAD94j9cWwxNY0j?usp=sharing

This repository implements Assignment 1A for Enterprise Variant V3, Banking/FinTech/Insurance Regulatory Compliance. It uses `HuggingFaceTB/SmolLM2-360M` and is designed for a Google Colab T4 (16 GB VRAM). The assignment documents in `Assignment Documentation/` are the requirements source of truth.

## Objective

Build a reproducible domain adaptation pipeline: collect 10–50 MB of public banking/regulatory PDFs, extract and clean them, create a held-out corpus, pack tokens with the model’s own tokenizer, continually pre-train the model, measure perplexity and catastrophic forgetting, create at least 100 grounded instruction pairs, and train/compare three QLoRA adapters.

Variant V3 sources include RBI, SEBI, IRDAI, BIS Basel III, and FATF publications. Download only documents you are permitted to use and place them anywhere below `data/raw_pdfs/`; nested folders are discovered recursively and mirrored under the extracted/cleaned/split corpus directories.

## Folder structure

```text
LLM_ASSIGNMENT/
├── Assignment Documentation/       # supplied Word requirements
├── data/{raw_pdfs,extracted_txt,cleaned_txt,train_corpus,eval_corpus,instruction_dataset}/
├── models/{base_model,cpt_model,adapter_A,adapter_B,adapter_C}/
├── notebooks/{Assignment_PartA,Assignment_PartB}.ipynb
├── reports/{figures,tables,final_report}/
├── results/{baseline_outputs,perplexity,forgetting_check,adapter_comparison}/
├── src/{config,data,training,evaluation,qlora,utils}/
├── requirements.txt
└── run_pipeline.py
```

## Setup

The intended runtime is Colab/Linux with a CUDA T4. Install dependencies in a fresh environment:

```bash
cd LLM_ASSIGNMENT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login                 # if your model access policy requires it
```

On Windows, activate with `.venv\Scripts\activate`. Keep `data/` and `models/` on persistent storage when using Colab.

## Training commands

Run the complete Assignment 1A pipeline after placing PDFs in `data/raw_pdfs/`:

```bash
python run_pipeline.py --max-steps 500
```

Every stage is independent and can be rerun:

```bash
python -m src.data.pdf_extractor
python -m src.data.clean_corpus
python -m src.data.train_eval_split
python -m src.data.tokenize_and_pack
python -m src.evaluation.model_inspection
python -m src.evaluation.baseline_inference
python -m src.training.cpt_train --max-steps 500
python -m src.evaluation.loss_analysis
python -m src.evaluation.perplexity
python -m src.evaluation.forgetting_check
python -m src.data.create_instruction_dataset --minimum 100
python -m src.qlora.train_adapter_A
python -m src.qlora.train_adapter_B
python -m src.qlora.train_adapter_C
python -m src.evaluation.adapter_comparison
```

For a smoke/short run, use `--max-steps 10`; the submitted experiment should use a documented value and report the observed loss curve. The pipeline does not fabricate data: it stops with an actionable error if the PDF corpus cannot produce the required 100 grounded pairs.

## Evaluation outputs

- `reports/tables/extraction_report.csv`, `cleaning_report.csv`, `split_report.csv`, `token_statistics.csv`, and `model_architecture.csv`
- `data/train_packed.parquet` with BOS/EOS-wrapped, fixed-length sequences
- `models/cpt_model/` with the final CPT checkpoint and tokenizer
- `reports/tables/loss_history.csv` and `reports/figures/loss_curve.png`
- `results/perplexity/ppl_results.csv` with base PPL, CPT PPL, and percentage improvement
- `results/forgetting_check/forgetting_comparison.csv` and Markdown side-by-side table
- `data/instruction_dataset/instruction_dataset.jsonl`, `train.jsonl`, `eval.jsonl`, and `split_report.json`
- `models/adapter_A/`, `adapter_B/`, `adapter_C/` and `results/adapter_comparison/`

## Reproducibility and interpretation

The default seed is 42 and can be overridden with `--seed` or `SEED`. The 10% evaluation documents are never used in CPT. Lower held-out domain PPL after CPT indicates domain adaptation. Inspect the forgetting table manually; a significant degradation suggests lowering the learning rate by 10× or halving `max_steps`, as specified by the assignment. The Adapter comparison also reports a keyword-grounding proxy to make its automatic “most relevant” flag auditable; final accuracy remains a human review criterion.

For compliance, the Variant V3 guidance requires deterministic greedy decoding in any final demonstration. Model outputs are not regulatory advice and must be checked against current primary-source regulations before use.
