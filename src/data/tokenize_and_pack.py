"""Tokenize the selected model's corpus and pack it into fixed-length Parquet rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, set_seed


def tokenize_documents(input_dir: Path, tokenizer, logger) -> list[int]:
    stream: list[int] = []
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("The selected pretrained tokenizer must provide BOS and EOS token IDs.")
    for path in sorted(input_dir.rglob("*.txt")):
        ids = tokenizer(path.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False)["input_ids"]
        stream.extend([bos, *ids, eos])
        logger.info("Tokenized %s: %d tokens", path.name, len(ids))
    return stream


def run(input_dir: Path, output_path: Path, stats_path: Path, model_id: str, seq_len: int, seed: int, logger) -> None:
    from transformers import AutoTokenizer
    import pyarrow as pa
    import pyarrow.parquet as pq
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    stream = tokenize_documents(input_dir, tokenizer, logger)
    sequences = [stream[i:i + seq_len] for i in range(0, len(stream) - seq_len + 1, seq_len)]
    if not sequences:
        raise ValueError("The corpus is shorter than one packed sequence; add more domain text or lower --seq-len.")
    table = pa.table({"input_ids": sequences, "attention_mask": [[1] * len(row) for row in sequences]})
    output_path.parent.mkdir(parents=True, exist_ok=True); pq.write_table(table, output_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    documents = list(input_dir.rglob("*.txt"))
    avg = sum(len(tokenizer(p.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]) for p in documents) / max(len(documents), 1)
    with stats_path.open("w", newline="", encoding="utf-8") as h:
        w=csv.DictWriter(h, fieldnames=["model_id","sequence_length","documents","total_tokens_with_bos_eos","average_document_tokens","packed_sequences","discarded_tail_tokens","seed"]); w.writeheader()
        w.writerow({"model_id":model_id,"sequence_length":seq_len,"documents":len(documents),"total_tokens_with_bos_eos":len(stream),"average_document_tokens":round(avg,2),"packed_sequences":len(sequences),"discarded_tail_tokens":len(stream)-len(sequences)*seq_len,"seed":seed})
    logger.info("Wrote %d packed sequences to %s", len(sequences), output_path)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",type=Path,default=SETTINGS.train_corpus)
    p.add_argument("--output",type=Path,default=SETTINGS.data_dir/"train_packed.parquet"); p.add_argument("--stats",type=Path,default=SETTINGS.reports_dir/"tables"/"token_statistics.csv")
    p.add_argument("--model-id",default=SETTINGS.model_id); p.add_argument("--seq-len",type=int,default=SETTINGS.max_seq_length); p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args(); run(a.input_dir,a.output,a.stats,a.model_id,a.seq_len,a.seed,configure_logging("tokenize_and_pack"))
