"""Create grounded banking instruction pairs from cleaned sentences using heuristics."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from src.config.settings import SETTINGS
from src.utils.common import configure_logging, set_seed, write_json


TEMPLATES = [
    "What does the source document state about {topic}?",
    "Summarize the compliance requirement described in this provision about {topic}.",
    "Which banking or regulatory rule is described regarding {topic}?",
    "What should a compliance analyst know about {topic} from this provision?",
]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(s.strip()) >= 40]


def topic(sentence: str) -> str:
    words=re.findall(r"[A-Za-z][A-Za-z0-9-]*", sentence)
    return " ".join(words[: min(9, len(words))]).rstrip(".,:;")


def build_pairs(input_dir: Path, minimum: int, seed: int) -> list[dict[str, str]]:
    candidates=[];seen=set()
    for source in sorted(input_dir.rglob("*.txt")):
        for sentence in sentences(source.read_text(encoding="utf-8",errors="replace")):
            for template in TEMPLATES:
                instruction=template.format(topic=topic(sentence))
                key=(instruction,sentence)
                if key not in seen:
                    seen.add(key);candidates.append({"instruction":instruction,"response":sentence,"source_document":source.relative_to(input_dir).as_posix()})
    random.Random(seed).shuffle(candidates)
    if len(candidates)<minimum:
        raise ValueError(f"Only {len(candidates)} grounded pairs could be created; at least {minimum} are required. Add more cleaned banking PDFs.")
    return candidates


def run(input_dir: Path, output_dir: Path, minimum: int, seed: int, logger) -> None:
    set_seed(seed);pairs=build_pairs(input_dir,minimum,seed);split=max(1,round(len(pairs)*.8));train,eval_pairs=pairs[:split],pairs[split:]
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in (("instruction_dataset.jsonl",pairs),("train.jsonl",train),("eval.jsonl",eval_pairs)):
        with (output_dir/name).open("w",encoding="utf-8") as h:
            for row in data: h.write(json.dumps(row,ensure_ascii=False)+"\n")
    write_json(output_dir/"split_report.json",{"total_pairs":len(pairs),"train_pairs":len(train),"eval_pairs":len(eval_pairs),"train_fraction":len(train)/len(pairs),"seed":seed,"generation_method":"deterministic heuristic sentence grounding"})
    logger.info("Created %d pairs (%d train / %d eval)",len(pairs),len(train),len(eval_pairs))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",type=Path,default=SETTINGS.cleaned_txt);p.add_argument("--output-dir",type=Path,default=SETTINGS.instruction_dataset);p.add_argument("--minimum",type=int,default=100);p.add_argument("--seed",type=int,default=SETTINGS.seed)
    a=p.parse_args();run(a.input_dir,a.output_dir,a.minimum,a.seed,configure_logging("create_instruction_dataset"))
