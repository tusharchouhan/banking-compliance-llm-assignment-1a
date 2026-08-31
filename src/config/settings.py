"""Central configuration for the reproducible Assignment 1A pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    model_id: str = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM2-360M")
    seed: int = int(os.getenv("SEED", "42"))
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.1"))
    max_steps: int = int(os.getenv("MAX_STEPS", "500"))

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_pdfs(self) -> Path:
        return self.data_dir / "raw_pdfs"

    @property
    def extracted_txt(self) -> Path:
        return self.data_dir / "extracted_txt"

    @property
    def cleaned_txt(self) -> Path:
        return self.data_dir / "cleaned_txt"

    @property
    def train_corpus(self) -> Path:
        return self.data_dir / "train_corpus"

    @property
    def eval_corpus(self) -> Path:
        return self.data_dir / "eval_corpus"

    @property
    def instruction_dataset(self) -> Path:
        return self.data_dir / "instruction_dataset"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    def ensure_directories(self) -> None:
        """Create runtime directories while preserving any user-provided files."""
        for path in (
            self.raw_pdfs, self.extracted_txt, self.cleaned_txt,
            self.train_corpus, self.eval_corpus, self.instruction_dataset,
            self.models_dir / "base_model", self.models_dir / "cpt_model",
            self.models_dir / "adapter_A", self.models_dir / "adapter_B",
            self.models_dir / "adapter_C", self.reports_dir / "figures",
            self.reports_dir / "tables", self.reports_dir / "final_report",
            self.results_dir / "baseline_outputs", self.results_dir / "perplexity",
            self.results_dir / "forgetting_check", self.results_dir / "adapter_comparison",
        ):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()

