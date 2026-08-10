"""외부 모델 다운로드 없는 tiny causal LM dry-run 도구."""

from __future__ import annotations

from pathlib import Path

from .student_data import build_student_dataset
from .student_training import load_student_config, run_student


def create_tiny_causal_lm(path: Path, seed: int = 17) -> Path:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    if path.exists() and any(path.iterdir()):
        raise ValueError(f"tiny base model output collision: {path}")
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "###": 4, "질문": 5, "답변": 6, "정답": 7}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_positions=64,
            n_ctx=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            loss_type="ForCausalLM",
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def run_tiny_student_dry_run(config_path: Path, root: Path = Path(".")) -> dict:
    config = load_student_config(config_path)
    dry = config.get("dry_run")
    if not isinstance(dry, dict) or not dry.get("source_dataset"):
        raise ValueError("dry_run.source_dataset is required")
    base_model = Path(config["base_model"])
    base_model = base_model if base_model.is_absolute() else root / base_model
    dataset = Path(config["dataset"])
    dataset = dataset if dataset.is_absolute() else root / dataset
    source = Path(dry["source_dataset"])
    source = source if source.is_absolute() else root / source
    create_tiny_causal_lm(base_model, seed=int(config["seed"]))
    build_student_dataset(source, dataset, condition=config["condition"])
    return run_student(config_path, root=root)
