#!/usr/bin/env python3
"""
train_dpo.py ─ Direct-Preference-Optimization fine-tuning script
------------------------------------------------------------------
Typical usage
------------------------------------------------------------------
python train_dpo.py \
  --prefs_file data/processed/preferences.jsonl \
  --sl_cai_path models/stage_02_sl_cai \
  --output_dir  models/stage_03_dpo_cai \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-6 \
  --seed 42

or from a YAML:

python train_dpo.py --config configs/dpo_run.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

# Dependency checking and installation
def check_and_install_dependencies():
    """Check and install required dependencies if missing."""
    required_packages = [
        "torch", "transformers", "datasets", "trl", "accelerate", "packaging"
    ]
    optional_packages = [
        ("bitsandbytes", "8-bit training support"),
        ("yaml", "YAML config file support")
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("💡 Please install with: pip install torch transformers datasets trl accelerate packaging")
        sys.exit(1)
    
    # Check optional packages
    for package, description in optional_packages:
        try:
            __import__(package)
        except ImportError:
            if package == "yaml":
                print(f"⚠️  {description} not available. Install with: pip install pyyaml")
            else:
                print(f"⚠️  {description} not available. Install with: pip install {package}")

# Run dependency check immediately
check_and_install_dependencies()

import torch
from datasets import Dataset, load_dataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

try:
    import yaml  #  optional, only if users want --config
except ModuleNotFoundError:
    yaml = None  # type: ignore


# helper utilities
def parse_args() -> argparse.Namespace:
    """Parse CLI args, optionally merging a YAML config."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="YAML file with any of the args below")

    # data / io
    p.add_argument("--prefs_file", type=str, help="JSONL file with preference pairs")
    p.add_argument("--sl_cai_path", type=str, help="Path or HF hub ID for SL-CAI model")
    p.add_argument("--ref_model_path", type=str, default=None, help="Frozen reference model (defaults to sl_cai_path)")
    p.add_argument("--output_dir", type=str, help="Where to save the DPO-tuned model")
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="'latest' or path to checkpoint")

    # optimisation
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.1, help="Inverse-temperature for DPO loss")

    # misc / reproducibility
    p.add_argument("--eval_split", type=float, default=0.05, help="Fraction of data for evaluation")
    p.add_argument("--max_length", type=int, default=1024, help="Context length for tokenisation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_strategy", choices=["no", "epoch", "steps"], default="epoch")

    # memory / speed tricks
    p.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantisation via bitsandbytes")
    p.add_argument("--no_wandb", action="store_true", help="Force-disable Weights & Biases even if env vars are set")

    args, unknown = p.parse_known_args()
    if unknown:
        p.error(f"Unknown args: {unknown}")

    # merge YAML if present
    if args.config:
        if yaml is None:
            raise RuntimeError("PyYAML not installed but --config provided")
        cfg_path = Path(args.config).expanduser().resolve()
        with cfg_path.open() as f:
            cfg_dict = yaml.safe_load(f)
        
        # Parse args again without defaults to see what was explicitly provided
        p_no_defaults = argparse.ArgumentParser()
        p_no_defaults.add_argument("--config", type=str)
        p_no_defaults.add_argument("--prefs_file", type=str)
        p_no_defaults.add_argument("--sl_cai_path", type=str)
        p_no_defaults.add_argument("--ref_model_path", type=str)
        p_no_defaults.add_argument("--output_dir", type=str)
        p_no_defaults.add_argument("--resume_from_checkpoint", type=str)
        p_no_defaults.add_argument("--num_train_epochs", type=float)
        p_no_defaults.add_argument("--per_device_train_batch_size", type=int)
        p_no_defaults.add_argument("--per_device_eval_batch_size", type=int)
        p_no_defaults.add_argument("--gradient_accumulation_steps", type=int)
        p_no_defaults.add_argument("--learning_rate", type=float)
        p_no_defaults.add_argument("--warmup_ratio", type=float)
        p_no_defaults.add_argument("--beta", type=float)
        p_no_defaults.add_argument("--eval_split", type=float)
        p_no_defaults.add_argument("--max_length", type=int)
        p_no_defaults.add_argument("--seed", type=int)
        p_no_defaults.add_argument("--logging_steps", type=int)
        p_no_defaults.add_argument("--eval_steps", type=int)
        p_no_defaults.add_argument("--save_strategy", type=str)
        p_no_defaults.add_argument("--load_in_8bit", action="store_true")
        p_no_defaults.add_argument("--no_wandb", action="store_true")
        
        args_no_defaults, _ = p_no_defaults.parse_known_args()
        
        # Apply YAML values only for arguments not explicitly provided by CLI
        for k, v in cfg_dict.items():
            if hasattr(args, k) and getattr(args_no_defaults, k) is None:
                setattr(args, k, v)

    # final sanity
    required = ["prefs_file", "sl_cai_path", "output_dir"]
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"Missing mandatory args: {', '.join(missing)}")

    return args


def set_seed_all(seed: int) -> None:
    """Deterministic everywhere."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_dtype() -> torch.dtype:
    """Choose fp16 for CUDA, bf16 for MPS≥2.1, else fp32."""
    if torch.cuda.is_available():
        return torch.float16
    _is_mps = torch.backends.mps.is_available()
    if _is_mps and version.parse(torch.__version__) >= version.parse("2.1"):
        return torch.bfloat16
    return torch.float32


def build_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train_dpo")


def validate_paths(args: argparse.Namespace) -> None:
    """Fail fast if files/dirs are wrong."""
    prefs = Path(args.prefs_file).expanduser().resolve()
    if not prefs.is_file():
        raise FileNotFoundError(prefs)
    _ = Path(args.output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def load_preference_dataset(path: str) -> Dataset:
    """Load JSONL with {'prompt','chosen','rejected'} records."""
    with Path(path).open() as f:
        data = [json.loads(l) for l in f]
    
    # Validate data format
    required_keys = {"prompt", "chosen", "rejected"}
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {i} is not a dictionary")
        missing_keys = required_keys - record.keys()
        if missing_keys:
            raise ValueError(f"Record {i} missing required keys: {missing_keys}")
        # Ensure all values are strings
        for key in required_keys:
            if not isinstance(record[key], str):
                raise ValueError(f"Record {i}, key '{key}' must be a string")
    
    return Dataset.from_list(data)


def load_model(path: str, dtype: torch.dtype, load_in_8bit: bool) -> AutoModelForCausalLM:
    """Handle optional 8-bit quantisation with error recovery."""
    if load_in_8bit:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        try:
            return AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=dtype,
                quantization_config=bnb_cfg,
                low_cpu_mem_usage=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.getLogger("train_dpo").warning(
                    "OOM with 8-bit loading, falling back to CPU offloading"
                )
                return AutoModelForCausalLM.from_pretrained(
                    path,
                    device_map="auto",
                    torch_dtype=dtype,
                    quantization_config=bnb_cfg,
                    low_cpu_mem_usage=True,
                    max_memory={0: "0.8GiB", "cpu": "16GiB"},
                )
            raise
    
    try:
        return AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.getLogger("train_dpo").warning(
                "OOM loading model, trying with CPU offloading"
            )
            return AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                max_memory={0: "0.8GiB", "cpu": "16GiB"},
            )
        raise


def simple_preflight_check(args):
    """Quick validation before training starts."""
    print("🛫 Running preflight check...")
    
    # Check data file
    if not Path(args.prefs_file).exists():
        raise FileNotFoundError(f"Preferences file not found: {args.prefs_file}")
    
    # Check SL model directory
    sl_path = Path(args.sl_cai_path)
    if not sl_path.exists():
        raise FileNotFoundError(f"SL model directory not found: {args.sl_cai_path}")
    
    config_file = sl_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Model config not found: {config_file}")
    
    model_files = list(sl_path.glob("*.bin")) + list(sl_path.glob("*.safetensors"))
    if not model_files:
        raise FileNotFoundError(f"Model weights not found in: {args.sl_cai_path}")
    
    print("✅ Preflight check passed!")


# Main
def main() -> None:
    args = parse_args()
    validate_paths(args)
    simple_preflight_check(args)  # Add preflight check
    log = build_logger()

    # Disable WandB if requested
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    set_seed_all(args.seed)
    dtype = select_dtype()
    log.info(f"Using torch dtype: {dtype}")

    # ------------------------------------------------------------------ data
    full_ds = load_preference_dataset(args.prefs_file)
    split = full_ds.train_test_split(
        test_size=args.eval_split,
        shuffle=True,
        seed=args.seed,
    )
    train_ds, eval_ds = split["train"], split["test"]
    log.info(f"Dataset sizes – train: {len(train_ds)}, eval: {len(eval_ds)}")

    # ------------------------------------------------------------ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sl_cai_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ------------------------------------------------------------- models
    policy_model = load_model(args.sl_cai_path, dtype, args.load_in_8bit)
    ref_path = args.ref_model_path or args.sl_cai_path
    ref_model = load_model(ref_path, dtype, args.load_in_8bit)
    ref_model.eval()  # freeze & eval
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ---------------------------------------------------- training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        report_to=[],  # fully disable WandB/Comet
        seed=args.seed,
        dataloader_num_workers=4,
        run_name="dpo_cai",
    )

    # ------------------------------------------------------------ trainer
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # ----------------------------------------------------------- training
    log.info("Starting DPO training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    log.info("Training finished")

    # ------------------------------------------------------------- save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info(f"Final artefacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
