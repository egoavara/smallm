#!/usr/bin/env python3
"""Train a LLaMA-style language model."""

import argparse
import torch

from smallm.model import ModelConfig, LLaMA, CONFIGS
from smallm.tokenizer import RegexBPE
from smallm.data import load_wikitext, create_dataloader
from smallm.training import Trainer, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Train a LLaMA-style LLM")

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium"],
        help="Model size preset (default: small)",
    )

    # Data configuration
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer",
        help="Path to tokenizer (without extension)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length (default: 512)",
    )

    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum training steps (default: 10000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )

    # Other options
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = RegexBPE()
    tokenizer.load(args.tokenizer)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create model config
    config = CONFIGS[args.model_size]
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = args.seq_len

    print(f"\nModel configuration ({args.model_size}):")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads} (KV: {config.n_kv_heads})")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max seq len: {config.max_seq_len}")

    # Create model
    model = LLaMA(config)
    n_params = model.count_parameters()
    print(f"\nTotal parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Load datasets
    print("\nLoading training data...")
    train_dataset = load_wikitext(tokenizer, split="train", seq_len=args.seq_len)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    print(f"Training samples: {len(train_dataset):,}")

    print("\nLoading validation data...")
    eval_dataset = load_wikitext(tokenizer, split="validation", seq_len=args.seq_len)
    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"Validation samples: {len(eval_dataset):,}")

    # Create training config
    train_config = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        compile_model=args.compile,
    )

    # Create trainer
    trainer = Trainer(model, train_loader, train_config, eval_loader)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Estimate memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Print memory usage
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory: {max_memory:.2f} GB")


if __name__ == "__main__":
    main()
