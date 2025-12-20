#!/usr/bin/env python3
"""Train a BPE tokenizer on WikiText-103."""

import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from smallm.tokenizer import RegexBPE


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer",
        help="Output path (without extension)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print merge details",
    )
    args = parser.parse_args()

    # Load WikiText-103
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Collect text
    print("Collecting text...")
    texts = []
    for item in tqdm(dataset):
        text = item["text"]
        if text.strip():
            texts.append(text)

    full_text = "\n".join(texts)
    print(f"Total characters: {len(full_text):,}")

    # Create and train tokenizer
    print(f"\nTraining tokenizer with vocab_size={args.vocab_size}...")
    tokenizer = RegexBPE()

    # Register special tokens
    special_tokens = {
        "<|endoftext|>": args.vocab_size,
        "<|pad|>": args.vocab_size + 1,
    }
    tokenizer.register_special_tokens(special_tokens)

    # Train
    tokenizer.train(full_text, args.vocab_size, verbose=args.verbose)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"\nSaved tokenizer to {output_path}.model and {output_path}.vocab")

    # Test
    print("\nTesting tokenizer:")
    test_text = "Hello, world! This is a test of the tokenizer."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"  Original: {test_text!r}")
    print(f"  Tokens:   {tokens}")
    print(f"  Decoded:  {decoded!r}")
    print(f"  Match:    {test_text == decoded}")


if __name__ == "__main__":
    main()
