#!/usr/bin/env python3
"""Decode stored blocks back to text, to see what a tokenization actually holds.

WHY THIS EXISTS. This project's rule is to verify a tokenization by DECODING its
blocks, not by trusting occupancy — occupancy looked healthy through every defect
the 2026-09-11 corpus arc found. Bloomberg tokenized on `Headline` reported ~99%
real tokens; so did OpenHermes tokenized on the raw list-of-dicts, which was 490
blocks of pure `<|im_end|>`. Only decoding showed it.

Every recorded instance of that practice was an ad-hoc one-liner. This is the
executable version, so the check is repeatable and the evidence is quotable.

WHAT IT ANSWERS

  * Is this corpus SHUFFLED? Blocks 0, N/2 and N-1 of an unshuffled corpus are
    three consecutive slices of one document stream and usually read as the same
    source, era or repository. Shuffled, they should look unrelated. Extraction
    reads a PREFIX, so an ordered corpus means every extraction ever run saw the
    same opening slice.
  * Is it tokenized on the right column, with the right template, at the right
    precision? The decoded text says so immediately.

USAGE

    python scripts/decode_tokenized_blocks.py --path /data/datasets/<dir>
    python scripts/decode_tokenized_blocks.py --path <dir> --blocks 0,1,2 --chars 600
    python scripts/decode_tokenized_blocks.py --path <dir> --tokenizer LiquidAI/LFM2.5-1.2B-Instruct

`--tokenizer` is optional: with no tokenizer the block is reported structurally
(length, mask occupancy, first/last ids), which is still enough to spot a padded
or degenerate block. With one, you get the text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", required=True, help="the tokenized Arrow directory")
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF repo id or local path; omit for a structural report only",
    )
    p.add_argument(
        "--blocks",
        default=None,
        help="comma-separated block indices; default is first, middle and last",
    )
    p.add_argument("--chars", type=int, default=400, help="characters of text per block")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.path)
    if not path.is_dir():
        print(f"not a directory: {path}", file=sys.stderr)
        return 2

    from datasets import load_from_disk

    dataset = load_from_disk(str(path))
    n = len(dataset)

    # An `indices.arrow` means the rows are a VIEW over data still physically in
    # its original order. `shuffle_blocks` calls `flatten_indices()` precisely so
    # this file does not appear: every consumer then reads a plain sequential
    # dataset, which is what they all assume.
    mapping = sorted(path.glob("indices*.arrow"))
    print(f"blocks        : {n:,}")
    print(f"columns       : {list(dataset.features)}")
    print(f"indices map   : {'PRESENT — rows are a view, not physically reordered' if mapping else 'none (physically ordered)'}")

    if args.blocks:
        picks = [int(x) for x in args.blocks.split(",") if x.strip()]
    else:
        picks = sorted({0, n // 2, n - 1}) if n else []
    picks = [i for i in picks if 0 <= i < n]
    if not picks:
        print("no blocks to show", file=sys.stderr)
        return 2

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"tokenizer     : {args.tokenizer}")

    print()
    print("Compare these. In an UNSHUFFLED corpus they are three consecutive")
    print("slices of one stream and usually share a source, era or repository.")
    print("Shuffled, they should look unrelated to each other.")

    for i in picks:
        row = dataset[i]
        ids = row["input_ids"]
        mask = row.get("attention_mask")
        real = sum(mask) if mask is not None else len(ids)
        print()
        print("=" * 78)
        print(f"block {i:,} of {n:,}   tokens={len(ids):,}   real={real:,} "
              f"({real / max(1, len(ids)):.1%})   first_ids={list(ids[:8])}")
        print("-" * 78)
        if tokenizer is None:
            print(f"  (no tokenizer given; last ids: {list(ids[-8:])})")
            continue
        text = tokenizer.decode(ids[:args.chars], skip_special_tokens=False)
        print(text[:args.chars].replace("\n", "\n  "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
