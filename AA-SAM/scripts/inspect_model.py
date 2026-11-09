from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from aggregate_sam3.model import build_aggregate_sam3_image_model


def main() -> None:
    parser = argparse.ArgumentParser(description="检查Aggregate-SAM3可训练参数。")
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--freeze-mode",
        choices=("adapter", "vision_frozen", "full"),
        default="adapter",
    )
    parser.add_argument("--branch-base-channels", type=int, default=32)
    args = parser.parse_args()

    bpe_path = args.sam3_root.resolve() / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_aggregate_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(args.checkpoint.resolve()),
        device="cpu",
        freeze_mode=args.freeze_mode,
        branch_base_channels=args.branch_base_channels,
        load_from_HF=False,
        enable_segmentation=True,
        compile=False,
    )

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_total = sum(n for _, n in trainable)
    print(f"[参数] 可训练={trainable_total:,}；总参数={total:,}；比例={trainable_total/max(total,1):.2%}")
    print("[可训练模块示例]")
    for name, count in trainable[:80]:
        print(f"  {count:>12,}  {name}")


if __name__ == "__main__":
    main()
