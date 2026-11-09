#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复 COCO 数据集中的 text_input 字段。

作用
----
- 为 train / val / test 的每条 images 记录写入：
      "text_input": "aggregate particle"
- 同步补充 categories 中的 prompt 字段；
- 同步补充 info 中的 prompt 字段；
- 修改前自动备份原 JSON；
- 原子写入，避免中途损坏 JSON。

示例
----
python patch_coco_text_input.py ^
  --dataset-root "./data/example" ^
  --text-input "aggregate particle"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为 COCO images 批量补充 text_input 字段。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="COCO 数据集根目录，包含 train/val/test。",
    )
    parser.add_argument(
        "--text-input",
        type=str,
        required=True,
        help='写入每张图的文本提示，例如 "aggregate particle"。',
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="要处理的划分，逗号分隔。默认 train,val,test。",
    )
    parser.add_argument(
        "--annotation-name",
        type=str,
        default="_annotations.coco.json",
        help="COCO 标注文件名。",
    )
    parser.add_argument(
        "--force-backup",
        action="store_true",
        help="备份文件已存在时覆盖备份。",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def patch_one(
    annotation_path: Path,
    text_input: str,
    force_backup: bool,
) -> dict[str, int]:
    with annotation_path.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)

    images = coco.get("images")
    if not isinstance(images, list):
        raise ValueError(f"COCO images 不是列表：{annotation_path}")

    backup_path = annotation_path.with_suffix(
        annotation_path.suffix + ".backup_before_text_input"
    )
    if backup_path.exists():
        if force_backup:
            shutil.copy2(annotation_path, backup_path)
    else:
        shutil.copy2(annotation_path, backup_path)

    changed_images = 0
    already_correct = 0

    for image in images:
        old_value = image.get("text_input", "")
        if old_value == text_input:
            already_correct += 1
        else:
            image["text_input"] = text_input
            changed_images += 1

    categories = coco.get("categories", [])
    changed_categories = 0
    if isinstance(categories, list):
        for category in categories:
            if category.get("prompt") != text_input:
                category["prompt"] = text_input
                changed_categories += 1

    info = coco.get("info")
    if not isinstance(info, dict):
        info = {}
        coco["info"] = info
    info["prompt"] = text_input
    info["text_input"] = text_input

    atomic_write_json(annotation_path, coco)

    # 重新读取并严格验证
    with annotation_path.open("r", encoding="utf-8") as stream:
        checked = json.load(stream)

    bad_values = sorted({
        str(image.get("text_input", ""))
        for image in checked.get("images", [])
        if image.get("text_input", "") != text_input
    })
    if bad_values:
        raise RuntimeError(
            f"修复后验证失败：{annotation_path}；异常值={bad_values}"
        )

    return {
        "image_count": len(images),
        "changed_images": changed_images,
        "already_correct": already_correct,
        "changed_categories": changed_categories,
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    text_input = args.text_input.strip()

    if not root.is_dir():
        raise NotADirectoryError(root)
    if not text_input:
        raise ValueError("--text-input 不能为空。")

    splits = [
        item.strip()
        for item in args.splits.split(",")
        if item.strip()
    ]

    print("=" * 72)
    print("COCO text_input 修复")
    print(f"数据集：{root}")
    print(f"text_input：{text_input!r}")
    print("=" * 72)

    processed = 0
    total_images = 0

    for split in splits:
        annotation_path = root / split / args.annotation_name
        if not annotation_path.is_file():
            print(f"[跳过] {split}：未找到 {annotation_path}")
            continue

        result = patch_one(
            annotation_path,
            text_input,
            args.force_backup,
        )
        processed += 1
        total_images += result["image_count"]

        print(
            f"[完成] {split}: images={result['image_count']}；"
            f"新写入={result['changed_images']}；"
            f"原本正确={result['already_correct']}；"
            f"categories修复={result['changed_categories']}"
        )
        print(
            f"       备份："
            f"{annotation_path.with_suffix(annotation_path.suffix + '.backup_before_text_input')}"
        )

    if processed == 0:
        raise RuntimeError("没有找到任何可处理的 COCO 标注文件。")

    print("=" * 72)
    print(f"[全部完成] 已处理 {processed} 个划分，共 {total_images} 张图。")
    print("=" * 72)


if __name__ == "__main__":
    main()
