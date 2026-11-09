#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将“原图文件夹 + 实例标签图文件夹”转换为标准 COCO 实例分割数据集。

默认标签约定
------------
1. 标签图与原图文件名主体一致，例如：
       images/Image_001.jpg
       masks/Image_001.png
2. 单通道实例标签：
       0     = 背景
       1..N  = 不同实例
   推荐使用 uint16 PNG，避免实例数超过 255 时发生编号溢出。
3. 如果标签是二值图（仅 0/1 或 0/255），脚本会自动按连通域拆成实例。
4. 如果标签是 RGB 彩色实例图，每一种非背景颜色视为一个实例。

输出结构
--------
output_root/
├─ train/
│  ├─ images/
│  ├─ labels/                    # 使用 --copy-masks 时生成
│  └─ _annotations.coco.json
├─ val/
│  ├─ images/
│  ├─ labels/
│  └─ _annotations.coco.json
├─ test/
│  ├─ images/
│  ├─ labels/
│  └─ _annotations.coco.json
├─ conversion_summary.json
├─ image_instance_statistics.csv
└─ conversion_issues.csv

依赖
----
conda install -c conda-forge numpy opencv pycocotools
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise SystemExit(
        "缺少 pycocotools。\n"
        "请执行：conda install -c conda-forge pycocotools"
    ) from exc


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
MASK_EXTENSIONS = {
    ".png", ".tif", ".tiff", ".bmp"
}

COMMON_MASK_SUFFIXES = (
    "",
    "_mask",
    "_masks",
    "_label",
    "_labels",
    "_instance",
    "_instances",
    "_instance_mask",
    "_instance_masks",
)


@dataclass
class PairRecord:
    image_path: Path
    mask_path: Path
    relative_image_path: Path
    relative_mask_path: Path
    image_width: int
    image_height: int
    instance_count: int
    kept_instance_count: int
    dropped_small_count: int
    empty_mask: bool
    group_key: str
    split: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将原图文件夹和实例标签图文件夹转换为 "
            "train/val/test COCO 实例分割数据集。"
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="原图文件夹。",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="实例标签图文件夹。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="COCO数据集输出根目录。",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="aggregate particle",
        help='类别名称。默认："aggregate particle"',
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "可选文本提示词；不填写时使用 --category-name。"
            "会写入 category 的 prompt 字段。"
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="训练集比例。默认：0.80",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="验证集比例。默认：0.10",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="测试集比例。默认：0.10",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。默认：42",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录。默认仅搜索文件夹第一层。",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="auto",
        help=(
            '标签文件名后缀。默认 "auto"，自动尝试同名、_mask、'
            "_label、_instances 等。也可指定如：_label"
        ),
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
        help="单通道标签中的背景值。默认：0",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=16,
        help="实例掩膜最小像素面积，小于该值则删除。默认：16",
    )
    parser.add_argument(
        "--binary-connectivity",
        choices=(4, 8),
        type=int,
        default=8,
        help="二值标签按连通域拆分时的邻接方式。默认：8",
    )
    parser.add_argument(
        "--split-disconnected",
        action="store_true",
        help=(
            "若同一实例ID包含多个不连通区域，将其拆成多个实例。"
            "默认保留为同一实例并记录警告。"
        ),
    )
    parser.add_argument(
        "--color-mask",
        choices=("auto", "yes", "no"),
        default="auto",
        help=(
            "彩色标签处理方式。auto自动判断；yes每种颜色为一个实例；"
            "no要求标签为单通道。"
        ),
    )
    parser.add_argument(
        "--copy-masks",
        action="store_true",
        help="把原始标签图同步复制到每个split的labels目录。",
    )
    parser.add_argument(
        "--group-mode",
        choices=("none", "auto_tile", "parent"),
        default="auto_tile",
        help=(
            "数据划分分组方式。auto_tile会把同一原图产生的旧图块放在同一"
            "split；none每张图独立；parent按父目录分组。默认：auto_tile"
        ),
    )
    parser.add_argument(
        "--group-regex",
        type=str,
        default=None,
        help=(
            "可选正则表达式，用第1个捕获组作为分组ID。"
            "优先级高于 --group-mode。"
        ),
    )
    parser.add_argument(
        "--stratify-bins",
        type=str,
        default="0,50,100,180,300,500,1000000000",
        help=(
            "按实例数分层划分的边界。默认："
            "0,50,100,180,300,500,1000000000"
        ),
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="保留全背景标签图。默认丢弃空标签图。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录存在时先删除后重建。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查配对和标签统计，不复制文件、不生成COCO数据集。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(value < 0 for value in ratios):
        raise ValueError("train/val/test 比例不能为负数。")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "--train-ratio、--val-ratio、--test-ratio 之和必须等于1。"
        )
    if args.min_area <= 0:
        raise ValueError("--min-area 必须大于0。")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(args.image_dir)
    if not args.mask_dir.is_dir():
        raise NotADirectoryError(args.mask_dir)
    if args.image_dir.resolve() == args.output_root.resolve():
        raise ValueError("--output-root 不能与 --image-dir 相同。")
    if args.mask_dir.resolve() == args.output_root.resolve():
        raise ValueError("--output-root 不能与 --mask-dir 相同。")
    if args.group_regex is not None:
        re.compile(args.group_regex)


def list_files(
    root: Path,
    extensions: set[str],
    recursive: bool,
) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in extensions
    )


def normalized_stem(stem: str) -> str:
    return stem.casefold()


def build_mask_indices(
    mask_paths: list[Path],
    mask_root: Path,
) -> tuple[
    dict[tuple[str, str], list[Path]],
    dict[str, list[Path]],
]:
    by_relative_parent_and_stem: dict[
        tuple[str, str], list[Path]
    ] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)

    for path in mask_paths:
        relative = path.relative_to(mask_root)
        parent_key = relative.parent.as_posix().casefold()
        stem_key = normalized_stem(path.stem)
        by_relative_parent_and_stem[(parent_key, stem_key)].append(path)
        by_stem[stem_key].append(path)

    return by_relative_parent_and_stem, by_stem


def candidate_mask_stems(image_stem: str, mask_suffix: str) -> list[str]:
    if mask_suffix != "auto":
        return [normalized_stem(image_stem + mask_suffix)]

    return [
        normalized_stem(image_stem + suffix)
        for suffix in COMMON_MASK_SUFFIXES
    ]


def find_mask_for_image(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
    mask_suffix: str,
    by_parent_stem: dict[tuple[str, str], list[Path]],
    by_stem: dict[str, list[Path]],
) -> tuple[Path | None, str | None]:
    relative = image_path.relative_to(image_root)
    parent_key = relative.parent.as_posix().casefold()
    candidates = candidate_mask_stems(image_path.stem, mask_suffix)

    local_matches: list[Path] = []
    for stem in candidates:
        local_matches.extend(
            by_parent_stem.get((parent_key, stem), [])
        )

    local_matches = sorted(set(local_matches))
    if len(local_matches) == 1:
        return local_matches[0], None
    if len(local_matches) > 1:
        return None, (
            "同一相对目录下找到多个可能标签："
            + " | ".join(str(path) for path in local_matches)
        )

    global_matches: list[Path] = []
    for stem in candidates:
        global_matches.extend(by_stem.get(stem, []))

    global_matches = sorted(set(global_matches))
    if len(global_matches) == 1:
        return global_matches[0], None
    if len(global_matches) > 1:
        return None, (
            "跨目录找到多个同名标签，无法唯一配对："
            + " | ".join(str(path) for path in global_matches)
        )

    return None, "未找到同名或常见后缀标签。"


def read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    return int(image.shape[1]), int(image.shape[0])


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"无法读取标签图：{path}")
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = mask[:, :, :3]
    return mask


def color_to_instance_map(
    mask: np.ndarray,
    background_value: int,
) -> np.ndarray:
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError("彩色标签必须是3通道图像。")

    # OpenCV读取顺序为BGR。颜色本身只用于区分实例，不影响结果。
    flat = mask.reshape(-1, 3)
    unique_colors, inverse = np.unique(
        flat,
        axis=0,
        return_inverse=True,
    )

    background_color = np.asarray(
        [background_value] * 3,
        dtype=unique_colors.dtype,
    )
    labels_for_color = np.zeros(
        len(unique_colors),
        dtype=np.int32,
    )

    next_id = 1
    for index, color in enumerate(unique_colors):
        if np.array_equal(color, background_color):
            continue
        labels_for_color[index] = next_id
        next_id += 1

    return labels_for_color[inverse].reshape(mask.shape[:2])


def single_channel_to_instance_map(
    mask: np.ndarray,
    background_value: int,
    connectivity: int,
) -> tuple[np.ndarray, bool]:
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError("标签不是单通道图。")

    unique_values = np.unique(mask)
    foreground_values = [
        value
        for value in unique_values.tolist()
        if int(value) != int(background_value)
    ]

    # 只有一个非背景值时，视为二值标签，按连通域拆分实例。
    if len(foreground_values) <= 1:
        binary = (
            mask != np.asarray(background_value, dtype=mask.dtype)
        ).astype(np.uint8)
        count, components = cv2.connectedComponents(
            binary,
            connectivity=connectivity,
        )
        return components.astype(np.int32), True

    # 多个灰度值时，每个非背景值视为一个实例ID。
    result = np.zeros(mask.shape[:2], dtype=np.int32)
    next_id = 1
    for value in foreground_values:
        result[mask == value] = next_id
        next_id += 1

    return result, False


def normalize_instance_mask(
    raw_mask: np.ndarray,
    background_value: int,
    color_mode: str,
    connectivity: int,
) -> tuple[np.ndarray, str]:
    if raw_mask.ndim == 2 or (
        raw_mask.ndim == 3 and raw_mask.shape[2] == 1
    ):
        instance_map, was_binary = single_channel_to_instance_map(
            raw_mask,
            background_value,
            connectivity,
        )
        return (
            instance_map,
            "binary_components" if was_binary else "instance_ids",
        )

    if raw_mask.ndim == 3 and raw_mask.shape[2] == 3:
        if color_mode == "no":
            raise ValueError(
                "检测到3通道标签，但 --color-mask=no。"
            )

        # 三通道完全相同通常是灰度标签被保存成RGB。
        if np.array_equal(raw_mask[:, :, 0], raw_mask[:, :, 1]) and (
            np.array_equal(raw_mask[:, :, 1], raw_mask[:, :, 2])
        ):
            instance_map, was_binary = single_channel_to_instance_map(
                raw_mask[:, :, 0],
                background_value,
                connectivity,
            )
            return (
                instance_map,
                "binary_components" if was_binary else "instance_ids",
            )

        return (
            color_to_instance_map(raw_mask, background_value),
            "color_instances",
        )

    raise ValueError(
        f"不支持的标签维度：shape={raw_mask.shape}"
    )


def connected_components_of_instance(
    binary: np.ndarray,
    connectivity: int,
) -> tuple[int, np.ndarray]:
    count, labels = cv2.connectedComponents(
        binary.astype(np.uint8),
        connectivity=connectivity,
    )
    return count - 1, labels


def analyze_instance_map(
    instance_map: np.ndarray,
    min_area: int,
    split_disconnected: bool,
    connectivity: int,
) -> tuple[list[np.ndarray], dict[str, int]]:
    masks: list[np.ndarray] = []
    dropped_small = 0
    disconnected_id_count = 0
    split_component_count = 0

    ids = np.unique(instance_map)
    ids = ids[ids != 0]

    for instance_id in ids.tolist():
        binary = (instance_map == instance_id).astype(np.uint8)
        component_count, components = connected_components_of_instance(
            binary,
            connectivity,
        )

        if component_count > 1:
            disconnected_id_count += 1

        if split_disconnected and component_count > 1:
            for component_id in range(1, component_count + 1):
                component_mask = (
                    components == component_id
                ).astype(np.uint8)
                area = int(component_mask.sum())
                if area < min_area:
                    dropped_small += 1
                    continue
                masks.append(component_mask)
                split_component_count += 1
        else:
            area = int(binary.sum())
            if area < min_area:
                dropped_small += 1
                continue
            masks.append(binary)

    return masks, {
        "raw_instance_count": int(len(ids)),
        "kept_instance_count": int(len(masks)),
        "dropped_small_count": int(dropped_small),
        "disconnected_id_count": int(disconnected_id_count),
        "split_component_count": int(split_component_count),
    }


def mask_to_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8))
    )
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(value) for value in rle["size"]],
        "counts": counts,
    }


def derive_group_key(
    relative_path: Path,
    mode: str,
    regex: str | None,
) -> str:
    stem = relative_path.stem

    if regex:
        match = re.search(regex, stem)
        if not match:
            return stem
        if match.lastindex:
            return match.group(1)
        return match.group(0)

    if mode == "none":
        return relative_path.as_posix()

    if mode == "parent":
        parent = relative_path.parent.as_posix()
        return parent if parent != "." else stem

    # auto_tile：移除常见图块坐标后缀，避免同一来源图像泄漏。
    patterns = (
        r"__tile_y\d+_x\d+.*$",
        r"__x\d+_y\d+.*$",
        r"_tile_y\d+_x\d+.*$",
        r"_tile_x\d+_y\d+.*$",
        r"__src\d+__x\d+_y\d+.*$",
    )
    result = stem
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    return result or stem


def parse_bins(text: str) -> list[int]:
    bins = sorted(
        {int(value.strip()) for value in text.split(",") if value.strip()}
    )
    if len(bins) < 2:
        raise ValueError("--stratify-bins 至少需要两个边界。")
    return bins


def bin_index(value: int, bins: list[int]) -> int:
    for index in range(len(bins) - 1):
        if bins[index] <= value < bins[index + 1]:
            return index
    return len(bins) - 2


def deterministic_group_tiebreak(group_key: str, seed: int) -> str:
    return hashlib.sha1(
        f"{seed}:{group_key}".encode("utf-8")
    ).hexdigest()


def allocate_groups_to_splits(
    records: list[PairRecord],
    bins: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    groups: dict[str, list[PairRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_key].append(record)

    grouped_by_bin: dict[int, list[tuple[str, list[PairRecord]]]] = (
        defaultdict(list)
    )
    for group_key, members in groups.items():
        representative_count = int(
            round(np.mean([item.kept_instance_count for item in members]))
        )
        grouped_by_bin[bin_index(representative_count, bins)].append(
            (group_key, members)
        )

    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    global_target = {
        split: len(records) * ratio
        for split, ratio in ratios.items()
    }
    global_count = Counter()

    rng = random.Random(seed)

    for current_bin in sorted(grouped_by_bin):
        group_items = grouped_by_bin[current_bin]
        rng.shuffle(group_items)

        # 大组优先分配，减少最终比例偏差。
        group_items.sort(
            key=lambda item: (
                -len(item[1]),
                deterministic_group_tiebreak(item[0], seed),
            )
        )

        bin_total = sum(len(members) for _, members in group_items)
        bin_target = {
            split: bin_total * ratio
            for split, ratio in ratios.items()
        }
        bin_count = Counter()

        for _, members in group_items:
            group_size = len(members)

            def score(split: str) -> tuple[float, float, int]:
                # 首要目标：当前实例数分层内比例接近目标；
                # 次要目标：全局比例接近目标。
                bin_deficit = (
                    bin_target[split] - bin_count[split]
                )
                global_deficit = (
                    global_target[split] - global_count[split]
                )
                # 比例为0的split禁止接收样本。
                forbidden = 1 if ratios[split] <= 0 else 0
                return (
                    -forbidden,
                    bin_deficit,
                    global_deficit,
                )

            candidates = [
                split for split in ("train", "val", "test")
                if ratios[split] > 0
            ]
            chosen = max(candidates, key=score)

            for record in members:
                record.split = chosen

            bin_count[chosen] += group_size
            global_count[chosen] += group_size

    # 极小数据集的兜底：若某个非零比例split为空，尝试移动一个独立组。
    for required_split, ratio in ratios.items():
        if ratio <= 0 or global_count[required_split] > 0:
            continue

        donors = sorted(
            (
                split for split in ("train", "val", "test")
                if global_count[split] > 1
            ),
            key=lambda split: global_count[split],
            reverse=True,
        )
        if not donors:
            continue

        donor = donors[0]
        donor_groups = defaultdict(list)
        for record in records:
            if record.split == donor:
                donor_groups[record.group_key].append(record)

        smallest_group = min(
            donor_groups.values(),
            key=len,
        )
        for record in smallest_group:
            record.split = required_split
        global_count[donor] -= len(smallest_group)
        global_count[required_split] += len(smallest_group)


def output_relative_image_path(
    relative_source: Path,
) -> Path:
    # 保留相对目录，防止不同子目录中同名图片冲突。
    return relative_source


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_dir = args.image_dir.expanduser().resolve()
    mask_dir = args.mask_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    bins = parse_bins(args.stratify_bins)

    image_paths = list_files(
        image_dir,
        IMAGE_EXTENSIONS,
        args.recursive,
    )
    mask_paths = list_files(
        mask_dir,
        MASK_EXTENSIONS,
        args.recursive,
    )

    if not image_paths:
        raise RuntimeError(f"原图目录中没有找到图像：{image_dir}")
    if not mask_paths:
        raise RuntimeError(f"标签目录中没有找到标签图：{mask_dir}")

    if output_root.exists() and not args.dry_run:
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{output_root}\n"
                "确认可以删除后添加 --overwrite，或更换输出目录。"
            )
        shutil.rmtree(output_root)

    by_parent_stem, by_stem = build_mask_indices(
        mask_paths,
        mask_dir,
    )

    issues: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []
    records: list[PairRecord] = []
    parsed_masks: dict[Path, tuple[list[np.ndarray], str, dict[str, int]]] = {}

    used_masks: set[Path] = set()

    print("=" * 72)
    print("原图 + 实例标签图 → COCO实例分割数据集")
    print(f"原图目录：{image_dir}")
    print(f"标签目录：{mask_dir}")
    print(f"输出目录：{output_root}")
    print(f"原图数量：{len(image_paths)}")
    print(f"标签数量：{len(mask_paths)}")
    print("=" * 72)

    for index, image_path in enumerate(image_paths, start=1):
        relative_image = image_path.relative_to(image_dir)
        mask_path, pairing_error = find_mask_for_image(
            image_path,
            image_dir,
            mask_dir,
            args.mask_suffix,
            by_parent_stem,
            by_stem,
        )

        if mask_path is None:
            issues.append(
                {
                    "image": relative_image.as_posix(),
                    "mask": "",
                    "issue": "pairing_error",
                    "details": pairing_error or "",
                }
            )
            continue

        used_masks.add(mask_path)
        relative_mask = mask_path.relative_to(mask_dir)

        try:
            image_w, image_h = read_image_size(image_path)
            raw_mask = read_mask(mask_path)
        except Exception as exc:
            issues.append(
                {
                    "image": relative_image.as_posix(),
                    "mask": relative_mask.as_posix(),
                    "issue": "read_error",
                    "details": repr(exc),
                }
            )
            continue

        mask_h, mask_w = raw_mask.shape[:2]
        if image_w != mask_w or image_h != mask_h:
            issues.append(
                {
                    "image": relative_image.as_posix(),
                    "mask": relative_mask.as_posix(),
                    "issue": "size_mismatch",
                    "details": (
                        f"image={image_w}x{image_h}; "
                        f"mask={mask_w}x{mask_h}"
                    ),
                }
            )
            continue

        try:
            instance_map, mask_mode = normalize_instance_mask(
                raw_mask,
                args.background_value,
                args.color_mask,
                args.binary_connectivity,
            )
            masks, mask_stats = analyze_instance_map(
                instance_map,
                args.min_area,
                args.split_disconnected,
                args.binary_connectivity,
            )
        except Exception as exc:
            issues.append(
                {
                    "image": relative_image.as_posix(),
                    "mask": relative_mask.as_posix(),
                    "issue": "mask_parse_error",
                    "details": repr(exc),
                }
            )
            continue

        if mask_stats["disconnected_id_count"] > 0:
            issues.append(
                {
                    "image": relative_image.as_posix(),
                    "mask": relative_mask.as_posix(),
                    "issue": "disconnected_instance_ids",
                    "details": (
                        f"{mask_stats['disconnected_id_count']}个实例ID"
                        "包含多个不连通区域；"
                        + (
                            "已按连通域拆分。"
                            if args.split_disconnected
                            else "当前仍作为同一实例保留。"
                        )
                    ),
                }
            )

        empty = len(masks) == 0
        if empty and not args.keep_empty:
            issues.append(
                {
                    "image": relative_image.as_posix(),
                    "mask": relative_mask.as_posix(),
                    "issue": "empty_mask_skipped",
                    "details": "标签中没有达到最小面积的有效实例。",
                }
            )
            continue

        group_key = derive_group_key(
            relative_image,
            args.group_mode,
            args.group_regex,
        )

        record = PairRecord(
            image_path=image_path,
            mask_path=mask_path,
            relative_image_path=relative_image,
            relative_mask_path=relative_mask,
            image_width=image_w,
            image_height=image_h,
            instance_count=mask_stats["raw_instance_count"],
            kept_instance_count=mask_stats["kept_instance_count"],
            dropped_small_count=mask_stats["dropped_small_count"],
            empty_mask=empty,
            group_key=group_key,
        )
        records.append(record)
        parsed_masks[image_path] = (masks, mask_mode, mask_stats)

        statistics_rows.append(
            {
                "image": relative_image.as_posix(),
                "mask": relative_mask.as_posix(),
                "width": image_w,
                "height": image_h,
                "mask_mode": mask_mode,
                "raw_instance_count": mask_stats["raw_instance_count"],
                "kept_instance_count": mask_stats["kept_instance_count"],
                "dropped_small_count": mask_stats["dropped_small_count"],
                "disconnected_id_count": (
                    mask_stats["disconnected_id_count"]
                ),
                "group_key": group_key,
                "split": "",
            }
        )

        if index % 20 == 0 or index == len(image_paths):
            print(
                f"[检查] {index}/{len(image_paths)}；"
                f"有效配对={len(records)}；问题={len(issues)}"
            )

    unused_masks = sorted(set(mask_paths) - used_masks)
    for mask_path in unused_masks:
        issues.append(
            {
                "image": "",
                "mask": mask_path.relative_to(mask_dir).as_posix(),
                "issue": "unused_mask",
                "details": "该标签图未匹配到原图。",
            }
        )

    if not records:
        raise RuntimeError(
            "没有可转换的有效图像—标签配对，请检查文件名和标签格式。"
        )

    allocate_groups_to_splits(
        records,
        bins,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    split_by_image = {
        record.relative_image_path.as_posix(): record.split
        for record in records
    }
    for row in statistics_rows:
        row["split"] = split_by_image.get(row["image"], "")

    if args.dry_run:
        print("\n[DRY RUN] 仅完成检查，不写出数据集。")
        print(
            json.dumps(
                {
                    "valid_pairs": len(records),
                    "issues": len(issues),
                    "split_counts": dict(
                        Counter(record.split for record in records)
                    ),
                    "instance_counts": {
                        "min": min(
                            record.kept_instance_count
                            for record in records
                        ),
                        "max": max(
                            record.kept_instance_count
                            for record in records
                        ),
                        "mean": float(
                            np.mean(
                                [
                                    record.kept_instance_count
                                    for record in records
                                ]
                            )
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)

    category_prompt = args.prompt or args.category_name
    category = {
        "id": 1,
        "name": args.category_name,
        "supercategory": args.category_name,
        "prompt": category_prompt,
    }

    split_coco: dict[str, dict[str, Any]] = {}
    next_image_id = Counter()
    next_annotation_id = Counter()

    for split in ("train", "val", "test"):
        split_dir = output_root / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        if args.copy_masks:
            (split_dir / "labels").mkdir(
                parents=True,
                exist_ok=True,
            )

        split_coco[split] = {
            "info": {
                "description": (
                    "COCO instance segmentation dataset converted from "
                    "paired images and instance label masks."
                ),
                "category_name": args.category_name,
                "prompt": category_prompt,
                "seed": args.seed,
            },
            "licenses": [],
            "categories": [category],
            "images": [],
            "annotations": [],
        }
        next_image_id[split] = 1
        next_annotation_id[split] = 1

    for index, record in enumerate(records, start=1):
        split = record.split
        coco = split_coco[split]

        output_relative = output_relative_image_path(
            record.relative_image_path
        )
        destination_image = (
            output_root / split / "images" / output_relative
        )
        destination_image.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(record.image_path, destination_image)

        if args.copy_masks:
            destination_mask = (
                output_root
                / split
                / "labels"
                / record.relative_mask_path
            )
            destination_mask.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(record.mask_path, destination_mask)

        image_id = int(next_image_id[split])
        next_image_id[split] += 1

        coco["images"].append(
            {
                "id": image_id,
                "file_name": output_relative.as_posix(),
                "width": int(record.image_width),
                "height": int(record.image_height),
                "source_image": record.relative_image_path.as_posix(),
                "source_mask": record.relative_mask_path.as_posix(),
                "group_key": record.group_key,
            }
        )

        masks, mask_mode, _ = parsed_masks[record.image_path]

        for source_instance_index, binary_mask in enumerate(
            masks,
            start=1,
        ):
            rle = mask_to_rle(binary_mask)
            area = float(mask_utils.area(rle))
            bbox = [
                float(value)
                for value in mask_utils.toBbox(rle).tolist()
            ]

            if area < args.min_area:
                # 正常情况下前面已经过滤；保留二次审计。
                continue
            if bbox[2] <= 0 or bbox[3] <= 0:
                issues.append(
                    {
                        "image": (
                            record.relative_image_path.as_posix()
                        ),
                        "mask": (
                            record.relative_mask_path.as_posix()
                        ),
                        "issue": "nonpositive_bbox_skipped",
                        "details": (
                            f"source_instance_index="
                            f"{source_instance_index}; bbox={bbox}"
                        ),
                    }
                )
                continue

            annotation_id = int(next_annotation_id[split])
            next_annotation_id[split] += 1

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "source_instance_index": source_instance_index,
                    "source_mask_mode": mask_mode,
                }
            )

        if index % 20 == 0 or index == len(records):
            print(
                f"[写出] {index}/{len(records)}；"
                f"当前split={split}"
            )

    for split, coco in split_coco.items():
        save_json(
            output_root / split / "_annotations.coco.json",
            coco,
        )

    write_csv(
        output_root / "image_instance_statistics.csv",
        statistics_rows,
        [
            "image",
            "mask",
            "width",
            "height",
            "mask_mode",
            "raw_instance_count",
            "kept_instance_count",
            "dropped_small_count",
            "disconnected_id_count",
            "group_key",
            "split",
        ],
    )
    write_csv(
        output_root / "conversion_issues.csv",
        issues,
        ["image", "mask", "issue", "details"],
    )

    split_counts = Counter(record.split for record in records)
    instance_counts = [
        record.kept_instance_count for record in records
    ]
    group_split_map: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_split_map[record.group_key].add(record.split)
    leaked_groups = {
        key: sorted(values)
        for key, values in group_split_map.items()
        if len(values) > 1
    }

    summary = {
        "input": {
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "image_file_count": len(image_paths),
            "mask_file_count": len(mask_paths),
        },
        "parameters": {
            "category_name": args.category_name,
            "prompt": category_prompt,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "min_area": args.min_area,
            "background_value": args.background_value,
            "split_disconnected": args.split_disconnected,
            "group_mode": args.group_mode,
            "group_regex": args.group_regex,
            "stratify_bins": bins,
        },
        "results": {
            "valid_pair_count": len(records),
            "unused_mask_count": len(unused_masks),
            "issue_count": len(issues),
            "issue_breakdown": dict(
                Counter(row["issue"] for row in issues)
            ),
            "split_image_counts": dict(split_counts),
            "split_annotation_counts": {
                split: len(coco["annotations"])
                for split, coco in split_coco.items()
            },
            "instance_count_statistics": {
                "minimum": int(min(instance_counts)),
                "maximum": int(max(instance_counts)),
                "mean": float(np.mean(instance_counts)),
                "median": float(np.median(instance_counts)),
                "p90": float(np.percentile(instance_counts, 90)),
                "p95": float(np.percentile(instance_counts, 95)),
                "p99": float(np.percentile(instance_counts, 99)),
            },
            "group_leakage_count": len(leaked_groups),
            "group_leakage": leaked_groups,
        },
        "outputs": {
            "root": str(output_root),
            "summary": str(
                output_root / "conversion_summary.json"
            ),
            "statistics_csv": str(
                output_root / "image_instance_statistics.csv"
            ),
            "issues_csv": str(
                output_root / "conversion_issues.csv"
            ),
        },
    }
    save_json(
        output_root / "conversion_summary.json",
        summary,
    )

    print("\n" + "=" * 72)
    print("[转换完成]")
    print(f"有效图像—标签配对：{len(records)}")
    print(f"划分数量：{dict(split_counts)}")
    print(
        "COCO实例数量："
        + str(
            {
                split: len(coco["annotations"])
                for split, coco in split_coco.items()
            }
        )
    )
    print(
        f"单图实例数：min={min(instance_counts)}, "
        f"mean={np.mean(instance_counts):.2f}, "
        f"max={max(instance_counts)}"
    )
    print(f"问题记录：{len(issues)}")
    print(f"分组泄漏：{len(leaked_groups)}")
    print(
        f"汇总文件：{output_root / 'conversion_summary.json'}"
    )
    print("=" * 72)

    if leaked_groups:
        raise RuntimeError(
            "检测到同一分组被分入多个split，请检查 --group-mode "
            "或 --group-regex。"
        )


if __name__ == "__main__":
    main()
