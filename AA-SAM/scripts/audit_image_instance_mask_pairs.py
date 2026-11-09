#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查“原图文件夹 + 实例标签图文件夹”，并把有问题的图像/标签复制到审计目录。

适用标签格式
------------
1. 单通道实例标签：
       0 = 背景
       1..N = 不同实例
   推荐 uint16 PNG。
2. 二值标签：
       0 = 背景
       1 或 255 = 前景
   脚本按连通域估算实例数。
3. RGB 彩色标签：
       每种非背景颜色视为一个实例。

检查项目
--------
- 原图缺标签、标签缺原图；
- 文件无法读取；
- 原图与标签尺寸不一致；
- 标签为空；
- 标签位深是否适合超过255个实例；
- 实例数量、最大标签ID、唯一值数量；
- 小面积实例；
- 同一实例ID包含多个不连通区域；
- 二值标签中大面积黏连区域；
- 标签ID不连续；
- 标签为3通道灰度伪彩或真正彩色实例图；
- 疑似uint8编号溢出；
- 文件名配对歧义。

输出结构
--------
audit_output/
├─ problematic_pairs/
│  ├─ images/
│  └─ labels/
├─ by_issue/
│  ├─ missing_mask/
│  ├─ orphan_mask/
│  ├─ unreadable_image/
│  ├─ unreadable_mask/
│  ├─ size_mismatch/
│  ├─ empty_mask/
│  ├─ uint8_overflow_risk/
│  ├─ disconnected_instance_ids/
│  ├─ small_instances/
│  ├─ binary_merged_regions/
│  ├─ non_contiguous_ids/
│  └─ ...
├─ audit_report.csv
├─ issue_details.csv
├─ audit_summary.json
└─ problematic_pairs.txt

说明
----
默认只复制，不删除原始文件。
若同一文件有多个问题，只在 problematic_pairs 中复制一份，
同时可在 by_issue 的多个问题目录中分别出现。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
MASK_EXTENSIONS = {
    ".png", ".bmp", ".tif", ".tiff"
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
class PairAudit:
    image_path: Path | None
    mask_path: Path | None
    image_relative: str
    mask_relative: str
    image_width: int | None = None
    image_height: int | None = None
    mask_width: int | None = None
    mask_height: int | None = None
    image_dtype: str = ""
    mask_dtype: str = ""
    mask_shape: str = ""
    mask_mode: str = ""
    raw_unique_count: int | None = None
    foreground_unique_count: int | None = None
    max_label_value: int | None = None
    min_label_value: int | None = None
    estimated_instance_count: int | None = None
    connected_component_count: int | None = None
    small_instance_count: int = 0
    disconnected_id_count: int = 0
    non_contiguous_id_count: int = 0
    largest_instance_area: int = 0
    median_instance_area: float = 0.0
    issues: set[str] = field(default_factory=set)
    issue_details: list[tuple[str, str]] = field(default_factory=list)

    def add_issue(self, issue: str, details: str) -> None:
        self.issues.add(issue)
        self.issue_details.append((issue, details))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查原图及实例标签，并复制所有问题样本。"
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
        "--output-dir",
        type=Path,
        required=True,
        help="问题样本与审计报告输出目录。",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录。",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default="auto",
        help=(
            '标签文件名后缀。默认 "auto"，自动尝试同名、_mask、'
            "_label、_instances等；也可指定如 _label。"
        ),
    )
    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
        help="背景值。默认：0",
    )
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=16,
        help="小实例面积阈值。默认：16像素。",
    )
    parser.add_argument(
        "--max-small-instance-ratio",
        type=float,
        default=0.01,
        help=(
            "小实例占比超过该值时判为问题。默认：0.01；"
            "设为0表示只要存在小实例就报告。"
        ),
    )
    parser.add_argument(
        "--binary-large-region-area",
        type=int,
        default=20000,
        help=(
            "二值标签中单个连通域超过该面积时，标记为疑似黏连。"
            "默认：20000像素。"
        ),
    )
    parser.add_argument(
        "--binary-connectivity",
        choices=(4, 8),
        type=int,
        default=8,
        help="连通域邻接方式。默认：8。",
    )
    parser.add_argument(
        "--max-disconnected-components",
        type=int,
        default=1,
        help=(
            "同一实例ID允许的最大连通区域数。默认：1；"
            "超过则标记 disconnected_instance_ids。"
        ),
    )
    parser.add_argument(
        "--expected-min-instances",
        type=int,
        default=0,
        help=(
            "若已知完整原图通常至少包含多少实例，可设置该阈值。"
            "低于该值时标记 too_few_instances。默认0不检查。"
        ),
    )
    parser.add_argument(
        "--expected-max-instances",
        type=int,
        default=0,
        help=(
            "若估计实例数超过该阈值则标记 too_many_instances。"
            "默认0不检查。"
        ),
    )
    parser.add_argument(
        "--copy-by-issue",
        action="store_true",
        help=(
            "除统一problematic_pairs目录外，再按每种问题分别复制。"
        ),
    )
    parser.add_argument(
        "--copy-good",
        action="store_true",
        help="同时将无问题样本复制到 good_pairs 目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录已存在时先删除后重建。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image_dir.is_dir():
        raise NotADirectoryError(args.image_dir)
    if not args.mask_dir.is_dir():
        raise NotADirectoryError(args.mask_dir)
    if args.min_instance_area <= 0:
        raise ValueError("--min-instance-area 必须大于0。")
    if not (0.0 <= args.max_small_instance_ratio <= 1.0):
        raise ValueError("--max-small-instance-ratio 必须位于[0,1]。")
    if args.binary_large_region_area <= 0:
        raise ValueError("--binary-large-region-area 必须大于0。")
    if args.max_disconnected_components < 1:
        raise ValueError("--max-disconnected-components 必须至少为1。")
    if args.output_dir.resolve() in {
        args.image_dir.resolve(),
        args.mask_dir.resolve(),
    }:
        raise ValueError("--output-dir 不能等于输入目录。")


def list_files(root: Path, extensions: set[str], recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in extensions
    )


def candidate_stems(image_stem: str, mask_suffix: str) -> list[str]:
    if mask_suffix != "auto":
        return [(image_stem + mask_suffix).casefold()]
    return [
        (image_stem + suffix).casefold()
        for suffix in COMMON_MASK_SUFFIXES
    ]


def build_mask_index(
    mask_paths: list[Path],
    mask_root: Path,
) -> tuple[dict[tuple[str, str], list[Path]], dict[str, list[Path]]]:
    local_index: dict[tuple[str, str], list[Path]] = defaultdict(list)
    global_index: dict[str, list[Path]] = defaultdict(list)

    for path in mask_paths:
        rel = path.relative_to(mask_root)
        parent = rel.parent.as_posix().casefold()
        stem = path.stem.casefold()
        local_index[(parent, stem)].append(path)
        global_index[stem].append(path)

    return local_index, global_index


def find_matching_mask(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
    mask_suffix: str,
    local_index: dict[tuple[str, str], list[Path]],
    global_index: dict[str, list[Path]],
) -> tuple[Path | None, str | None]:
    rel = image_path.relative_to(image_root)
    parent = rel.parent.as_posix().casefold()
    stems = candidate_stems(image_path.stem, mask_suffix)

    local_matches: list[Path] = []
    for stem in stems:
        local_matches.extend(local_index.get((parent, stem), []))
    local_matches = sorted(set(local_matches))

    if len(local_matches) == 1:
        return local_matches[0], None
    if len(local_matches) > 1:
        return None, "同一相对目录内存在多个候选标签：" + " | ".join(
            str(p) for p in local_matches
        )

    global_matches: list[Path] = []
    for stem in stems:
        global_matches.extend(global_index.get(stem, []))
    global_matches = sorted(set(global_matches))

    if len(global_matches) == 1:
        return global_matches[0], None
    if len(global_matches) > 1:
        return None, "跨目录存在多个同名候选标签：" + " | ".join(
            str(p) for p in global_matches
        )
    return None, "未找到同名或常见后缀标签。"


def safe_read(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def is_three_channel_grayscale(mask: np.ndarray) -> bool:
    return (
        mask.ndim == 3
        and mask.shape[2] >= 3
        and np.array_equal(mask[:, :, 0], mask[:, :, 1])
        and np.array_equal(mask[:, :, 1], mask[:, :, 2])
    )


def color_instance_map(
    mask: np.ndarray,
    background_value: int,
) -> tuple[np.ndarray, int]:
    bgr = mask[:, :, :3]
    flat = bgr.reshape(-1, 3)
    colors, inverse = np.unique(flat, axis=0, return_inverse=True)

    bg = np.asarray([background_value] * 3, dtype=colors.dtype)
    remap = np.zeros(len(colors), dtype=np.int32)
    next_id = 1
    for i, color in enumerate(colors):
        if np.array_equal(color, bg):
            continue
        remap[i] = next_id
        next_id += 1
    result = remap[inverse].reshape(bgr.shape[:2])
    return result, next_id - 1


def single_channel_instance_map(
    mask: np.ndarray,
    background_value: int,
    connectivity: int,
) -> tuple[np.ndarray, str]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    values = np.unique(mask)
    fg_values = [v for v in values.tolist() if int(v) != background_value]

    if len(fg_values) <= 1:
        binary = (mask != background_value).astype(np.uint8)
        count, labels = cv2.connectedComponents(
            binary,
            connectivity=connectivity,
        )
        return labels.astype(np.int32), "binary_components"

    result = np.zeros(mask.shape, dtype=np.int32)
    for new_id, value in enumerate(fg_values, start=1):
        result[mask == value] = new_id
    return result, "instance_ids"


def normalize_mask(
    mask: np.ndarray,
    background_value: int,
    connectivity: int,
) -> tuple[np.ndarray, str]:
    if mask.ndim == 2:
        return single_channel_instance_map(
            mask, background_value, connectivity
        )
    if mask.ndim == 3 and mask.shape[2] == 1:
        return single_channel_instance_map(
            mask[:, :, 0], background_value, connectivity
        )
    if mask.ndim == 3 and mask.shape[2] in (3, 4):
        if is_three_channel_grayscale(mask):
            return single_channel_instance_map(
                mask[:, :, 0], background_value, connectivity
            )
        result, _ = color_instance_map(mask, background_value)
        return result, "color_instances"
    raise ValueError(f"不支持的标签shape：{mask.shape}")


def analyze_instances(
    instance_map: np.ndarray,
    mode: str,
    min_area: int,
    connectivity: int,
    max_disconnected_components: int,
    binary_large_region_area: int,
) -> dict[str, Any]:
    ids = np.unique(instance_map)
    ids = ids[ids != 0]
    areas: list[int] = []
    small_ids: list[int] = []
    disconnected: dict[int, int] = {}
    binary_large_ids: list[int] = []

    for instance_id in ids.tolist():
        binary = (instance_map == instance_id).astype(np.uint8)
        area = int(binary.sum())
        areas.append(area)

        if area < min_area:
            small_ids.append(int(instance_id))

        count, _ = cv2.connectedComponents(
            binary,
            connectivity=connectivity,
        )
        components = int(count - 1)
        if components > max_disconnected_components:
            disconnected[int(instance_id)] = components

        if mode == "binary_components" and area >= binary_large_region_area:
            binary_large_ids.append(int(instance_id))

    return {
        "instance_count": int(len(ids)),
        "areas": areas,
        "small_ids": small_ids,
        "disconnected": disconnected,
        "binary_large_ids": binary_large_ids,
        "largest_area": int(max(areas) if areas else 0),
        "median_area": float(np.median(areas) if areas else 0.0),
    }


def raw_label_stats(
    mask: np.ndarray,
    background_value: int,
) -> dict[str, Any]:
    base = mask
    if mask.ndim == 3 and is_three_channel_grayscale(mask):
        base = mask[:, :, 0]

    if base.ndim == 2:
        unique = np.unique(base)
        fg = unique[unique != background_value]
        min_value = int(unique.min()) if unique.size else 0
        max_value = int(unique.max()) if unique.size else 0

        non_contiguous = 0
        if fg.size > 1 and np.issubdtype(base.dtype, np.integer):
            expected = max_value - int(fg.min()) + 1
            non_contiguous = max(0, expected - int(fg.size))

        return {
            "raw_unique_count": int(unique.size),
            "foreground_unique_count": int(fg.size),
            "min_label": min_value,
            "max_label": max_value,
            "non_contiguous": int(non_contiguous),
        }

    colors = np.unique(mask[:, :, :3].reshape(-1, 3), axis=0)
    bg = np.asarray([background_value] * 3, dtype=colors.dtype)
    fg_count = sum(0 if np.array_equal(c, bg) else 1 for c in colors)
    return {
        "raw_unique_count": int(len(colors)),
        "foreground_unique_count": int(fg_count),
        "min_label": 0,
        "max_label": int(fg_count),
        "non_contiguous": 0,
    }


def audit_pair(
    image_path: Path,
    mask_path: Path,
    image_root: Path,
    mask_root: Path,
    args: argparse.Namespace,
) -> PairAudit:
    record = PairAudit(
        image_path=image_path,
        mask_path=mask_path,
        image_relative=image_path.relative_to(image_root).as_posix(),
        mask_relative=mask_path.relative_to(mask_root).as_posix(),
    )

    image = safe_read(image_path)
    if image is None:
        record.add_issue("unreadable_image", "OpenCV无法读取原图。")
        return record

    mask = safe_read(mask_path)
    if mask is None:
        record.add_issue("unreadable_mask", "OpenCV无法读取标签图。")
        return record

    record.image_width = int(image.shape[1])
    record.image_height = int(image.shape[0])
    record.mask_width = int(mask.shape[1])
    record.mask_height = int(mask.shape[0])
    record.image_dtype = str(image.dtype)
    record.mask_dtype = str(mask.dtype)
    record.mask_shape = str(tuple(mask.shape))

    if (
        record.image_width != record.mask_width
        or record.image_height != record.mask_height
    ):
        record.add_issue(
            "size_mismatch",
            (
                f"image={record.image_width}x{record.image_height}; "
                f"mask={record.mask_width}x{record.mask_height}"
            ),
        )
        return record

    stats = raw_label_stats(mask, args.background_value)
    record.raw_unique_count = stats["raw_unique_count"]
    record.foreground_unique_count = stats["foreground_unique_count"]
    record.min_label_value = stats["min_label"]
    record.max_label_value = stats["max_label"]
    record.non_contiguous_id_count = stats["non_contiguous"]

    if mask.ndim == 3 and is_three_channel_grayscale(mask):
        record.add_issue(
            "three_channel_grayscale_mask",
            "标签为3通道但三个通道完全相同，建议转为单通道PNG。",
        )

    try:
        instance_map, mode = normalize_mask(
            mask,
            args.background_value,
            args.binary_connectivity,
        )
    except Exception as exc:
        record.add_issue("mask_parse_error", repr(exc))
        return record

    record.mask_mode = mode
    instance_stats = analyze_instances(
        instance_map,
        mode,
        args.min_instance_area,
        args.binary_connectivity,
        args.max_disconnected_components,
        args.binary_large_region_area,
    )

    record.estimated_instance_count = instance_stats["instance_count"]
    record.connected_component_count = instance_stats["instance_count"]
    record.small_instance_count = len(instance_stats["small_ids"])
    record.disconnected_id_count = len(instance_stats["disconnected"])
    record.largest_instance_area = instance_stats["largest_area"]
    record.median_instance_area = instance_stats["median_area"]

    if record.estimated_instance_count == 0:
        record.add_issue("empty_mask", "标签中没有有效前景实例。")

    if args.expected_min_instances > 0 and (
        record.estimated_instance_count < args.expected_min_instances
    ):
        record.add_issue(
            "too_few_instances",
            (
                f"估计实例数={record.estimated_instance_count}，"
                f"低于阈值={args.expected_min_instances}。"
            ),
        )

    if args.expected_max_instances > 0 and (
        record.estimated_instance_count > args.expected_max_instances
    ):
        record.add_issue(
            "too_many_instances",
            (
                f"估计实例数={record.estimated_instance_count}，"
                f"超过阈值={args.expected_max_instances}。"
            ),
        )

    # 对多ID标签判断ID不连续；二值连通域模式不做该判断。
    if (
        mode == "instance_ids"
        and record.non_contiguous_id_count > 0
    ):
        record.add_issue(
            "non_contiguous_ids",
            (
                f"标签ID中缺少约{record.non_contiguous_id_count}个编号；"
                "这不一定错误，但可能表示实例被删除或编号发生异常。"
            ),
        )

    if record.small_instance_count > 0:
        denominator = max(record.estimated_instance_count, 1)
        ratio = record.small_instance_count / denominator
        if ratio > args.max_small_instance_ratio:
            record.add_issue(
                "small_instances",
                (
                    f"面积<{args.min_instance_area}像素的实例="
                    f"{record.small_instance_count}，占比={ratio:.4f}；"
                    f"示例ID={instance_stats['small_ids'][:20]}"
                ),
            )

    if record.disconnected_id_count > 0:
        sample = list(instance_stats["disconnected"].items())[:20]
        record.add_issue(
            "disconnected_instance_ids",
            (
                f"{record.disconnected_id_count}个实例ID含多个不连通区域；"
                f"示例(ID:区域数)={sample}"
            ),
        )

    if instance_stats["binary_large_ids"]:
        record.add_issue(
            "binary_merged_regions",
            (
                "二值标签中存在超大连通域，可能有多个相邻颗粒黏连；"
                f"面积阈值={args.binary_large_region_area}，"
                f"示例连通域ID={instance_stats['binary_large_ids'][:20]}"
            ),
        )

    # uint8风险：实例ID标签的前景唯一值达到255附近，或最大值=255且
    # 完整图通常预计实例很多。二值模式中的255不视为溢出。
    if mode == "instance_ids" and mask.dtype == np.uint8:
        max_label = record.max_label_value or 0
        fg_unique = record.foreground_unique_count or 0
        if max_label >= 250 or fg_unique >= 250:
            record.add_issue(
                "uint8_overflow_risk",
                (
                    f"标签dtype=uint8，max={max_label}，"
                    f"非背景唯一值={fg_unique}。若真实实例数>255，"
                    "实例编号可能已溢出或合并，应重新导出uint16标签。"
                ),
            )

    # 对RGB彩色实例图做信息提示；不是必然错误。
    if mode == "color_instances":
        record.add_issue(
            "color_instance_mask",
            (
                f"检测到彩色实例标签，非背景颜色数="
                f"{record.estimated_instance_count}。请确认每个实例颜色唯一。"
            ),
        )

    return record


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def issue_safe_name(issue: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", issue)


def copy_pair_to_folder(
    record: PairAudit,
    root: Path,
    image_root: Path,
    mask_root: Path,
) -> None:
    if record.image_path is not None and record.image_path.exists():
        safe_copy(
            record.image_path,
            root / "images" / record.image_path.relative_to(image_root),
        )
    if record.mask_path is not None and record.mask_path.exists():
        safe_copy(
            record.mask_path,
            root / "labels" / record.mask_path.relative_to(mask_root),
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_root = args.image_dir.expanduser().resolve()
    mask_root = args.mask_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{output_root}\n"
                "请更换目录，或确认后添加 --overwrite。"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = list_files(
        image_root, IMAGE_EXTENSIONS, args.recursive
    )
    mask_paths = list_files(
        mask_root, MASK_EXTENSIONS, args.recursive
    )

    local_index, global_index = build_mask_index(mask_paths, mask_root)
    used_masks: set[Path] = set()
    audits: list[PairAudit] = []

    print("=" * 72)
    print("原图与实例标签审计")
    print(f"原图目录：{image_root}")
    print(f"标签目录：{mask_root}")
    print(f"输出目录：{output_root}")
    print(f"原图数量：{len(image_paths)}")
    print(f"标签数量：{len(mask_paths)}")
    print("=" * 72)

    for index, image_path in enumerate(image_paths, start=1):
        mask_path, error = find_matching_mask(
            image_path,
            image_root,
            mask_root,
            args.mask_suffix,
            local_index,
            global_index,
        )

        if mask_path is None:
            record = PairAudit(
                image_path=image_path,
                mask_path=None,
                image_relative=image_path.relative_to(image_root).as_posix(),
                mask_relative="",
            )
            issue = (
                "ambiguous_mask_pairing"
                if error and "多个" in error
                else "missing_mask"
            )
            record.add_issue(issue, error or "")
            audits.append(record)
        else:
            used_masks.add(mask_path)
            audits.append(
                audit_pair(
                    image_path,
                    mask_path,
                    image_root,
                    mask_root,
                    args,
                )
            )

        if index % 10 == 0 or index == len(image_paths):
            problem_count = sum(1 for r in audits if r.issues)
            print(
                f"[检查] {index}/{len(image_paths)}；"
                f"已发现问题样本={problem_count}"
            )

    # 标签存在但没有匹配原图。
    for mask_path in sorted(set(mask_paths) - used_masks):
        record = PairAudit(
            image_path=None,
            mask_path=mask_path,
            image_relative="",
            mask_relative=mask_path.relative_to(mask_root).as_posix(),
        )
        record.add_issue(
            "orphan_mask",
            "标签图未找到对应原图。",
        )
        audits.append(record)

    problematic = [record for record in audits if record.issues]
    good = [record for record in audits if not record.issues]

    # 统一复制所有问题样本。
    for record in problematic:
        copy_pair_to_folder(
            record,
            output_root / "problematic_pairs",
            image_root,
            mask_root,
        )

    # 按问题类型复制。
    if args.copy_by_issue:
        for record in problematic:
            for issue in sorted(record.issues):
                copy_pair_to_folder(
                    record,
                    output_root
                    / "by_issue"
                    / issue_safe_name(issue),
                    image_root,
                    mask_root,
                )

    if args.copy_good:
        for record in good:
            copy_pair_to_folder(
                record,
                output_root / "good_pairs",
                image_root,
                mask_root,
            )

    report_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for record in audits:
        report_rows.append(
            {
                "image": record.image_relative,
                "mask": record.mask_relative,
                "status": "problem" if record.issues else "good",
                "issues": "|".join(sorted(record.issues)),
                "image_width": record.image_width,
                "image_height": record.image_height,
                "mask_width": record.mask_width,
                "mask_height": record.mask_height,
                "image_dtype": record.image_dtype,
                "mask_dtype": record.mask_dtype,
                "mask_shape": record.mask_shape,
                "mask_mode": record.mask_mode,
                "raw_unique_count": record.raw_unique_count,
                "foreground_unique_count": record.foreground_unique_count,
                "min_label_value": record.min_label_value,
                "max_label_value": record.max_label_value,
                "estimated_instance_count": record.estimated_instance_count,
                "small_instance_count": record.small_instance_count,
                "disconnected_id_count": record.disconnected_id_count,
                "non_contiguous_id_count": record.non_contiguous_id_count,
                "largest_instance_area": record.largest_instance_area,
                "median_instance_area": record.median_instance_area,
            }
        )
        for issue, details in record.issue_details:
            detail_rows.append(
                {
                    "image": record.image_relative,
                    "mask": record.mask_relative,
                    "issue": issue,
                    "details": details,
                }
            )

    write_csv(
        output_root / "audit_report.csv",
        report_rows,
        [
            "image",
            "mask",
            "status",
            "issues",
            "image_width",
            "image_height",
            "mask_width",
            "mask_height",
            "image_dtype",
            "mask_dtype",
            "mask_shape",
            "mask_mode",
            "raw_unique_count",
            "foreground_unique_count",
            "min_label_value",
            "max_label_value",
            "estimated_instance_count",
            "small_instance_count",
            "disconnected_id_count",
            "non_contiguous_id_count",
            "largest_instance_area",
            "median_instance_area",
        ],
    )
    write_csv(
        output_root / "issue_details.csv",
        detail_rows,
        ["image", "mask", "issue", "details"],
    )

    with (output_root / "problematic_pairs.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        for record in problematic:
            stream.write(
                f"IMAGE={record.image_relative}\n"
                f"MASK={record.mask_relative}\n"
                f"ISSUES={','.join(sorted(record.issues))}\n\n"
            )

    issue_counts = Counter()
    for record in audits:
        issue_counts.update(record.issues)

    instance_counts = [
        record.estimated_instance_count
        for record in audits
        if record.estimated_instance_count is not None
    ]

    summary = {
        "input": {
            "image_dir": str(image_root),
            "mask_dir": str(mask_root),
            "image_file_count": len(image_paths),
            "mask_file_count": len(mask_paths),
        },
        "results": {
            "audited_record_count": len(audits),
            "good_pair_count": len(good),
            "problem_record_count": len(problematic),
            "issue_counts": dict(issue_counts),
            "estimated_instance_statistics": {
                "minimum": (
                    int(min(instance_counts)) if instance_counts else None
                ),
                "maximum": (
                    int(max(instance_counts)) if instance_counts else None
                ),
                "mean": (
                    float(np.mean(instance_counts))
                    if instance_counts else None
                ),
                "median": (
                    float(np.median(instance_counts))
                    if instance_counts else None
                ),
            },
        },
        "parameters": {
            "background_value": args.background_value,
            "min_instance_area": args.min_instance_area,
            "max_small_instance_ratio": args.max_small_instance_ratio,
            "binary_large_region_area": args.binary_large_region_area,
            "binary_connectivity": args.binary_connectivity,
            "max_disconnected_components": (
                args.max_disconnected_components
            ),
            "expected_min_instances": args.expected_min_instances,
            "expected_max_instances": args.expected_max_instances,
        },
        "outputs": {
            "problematic_pairs": str(
                output_root / "problematic_pairs"
            ),
            "audit_report": str(
                output_root / "audit_report.csv"
            ),
            "issue_details": str(
                output_root / "issue_details.csv"
            ),
        },
    }

    with (output_root / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("[审计完成]")
    print(f"正常配对：{len(good)}")
    print(f"问题记录：{len(problematic)}")
    print(f"问题分类：{dict(issue_counts)}")
    print(
        f"问题样本目录：{output_root / 'problematic_pairs'}"
    )
    print(f"总报告：{output_root / 'audit_report.csv'}")
    print(f"问题详情：{output_root / 'issue_details.csv'}")
    print(f"汇总：{output_root / 'audit_summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
