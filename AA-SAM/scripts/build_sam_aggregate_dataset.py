#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
build_sam3_aggregate_dataset.py

用途
----
将已经按 train / val / test 划分的沥青混合料实例分割数据集，一次性完成：

1. 原图与 uint16 实例标签同步切片；
2. 按完整实例保留比例删除严重截断的颗粒；
3. 删除面积过小的无效实例并将实例 ID 重排为 1...N；
4. 过滤无效切片，并按概率保留少量纯背景负样本；
5. 可同时保留完整原图样本；
6. 将处理后的标签直接转换成 SAM3 训练所需的 COCO-RLE JSON；
7. 最终输出可直接作为 SAM3 训练脚本的 --dataset-root。

输入目录
--------
input_root/
├─ train/
│  ├─ images/
│  └─ masks/
├─ val/
│  ├─ images/
│  └─ masks/
└─ test/
   ├─ images/
   └─ masks/

标签规则
--------
0      = 背景/沥青砂浆
1..N   = 不同集料颗粒实例

最终输出目录
------------
output_root/
├─ train/
│  ├─ images/
│  ├─ masks/
│  └─ _annotations.coco.json
├─ val/
│  ├─ images/
│  ├─ masks/
│  └─ _annotations.coco.json
├─ test/
│  ├─ images/
│  ├─ masks/
│  └─ _annotations.coco.json
├─ annotations/
│  ├─ aggregate_train_sam3.json
│  ├─ aggregate_val_sam3.json
│  ├─ aggregate_test_sam3.json
│  ├─ mud_train_sam3.json          # 兼容旧训练脚本
│  ├─ mud_val_sam3.json
│  └─ mud_test_sam3.json
└─ statistics/
   ├─ sample_manifest.csv
   ├─ image_statistics.csv
   ├─ instance_statistics.csv
   ├─ split_summary.csv
   └─ dataset_summary.json

安装依赖
--------
conda install -c conda-forge numpy opencv pycocotools -y

推荐运行示例
------------
python build_sam3_aggregate_dataset.py ^
  --input-root "./data/example" ^
  --output-root "./data/example" ^
  --prompt "aggregate particle" ^
  --tile-size 896 ^
  --overlap 0.20 ^
  --min-keep-ratio 0.50 ^
  --min-instance-area 20 ^
  --negative-keep-prob 0.10 ^
  --keep-full ^
  --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise SystemExit(
        "缺少 pycocotools。\n"
        "请执行：conda install -c conda-forge pycocotools -y"
    ) from exc


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}
SPLITS_DEFAULT = ("train", "val", "test")


@dataclass(frozen=True)
class SampleRecord:
    split: str
    sample_name: str
    sample_type: str
    source_image: str
    source_mask: str
    output_image: str
    output_mask: str
    x1: int
    y1: int
    x2: int
    y2: int
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    instance_count: int
    foreground_pixels: int
    foreground_ratio: float


@dataclass(frozen=True)
class ImageStatistics:
    split: str
    file_name: str
    image_id: int
    width: int
    height: int
    is_positive: int
    instance_count_original: int
    instance_count_kept: int
    instance_count_skipped: int
    text_input: str
    sample_type: str


@dataclass(frozen=True)
class InstanceStatistics:
    split: str
    file_name: str
    image_id: int
    annotation_id: int
    source_instance_id: int
    area_px: int
    area_normalized: float
    equivalent_diameter_px: float
    bbox_x_px: int
    bbox_y_px: int
    bbox_width_px: int
    bbox_height_px: int
    bbox_x_normalized: float
    bbox_y_normalized: float
    bbox_width_normalized: float
    bbox_height_normalized: float


# -----------------------------------------------------------------------------
# 基础文件与图像函数
# -----------------------------------------------------------------------------

def read_image_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise RuntimeError(f"无法读取图像：{path}")
    return image


def write_image_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if not suffix:
        raise ValueError(f"输出路径缺少扩展名：{path}")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码图像：{path}")
    encoded.tofile(str(path))


def normalize_suffix(suffix: str) -> str:
    suffix = suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


def build_file_map(
    directory: Path,
    extensions: set[str],
    remove_suffix: str = "",
) -> Dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")

    result: Dict[str, Path] = {}
    remove_suffix = normalize_suffix(remove_suffix)

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        relative = path.relative_to(directory)
        relative_without_ext = relative.with_suffix("")
        key_name = relative_without_ext.name
        if remove_suffix and key_name.endswith(remove_suffix):
            key_name = key_name[: -len(remove_suffix)]
        key = str(relative_without_ext.with_name(key_name)).replace("\\", "/")

        if key in result:
            raise RuntimeError(
                f"发现重复文件键“{key}”：\n"
                f"  1) {result[key]}\n"
                f"  2) {path}\n"
                "请修改文件名，确保原图和标签可以唯一配对。"
            )
        result[key] = path

    return result


def find_pairs(
    image_dir: Path,
    mask_dir: Path,
    mask_suffix: str,
) -> List[Tuple[str, Path, Path]]:
    image_map = build_file_map(image_dir, IMAGE_EXTENSIONS)
    mask_map = build_file_map(mask_dir, MASK_EXTENSIONS, remove_suffix=mask_suffix)

    common = sorted(set(image_map) & set(mask_map))
    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))

    if missing_masks:
        print(
            f"[警告] {image_dir} 中有 {len(missing_masks)} 张图像缺少标签，"
            f"示例：{', '.join(missing_masks[:10])}"
        )
    if missing_images:
        print(
            f"[警告] {mask_dir} 中有 {len(missing_images)} 个标签缺少原图，"
            f"示例：{', '.join(missing_images[:10])}"
        )
    if not common:
        raise RuntimeError(
            f"未找到可配对样本：\n图像目录：{image_dir}\n标签目录：{mask_dir}"
        )

    return [(key, image_map[key], mask_map[key]) for key in common]


def validate_mask(mask: np.ndarray, mask_path: Path) -> np.ndarray:
    if mask.ndim == 3:
        if (
            mask.shape[2] >= 3
            and np.array_equal(mask[..., 0], mask[..., 1])
            and np.array_equal(mask[..., 0], mask[..., 2])
        ):
            mask = mask[..., 0]
        else:
            raise ValueError(
                f"标签必须是单通道实例 ID 图：{mask_path}，shape={mask.shape}"
            )

    if mask.ndim != 2:
        raise ValueError(f"标签维度异常：{mask_path}，shape={mask.shape}")

    if np.issubdtype(mask.dtype, np.floating):
        if not np.all(np.isfinite(mask)) or not np.allclose(mask, np.round(mask)):
            raise ValueError(f"标签中存在非整数或非有限像素值：{mask_path}")
        mask = np.round(mask)

    if np.min(mask) < 0:
        raise ValueError(f"标签中存在负数实例 ID：{mask_path}")

    max_id = int(np.max(mask, initial=0))
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"实例 ID 最大值为 {max_id}，超过 uint16 上限 65535：{mask_path}"
        )

    return mask.astype(np.uint16, copy=False)


def relabel_contiguous(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    ids = np.unique(mask)
    ids = ids[ids > 0]

    output = np.zeros(mask.shape, dtype=np.uint16)
    for new_id, old_id in enumerate(ids.tolist(), start=1):
        output[mask == old_id] = new_id
    return output, len(ids)


def compute_instance_areas(mask: np.ndarray) -> Dict[int, int]:
    ids, counts = np.unique(mask, return_counts=True)
    return {
        int(instance_id): int(count)
        for instance_id, count in zip(ids, counts)
        if int(instance_id) != 0
    }


# -----------------------------------------------------------------------------
# 同步切片与标签清理
# -----------------------------------------------------------------------------

def generate_positions(length: int, tile_size: int, overlap: float) -> List[int]:
    if length <= tile_size:
        return [0]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    positions = list(range(0, length - tile_size + 1, stride))
    final_pos = length - tile_size
    if positions[-1] != final_pos:
        positions.append(final_pos)
    return sorted(set(positions))


def pad_to_size(
    image: np.ndarray,
    mask: np.ndarray,
    target_h: int,
    target_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    pad_bottom = max(0, target_h - h)
    pad_right = max(0, target_w - w)

    if pad_bottom == 0 and pad_right == 0:
        return image, mask

    image = cv2.copyMakeBorder(
        image,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    mask = cv2.copyMakeBorder(
        mask,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return image, mask


def process_tile_mask(
    tile_mask: np.ndarray,
    full_instance_areas: Dict[int, int],
    min_keep_ratio: float,
    min_instance_area: int,
) -> Tuple[np.ndarray, int, int, float]:
    cleaned = np.zeros(tile_mask.shape, dtype=np.uint16)
    tile_ids, tile_counts = np.unique(tile_mask, return_counts=True)
    kept_old_ids: List[int] = []

    for instance_id, tile_area in zip(tile_ids.tolist(), tile_counts.tolist()):
        instance_id = int(instance_id)
        tile_area = int(tile_area)
        if instance_id == 0:
            continue

        full_area = full_instance_areas.get(instance_id, 0)
        if full_area <= 0:
            continue

        keep_ratio = tile_area / float(full_area)
        if tile_area >= min_instance_area and keep_ratio >= min_keep_ratio:
            kept_old_ids.append(instance_id)

    for new_id, old_id in enumerate(kept_old_ids, start=1):
        cleaned[tile_mask == old_id] = new_id

    foreground_pixels = int(np.count_nonzero(cleaned))
    foreground_ratio = foreground_pixels / float(cleaned.size)
    return cleaned, len(kept_old_ids), foreground_pixels, foreground_ratio


def remove_small_instances(
    mask: np.ndarray,
    min_area: int,
) -> Tuple[np.ndarray, int, int]:
    """完整图样本使用：删除过小实例并重新连续编号。"""
    cleaned = np.zeros(mask.shape, dtype=np.uint16)
    ids, counts = np.unique(mask, return_counts=True)
    kept_ids: List[int] = []
    skipped = 0

    for instance_id, area in zip(ids.tolist(), counts.tolist()):
        instance_id = int(instance_id)
        area = int(area)
        if instance_id == 0:
            continue
        if area < min_area:
            skipped += 1
            continue
        kept_ids.append(instance_id)

    for new_id, old_id in enumerate(kept_ids, start=1):
        cleaned[mask == old_id] = new_id
    return cleaned, len(kept_ids), skipped


def extreme_pixel_ratio(image: np.ndarray, low: int, high: int) -> float:
    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray_float = gray.astype(np.float32)
        min_v = float(np.min(gray_float))
        max_v = float(np.max(gray_float))
        if max_v > min_v:
            gray = ((gray_float - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros_like(gray_float, dtype=np.uint8)

    extreme = (gray <= low) | (gray >= high)
    return float(np.mean(extreme))


def deterministic_probability(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def sanitize_sample_stem(key: str) -> str:
    """把相对路径转换成不会冲突的扁平样本名。"""
    return key.replace("/", "__").replace("\\", "__")


# -----------------------------------------------------------------------------
# SAM3 / COCO-RLE 标注生成
# -----------------------------------------------------------------------------

def encode_rle(binary_mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(
        np.asfortranarray(binary_mask.astype(np.uint8, copy=False))
    )
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


def mask_bbox_xywh(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(binary_mask)
    if xs.size == 0:
        raise ValueError("空实例掩膜没有边界框")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def make_dataset_header(split: str, prompt: str, category_name: str) -> dict:
    return {
        "info": {
            "description": "Asphalt mixture aggregate particle instance dataset",
            "version": "2.0",
            "split": split,
            "text_prompt": prompt,
            "generator": "build_sam3_aggregate_dataset.py",
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name}],
    }


def add_sample_to_sam3_dataset(
    dataset: dict,
    split: str,
    image_file_name: str,
    mask: np.ndarray,
    prompt: str,
    sample_type: str,
    image_id: int,
    annotation_id_start: int,
    sam3_min_area: int,
) -> Tuple[ImageStatistics, List[InstanceStatistics], int]:
    height, width = mask.shape
    image_record = {
        "id": image_id,
        "file_name": image_file_name,
        "text_input": prompt,
        "width": width,
        "height": height,
        "queried_category": "1",
        "is_instance_exhaustive": 1,
        "is_pixel_exhaustive": 1,
    }
    dataset["images"].append(image_record)

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    kept_count = 0
    skipped_count = 0
    annotation_id = annotation_id_start
    instance_rows: List[InstanceStatistics] = []

    for source_instance_id in map(int, instance_ids):
        binary = mask == source_instance_id
        area_px = int(np.count_nonzero(binary))
        if area_px < sam3_min_area:
            skipped_count += 1
            continue

        x, y, box_w, box_h = mask_bbox_xywh(binary)
        bbox_normalized = [
            x / float(width),
            y / float(height),
            box_w / float(width),
            box_h / float(height),
        ]
        area_normalized = area_px / float(width * height)

        dataset["annotations"].append(
            {
                "area": area_normalized,
                "id": annotation_id,
                "image_id": image_id,
                "source": "napari_manual_then_tiled",
                "bbox": bbox_normalized,
                "segmentation": encode_rle(binary),
                "category_id": 1,
                "iscrowd": 0,
            }
        )

        instance_rows.append(
            InstanceStatistics(
                split=split,
                file_name=image_file_name,
                image_id=image_id,
                annotation_id=annotation_id,
                source_instance_id=source_instance_id,
                area_px=area_px,
                area_normalized=area_normalized,
                equivalent_diameter_px=2.0 * math.sqrt(area_px / math.pi),
                bbox_x_px=x,
                bbox_y_px=y,
                bbox_width_px=box_w,
                bbox_height_px=box_h,
                bbox_x_normalized=bbox_normalized[0],
                bbox_y_normalized=bbox_normalized[1],
                bbox_width_normalized=bbox_normalized[2],
                bbox_height_normalized=bbox_normalized[3],
            )
        )
        annotation_id += 1
        kept_count += 1

    image_stats = ImageStatistics(
        split=split,
        file_name=image_file_name,
        image_id=image_id,
        width=width,
        height=height,
        is_positive=int(kept_count > 0),
        instance_count_original=int(instance_ids.size),
        instance_count_kept=kept_count,
        instance_count_skipped=skipped_count,
        text_input=prompt,
        sample_type=sample_type,
    )
    return image_stats, instance_rows, annotation_id


# -----------------------------------------------------------------------------
# 单个 split 的完整处理
# -----------------------------------------------------------------------------

def save_and_register_sample(
    *,
    split: str,
    sample_name: str,
    sample_type: str,
    source_image_path: Path,
    source_mask_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    output_root: Path,
    image_output_ext: str,
    source_width: int,
    source_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    dataset: dict,
    prompt: str,
    image_id: int,
    annotation_id: int,
    sam3_min_area: int,
) -> Tuple[SampleRecord, ImageStatistics, List[InstanceStatistics], int]:
    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"
    image_path = image_dir / f"{sample_name}{image_output_ext}"
    mask_path = mask_dir / f"{sample_name}.png"

    write_image_unicode(image_path, image)
    write_image_unicode(mask_path, mask.astype(np.uint16, copy=False))

    foreground_pixels = int(np.count_nonzero(mask))
    foreground_ratio = foreground_pixels / float(mask.size)
    instance_count = int(np.unique(mask[mask > 0]).size)

    image_stats, instance_rows, next_annotation_id = add_sample_to_sam3_dataset(
        dataset=dataset,
        split=split,
        image_file_name=image_path.name,
        mask=mask,
        prompt=prompt,
        sample_type=sample_type,
        image_id=image_id,
        annotation_id_start=annotation_id,
        sam3_min_area=sam3_min_area,
    )

    record = SampleRecord(
        split=split,
        sample_name=sample_name,
        sample_type=sample_type,
        source_image=str(source_image_path),
        source_mask=str(source_mask_path),
        output_image=str(image_path),
        output_mask=str(mask_path),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        source_width=source_width,
        source_height=source_height,
        output_width=int(mask.shape[1]),
        output_height=int(mask.shape[0]),
        instance_count=instance_count,
        foreground_pixels=foreground_pixels,
        foreground_ratio=foreground_ratio,
    )
    return record, image_stats, instance_rows, next_annotation_id


def process_split(
    *,
    split: str,
    input_root: Path,
    output_root: Path,
    image_subdir: str,
    mask_subdir: str,
    mask_suffix: str,
    prompt: str,
    category_name: str,
    tile_size: int,
    overlap: float,
    min_keep_ratio: float,
    min_instance_area: int,
    sam3_min_area: int,
    min_instances_per_tile: int,
    min_foreground_ratio: float,
    negative_keep_prob: float,
    keep_full: bool,
    image_output_ext: str,
    max_extreme_ratio: float,
    black_threshold: int,
    white_threshold: int,
    seed: int,
) -> Tuple[
    dict,
    List[SampleRecord],
    List[ImageStatistics],
    List[InstanceStatistics],
    Dict[str, int],
]:
    input_image_dir = input_root / split / image_subdir
    input_mask_dir = input_root / split / mask_subdir
    output_image_dir = output_root / split / "images"
    output_mask_dir = output_root / split / "masks"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(input_image_dir, input_mask_dir, mask_suffix)
    dataset = make_dataset_header(split, prompt, category_name)
    sample_records: List[SampleRecord] = []
    image_rows: List[ImageStatistics] = []
    instance_rows: List[InstanceStatistics] = []

    stats = {
        "source_images": 0,
        "full_samples": 0,
        "tile_candidates": 0,
        "tile_saved": 0,
        "positive_tiles": 0,
        "negative_tiles": 0,
        "filtered_empty": 0,
        "filtered_extreme": 0,
        "sam3_images": 0,
        "sam3_positive_images": 0,
        "sam3_negative_images": 0,
        "sam3_annotations": 0,
    }

    next_image_id = 1
    next_annotation_id = 1

    for index, (key, image_path, mask_path) in enumerate(pairs, start=1):
        print(f"[{split}] {index}/{len(pairs)}：{key}")
        image = read_image_unicode(image_path, cv2.IMREAD_UNCHANGED)
        mask = validate_mask(
            read_image_unicode(mask_path, cv2.IMREAD_UNCHANGED), mask_path
        )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"原图与标签尺寸不一致：\n"
                f"图像：{image_path}，shape={image.shape[:2]}\n"
                f"标签：{mask_path}，shape={mask.shape[:2]}"
            )

        stats["source_images"] += 1
        source_h, source_w = mask.shape
        flat_stem = sanitize_sample_stem(key)

        # 先把完整标签改成连续编号，保证后续面积统计稳定。
        full_mask, _ = relabel_contiguous(mask)
        full_instance_areas = compute_instance_areas(full_mask)

        if keep_full:
            cleaned_full, _, _ = remove_small_instances(
                full_mask, min_area=min_instance_area
            )
            full_name = f"{flat_stem}__full"
            record, image_stat, rows, next_annotation_id = save_and_register_sample(
                split=split,
                sample_name=full_name,
                sample_type="full",
                source_image_path=image_path,
                source_mask_path=mask_path,
                image=image,
                mask=cleaned_full,
                output_root=output_root,
                image_output_ext=image_output_ext,
                source_width=source_w,
                source_height=source_h,
                x1=0,
                y1=0,
                x2=source_w,
                y2=source_h,
                dataset=dataset,
                prompt=prompt,
                image_id=next_image_id,
                annotation_id=next_annotation_id,
                sam3_min_area=sam3_min_area,
            )
            sample_records.append(record)
            image_rows.append(image_stat)
            instance_rows.extend(rows)
            next_image_id += 1
            stats["full_samples"] += 1

        padded_image, padded_mask = pad_to_size(
            image, full_mask, tile_size, tile_size
        )
        padded_h, padded_w = padded_mask.shape
        x_positions = generate_positions(padded_w, tile_size, overlap)
        y_positions = generate_positions(padded_h, tile_size, overlap)

        for y1 in y_positions:
            for x1 in x_positions:
                stats["tile_candidates"] += 1
                x2 = x1 + tile_size
                y2 = y1 + tile_size

                image_tile = padded_image[y1:y2, x1:x2].copy()
                raw_mask_tile = padded_mask[y1:y2, x1:x2].copy()

                cleaned_mask, instance_count, _, foreground_ratio = process_tile_mask(
                    raw_mask_tile,
                    full_instance_areas,
                    min_keep_ratio=min_keep_ratio,
                    min_instance_area=min_instance_area,
                )

                if max_extreme_ratio < 1.0:
                    ratio = extreme_pixel_ratio(
                        image_tile,
                        low=black_threshold,
                        high=white_threshold,
                    )
                    if ratio > max_extreme_ratio:
                        stats["filtered_extreme"] += 1
                        continue

                positive = (
                    instance_count >= min_instances_per_tile
                    and foreground_ratio >= min_foreground_ratio
                )

                tile_key = f"{split}:{key}:{x1}:{y1}"
                if not positive:
                    keep_negative = (
                        negative_keep_prob > 0.0
                        and deterministic_probability(tile_key, seed)
                        < negative_keep_prob
                    )
                    if not keep_negative:
                        stats["filtered_empty"] += 1
                        continue
                    cleaned_mask.fill(0)
                    stats["negative_tiles"] += 1
                    sample_type = "tile_negative"
                else:
                    stats["positive_tiles"] += 1
                    sample_type = "tile_positive"

                tile_name = f"{flat_stem}__tile_y{y1:05d}_x{x1:05d}"
                record, image_stat, rows, next_annotation_id = save_and_register_sample(
                    split=split,
                    sample_name=tile_name,
                    sample_type=sample_type,
                    source_image_path=image_path,
                    source_mask_path=mask_path,
                    image=image_tile,
                    mask=cleaned_mask,
                    output_root=output_root,
                    image_output_ext=image_output_ext,
                    source_width=source_w,
                    source_height=source_h,
                    x1=x1,
                    y1=y1,
                    x2=min(x2, source_w),
                    y2=min(y2, source_h),
                    dataset=dataset,
                    prompt=prompt,
                    image_id=next_image_id,
                    annotation_id=next_annotation_id,
                    sam3_min_area=sam3_min_area,
                )
                sample_records.append(record)
                image_rows.append(image_stat)
                instance_rows.extend(rows)
                next_image_id += 1
                stats["tile_saved"] += 1

    stats["sam3_images"] = len(dataset["images"])
    stats["sam3_annotations"] = len(dataset["annotations"])
    stats["sam3_positive_images"] = sum(row.is_positive for row in image_rows)
    stats["sam3_negative_images"] = len(image_rows) - stats["sam3_positive_images"]

    return dataset, sample_records, image_rows, instance_rows, stats


# -----------------------------------------------------------------------------
# 输出与检查
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    converted = [asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row) for row in rows]
    fieldnames = list(converted[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(converted)


def write_dict_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def validate_generated_split(output_root: Path, split: str, dataset: dict) -> None:
    image_dir = output_root / split / "images"
    missing = []
    image_ids = set()
    annotation_ids = set()

    for image_record in dataset["images"]:
        image_id = image_record.get("id")
        if image_id in image_ids:
            raise ValueError(f"{split} JSON 中存在重复 image id：{image_id}")
        image_ids.add(image_id)

        image_path = image_dir / str(image_record.get("file_name", ""))
        if not image_path.is_file():
            missing.append(str(image_path))

    for annotation in dataset["annotations"]:
        annotation_id = annotation.get("id")
        if annotation_id in annotation_ids:
            raise ValueError(
                f"{split} JSON 中存在重复 annotation id：{annotation_id}"
            )
        annotation_ids.add(annotation_id)
        if annotation.get("image_id") not in image_ids:
            raise ValueError(
                f"{split} annotation {annotation_id} 引用了不存在的 image_id"
            )
        segmentation = annotation.get("segmentation", {})
        if not segmentation.get("counts") or not segmentation.get("size"):
            raise ValueError(
                f"{split} annotation {annotation_id} 缺少有效 COCO RLE"
            )

    if missing:
        raise FileNotFoundError(
            f"{split} 标注中有 {len(missing)} 个图像文件找不到，示例：{missing[:5]}"
        )


def save_split_annotations(
    output_root: Path,
    split: str,
    dataset: dict,
    write_legacy_mud_alias: bool,
) -> None:
    # 主要训练格式：dataset_root/train/_annotations.coco.json
    write_json(output_root / split / "_annotations.coco.json", dataset)

    # 便于显式指定标注文件，且语义与任务一致。
    write_json(
        output_root / "annotations" / f"aggregate_{split}_sam3.json",
        dataset,
    )

    # 兼容此前可能硬编码 mud_train_sam3.json 的训练代码。
    if write_legacy_mud_alias:
        write_json(
            output_root / "annotations" / f"mud_{split}_sam3.json",
            dataset,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="同步切片实例标签，并直接生成可训练的 SAM3 集料数据集。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLITS_DEFAULT),
        help="需要处理的数据集划分。",
    )
    parser.add_argument("--image-subdir", default="images")
    parser.add_argument("--mask-subdir", default="masks")
    parser.add_argument(
        "--mask-suffix",
        default="",
        help='例如原图 001.jpg、标签 001_mask.png，则填写 "_mask"。',
    )

    parser.add_argument(
        "--prompt",
        default="aggregate particle",
        help="SAM3 文本提示。",
    )
    parser.add_argument(
        "--category-name",
        default="aggregate_particle",
        help="COCO categories 中的类别名称。",
    )

    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--min-keep-ratio", type=float, default=0.50)
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=20,
        help="同步切片/完整图清理时保留实例的最小像素面积。",
    )
    parser.add_argument(
        "--sam3-min-area",
        type=int,
        default=1,
        help="写入 SAM3 JSON 时实例的最小像素面积。通常不应大于 min-instance-area。",
    )
    parser.add_argument("--min-instances-per-tile", type=int, default=1)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.0001)
    parser.add_argument("--negative-keep-prob", type=float, default=0.10)

    parser.add_argument(
        "--keep-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在切片样本之外同时保留完整图像样本。",
    )
    parser.add_argument("--image-output-ext", default=".png")
    parser.add_argument("--max-extreme-ratio", type=float, default=0.98)
    parser.add_argument("--black-threshold", type=int, default=5)
    parser.add_argument("--white-threshold", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-legacy-mud-alias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="额外写出 mud_train_sam3.json 等旧名称，兼容原泥块训练代码。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0:
        raise ValueError("--tile-size 必须大于 0。")
    if not (0.0 <= args.overlap < 0.9):
        raise ValueError("--overlap 必须位于 [0, 0.9) 内。")
    if not (0.0 <= args.min_keep_ratio <= 1.0):
        raise ValueError("--min-keep-ratio 必须位于 [0, 1] 内。")
    if args.min_instance_area < 1:
        raise ValueError("--min-instance-area 必须至少为 1。")
    if args.sam3_min_area < 1:
        raise ValueError("--sam3-min-area 必须至少为 1。")
    if args.sam3_min_area > args.min_instance_area:
        print(
            "[警告] --sam3-min-area 大于 --min-instance-area；"
            "部分已保存到标签中的实例不会写入 JSON。"
        )
    if args.min_instances_per_tile < 1:
        raise ValueError("--min-instances-per-tile 必须至少为 1。")
    if not (0.0 <= args.min_foreground_ratio <= 1.0):
        raise ValueError("--min-foreground-ratio 必须位于 [0, 1] 内。")
    if not (0.0 <= args.negative_keep_prob <= 1.0):
        raise ValueError("--negative-keep-prob 必须位于 [0, 1] 内。")
    if not (0.0 < args.max_extreme_ratio <= 1.0):
        raise ValueError("--max-extreme-ratio 必须位于 (0, 1] 内。")
    if not (0 <= args.black_threshold < args.white_threshold <= 255):
        raise ValueError(
            "必须满足 0 <= black-threshold < white-threshold <= 255。"
        )
    if not args.prompt.strip():
        raise ValueError("--prompt 不能为空。")
    if not args.category_name.strip():
        raise ValueError("--category-name 不能为空。")

    ext = args.image_output_ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"--image-output-ext 不受支持：{ext}；可选：{sorted(IMAGE_EXTENSIONS)}"
        )
    args.image_output_ext = ext


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if overwrite:
            print(f"[提示] 删除已有输出目录：{output_root}")
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise FileExistsError(
                f"输出目录已存在且非空：{output_root}\n"
                "请更换输出目录，或添加 --overwrite。"
            )
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"输入数据集根目录不存在：{input_root}")
    if input_root == output_root:
        raise ValueError("输入目录和输出目录不能相同。")

    prepare_output(output_root, args.overwrite)

    all_sample_records: List[SampleRecord] = []
    all_image_rows: List[ImageStatistics] = []
    all_instance_rows: List[InstanceStatistics] = []
    split_summary_rows: List[dict] = []
    dataset_summary = {
        "task": "asphalt_mixture_aggregate_instance_segmentation",
        "prompt": args.prompt.strip(),
        "category_name": args.category_name.strip(),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "min_keep_ratio": args.min_keep_ratio,
        "min_instance_area": args.min_instance_area,
        "sam3_min_area": args.sam3_min_area,
        "negative_keep_prob": args.negative_keep_prob,
        "keep_full": args.keep_full,
        "splits": {},
    }

    for split in args.splits:
        print("\n" + "=" * 76)
        print(f"开始处理并转换：{split}")
        print("=" * 76)

        dataset, sample_records, image_rows, instance_rows, stats = process_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            image_subdir=args.image_subdir,
            mask_subdir=args.mask_subdir,
            mask_suffix=args.mask_suffix,
            prompt=args.prompt.strip(),
            category_name=args.category_name.strip(),
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_keep_ratio=args.min_keep_ratio,
            min_instance_area=args.min_instance_area,
            sam3_min_area=args.sam3_min_area,
            min_instances_per_tile=args.min_instances_per_tile,
            min_foreground_ratio=args.min_foreground_ratio,
            negative_keep_prob=args.negative_keep_prob,
            keep_full=args.keep_full,
            image_output_ext=args.image_output_ext,
            max_extreme_ratio=args.max_extreme_ratio,
            black_threshold=args.black_threshold,
            white_threshold=args.white_threshold,
            seed=args.seed,
        )

        validate_generated_split(output_root, split, dataset)
        save_split_annotations(
            output_root=output_root,
            split=split,
            dataset=dataset,
            write_legacy_mud_alias=args.write_legacy_mud_alias,
        )

        all_sample_records.extend(sample_records)
        all_image_rows.extend(image_rows)
        all_instance_rows.extend(instance_rows)
        split_summary_rows.append({"split": split, **stats})
        dataset_summary["splits"][split] = stats

        print(f"[{split}] 原始图像：{stats['source_images']}")
        print(f"[{split}] 完整图样本：{stats['full_samples']}")
        print(f"[{split}] 候选切片：{stats['tile_candidates']}")
        print(f"[{split}] 保存切片：{stats['tile_saved']}")
        print(f"[{split}] SAM3 图像记录：{stats['sam3_images']}")
        print(f"[{split}] SAM3 实例标注：{stats['sam3_annotations']}")
        print(f"[{split}] 标注文件：{output_root / split / '_annotations.coco.json'}")

    statistics_dir = output_root / "statistics"
    write_csv(statistics_dir / "sample_manifest.csv", all_sample_records)
    write_csv(statistics_dir / "image_statistics.csv", all_image_rows)
    write_csv(statistics_dir / "instance_statistics.csv", all_instance_rows)
    write_dict_csv(statistics_dir / "split_summary.csv", split_summary_rows)

    dataset_summary["total_sam3_images"] = sum(
        row["sam3_images"] for row in split_summary_rows
    )
    dataset_summary["total_sam3_annotations"] = sum(
        row["sam3_annotations"] for row in split_summary_rows
    )
    write_json(statistics_dir / "dataset_summary.json", dataset_summary)

    print("\n" + "=" * 76)
    print("全部处理完成：输出目录已经可以直接用于 SAM3 训练")
    print(f"--dataset-root \"{output_root}\"")
    print("主要标注文件：")
    for split in args.splits:
        print(f"  {output_root / split / '_annotations.coco.json'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(
            f"\n[运行失败] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
