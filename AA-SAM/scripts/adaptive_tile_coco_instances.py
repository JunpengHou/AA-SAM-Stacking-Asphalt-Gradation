#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将现有 COCO 实例分割数据集自适应切分为局部图块，使每个图块中的
有效实例数不超过指定上限（默认 180）。

设计目标
--------
1. 保留原有 train / val / test 划分，避免数据泄漏；
2. 图像与实例掩膜同步切块；
3. 优先使用 896×896 图块；
4. 某图块实例数超过上限时，自动继续切为更小图块；
5. 边界实例按可见面积比例筛选；
6. 检查每个原始实例是否至少进入一个图块；
7. 输出新的 COCO RLE 标注、映射表、覆盖率和审计结果；
8. 不修改原始数据集。

依赖
----
pip install opencv-python numpy pycocotools

Windows/Conda 推荐：
conda install -c conda-forge opencv numpy
conda install -c conda-forge pycocotools
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise SystemExit(
        "缺少 pycocotools。请先执行：\n"
        "conda install -c conda-forge pycocotools"
    ) from exc


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    w: int
    h: int
    level: int
    reason: str = "adaptive"


@dataclass
class LocalInstance:
    ann_id: int
    category_id: int
    iscrowd: int
    x0: int
    y0: int
    mask: np.ndarray
    area: int
    source_bbox: list[float]

    @property
    def x1(self) -> int:
        return self.x0 + int(self.mask.shape[1])

    @property
    def y1(self) -> int:
        return self.y0 + int(self.mask.shape[0])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自适应切分 COCO 实例分割数据集，使每个图块实例数不超过上限。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "原始数据集根目录。默认结构："
            "train/images + train/_annotations.coco.json，val/test 同理。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="新切块数据集输出目录。不得与原始数据集相同。",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="需要处理的数据集划分，逗号分隔。默认：train,val,test",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="每个 split 下的图像目录名。默认：images",
    )
    parser.add_argument(
        "--annotation-name",
        type=str,
        default="_annotations.coco.json",
        help="每个 split 下的 COCO 标注文件名。",
    )
    parser.add_argument(
        "--tile-sizes",
        type=str,
        default="896,672,512,384",
        help=(
            "按从大到小排列的自适应图块尺寸。"
            "默认：896,672,512,384"
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="相邻图块重叠率。默认：0.20",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=180,
        help="单个图块允许的最大有效实例数。默认：180",
    )
    parser.add_argument(
        "--visible-ratio",
        type=float,
        default=0.50,
        help=(
            "边界实例裁剪后面积 / 原实例面积的最低保留比例。"
            "默认：0.50"
        ),
    )
    parser.add_argument(
        "--rescue-visible-ratio",
        type=float,
        default=0.25,
        help=(
            "覆盖补救图块允许使用的最低可见比例。"
            "仅在某原始实例没有进入任何常规图块时使用。默认：0.25"
        ),
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=16,
        help="裁剪后实例掩膜最小像素面积。默认：16",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="保留不含有效实例的空图块。默认不保留。",
    )
    parser.add_argument(
        "--include-crowd",
        action="store_true",
        help="保留 iscrowd=1 的标注。默认跳过 crowd 标注。",
    )
    parser.add_argument(
        "--image-format",
        choices=("same", "jpg", "png"),
        default="same",
        help="输出图像格式。默认 same，沿用原图格式。",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="输出 JPG 的质量。默认：95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出目录已存在，则先删除后重建。",
    )
    parser.add_argument(
        "--allow-missing-instances",
        action="store_true",
        help=(
            "即使仍有原始实例无法被任何图块覆盖，也继续输出。"
            "默认遇到缺失实例时停止并报错。"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="每处理多少张原图输出一次进度。默认：10",
    )
    return parser.parse_args()


def parse_tile_sizes(text: str) -> list[int]:
    values = sorted(
        {int(item.strip()) for item in text.split(",") if item.strip()},
        reverse=True,
    )
    if not values:
        raise ValueError("--tile-sizes 不能为空。")
    if any(value <= 0 for value in values):
        raise ValueError("--tile-sizes 中的尺寸必须大于0。")
    return values


def validate_args(args: argparse.Namespace, tile_sizes: list[int]) -> None:
    if not (0.0 <= args.overlap < 0.90):
        raise ValueError("--overlap 必须位于 [0, 0.90) 范围内。")
    if not (0.0 < args.visible_ratio <= 1.0):
        raise ValueError("--visible-ratio 必须位于 (0, 1] 范围内。")
    if not (0.0 < args.rescue_visible_ratio <= args.visible_ratio):
        raise ValueError(
            "--rescue-visible-ratio 必须大于0且不超过 --visible-ratio。"
        )
    if args.max_instances <= 0:
        raise ValueError("--max-instances 必须大于0。")
    if args.min_mask_area <= 0:
        raise ValueError("--min-mask-area 必须大于0。")
    if args.jpg_quality < 1 or args.jpg_quality > 100:
        raise ValueError("--jpg-quality 必须位于1～100。")
    if len(tile_sizes) != len(set(tile_sizes)):
        raise ValueError("--tile-sizes 中不能包含重复尺寸。")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def resolve_image_path(images_dir: Path, file_name: str) -> Path:
    candidate = images_dir / file_name
    if candidate.is_file():
        return candidate

    candidate = images_dir / Path(file_name).name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"未找到图像：COCO file_name={file_name!r}；图像目录={images_dir}"
    )


def polygon_to_local_mask(
    segmentation: list[Any],
    image_w: int,
    image_h: int,
    bbox: list[float],
) -> tuple[int, int, np.ndarray]:
    x, y, w, h = [float(value) for value in bbox]
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(image_w, int(math.ceil(x + w)))
    y1 = min(image_h, int(math.ceil(y + h)))

    if x1 <= x0 or y1 <= y0:
        return x0, y0, np.zeros((0, 0), dtype=np.uint8)

    local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    valid_polygons: list[np.ndarray] = []

    for polygon in segmentation:
        array = np.asarray(polygon, dtype=np.float32)
        if array.size < 6 or array.size % 2 != 0:
            continue
        points = array.reshape(-1, 2)
        points[:, 0] -= x0
        points[:, 1] -= y0
        points = np.round(points).astype(np.int32)
        valid_polygons.append(points)

    if valid_polygons:
        cv2.fillPoly(local, valid_polygons, color=1)

    return x0, y0, local


def rle_to_local_mask(
    segmentation: dict[str, Any],
    image_w: int,
    image_h: int,
    bbox: list[float],
) -> tuple[int, int, np.ndarray]:
    rle = segmentation
    if isinstance(rle.get("counts"), list):
        rle = mask_utils.frPyObjects(rle, image_h, image_w)

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2).astype(np.uint8)
    else:
        decoded = (decoded > 0).astype(np.uint8)

    x, y, w, h = [float(value) for value in bbox]
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(image_w, int(math.ceil(x + w)))
    y1 = min(image_h, int(math.ceil(y + h)))

    if x1 <= x0 or y1 <= y0:
        return x0, y0, np.zeros((0, 0), dtype=np.uint8)

    return x0, y0, decoded[y0:y1, x0:x1].copy()


def tighten_local_mask(
    x0: int,
    y0: int,
    mask: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    if mask.size == 0 or not np.any(mask):
        return x0, y0, np.zeros((0, 0), dtype=np.uint8)

    ys, xs = np.nonzero(mask)
    lx0 = int(xs.min())
    ly0 = int(ys.min())
    lx1 = int(xs.max()) + 1
    ly1 = int(ys.max()) + 1
    return (
        x0 + lx0,
        y0 + ly0,
        mask[ly0:ly1, lx0:lx1].astype(np.uint8, copy=True),
    )


def build_local_instance(
    ann: dict[str, Any],
    image_w: int,
    image_h: int,
) -> LocalInstance | None:
    segmentation = ann.get("segmentation")
    bbox = ann.get("bbox")

    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if segmentation is None:
        return None

    if isinstance(segmentation, list):
        x0, y0, local = polygon_to_local_mask(
            segmentation,
            image_w,
            image_h,
            bbox,
        )
    elif isinstance(segmentation, dict):
        x0, y0, local = rle_to_local_mask(
            segmentation,
            image_w,
            image_h,
            bbox,
        )
    else:
        return None

    x0, y0, local = tighten_local_mask(x0, y0, local)
    area = int(local.sum())

    if area <= 0:
        return None

    return LocalInstance(
        ann_id=int(ann["id"]),
        category_id=int(ann["category_id"]),
        iscrowd=int(ann.get("iscrowd", 0)),
        x0=x0,
        y0=y0,
        mask=local,
        area=area,
        source_bbox=[float(value) for value in bbox],
    )


def axis_starts(start: int, length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [start]

    step = max(1, int(round(tile * (1.0 - overlap))))
    last = start + length - tile
    values = list(range(start, last + 1, step))
    if not values or values[-1] != last:
        values.append(last)
    return sorted(set(values))


def make_windows_in_region(
    region: Window,
    tile_size: int,
    overlap: float,
    level: int,
    reason: str,
) -> list[Window]:
    tile_w = min(tile_size, region.w)
    tile_h = min(tile_size, region.h)

    xs = axis_starts(region.x, region.w, tile_w, overlap)
    ys = axis_starts(region.y, region.h, tile_h, overlap)

    return [
        Window(x=x, y=y, w=tile_w, h=tile_h, level=level, reason=reason)
        for y in ys
        for x in xs
    ]


def intersection_slice(
    instance: LocalInstance,
    window: Window,
) -> tuple[slice, slice, slice, slice] | None:
    ix0 = max(instance.x0, window.x)
    iy0 = max(instance.y0, window.y)
    ix1 = min(instance.x1, window.x + window.w)
    iy1 = min(instance.y1, window.y + window.h)

    if ix1 <= ix0 or iy1 <= iy0:
        return None

    src_x0 = ix0 - instance.x0
    src_y0 = iy0 - instance.y0
    src_x1 = ix1 - instance.x0
    src_y1 = iy1 - instance.y0

    dst_x0 = ix0 - window.x
    dst_y0 = iy0 - window.y
    dst_x1 = ix1 - window.x
    dst_y1 = iy1 - window.y

    return (
        slice(src_y0, src_y1),
        slice(src_x0, src_x1),
        slice(dst_y0, dst_y1),
        slice(dst_x0, dst_x1),
    )


def clipped_area(instance: LocalInstance, window: Window) -> int:
    slices = intersection_slice(instance, window)
    if slices is None:
        return 0
    src_ys, src_xs, _, _ = slices
    return int(instance.mask[src_ys, src_xs].sum())


def included_instances(
    instances: list[LocalInstance],
    window: Window,
    visible_ratio: float,
    min_mask_area: int,
) -> list[tuple[LocalInstance, int, float]]:
    included: list[tuple[LocalInstance, int, float]] = []

    for instance in instances:
        # 先做包围盒相交判断，避免无意义的掩膜裁剪。
        if (
            instance.x1 <= window.x
            or instance.y1 <= window.y
            or instance.x0 >= window.x + window.w
            or instance.y0 >= window.y + window.h
        ):
            continue

        area = clipped_area(instance, window)
        if area < min_mask_area:
            continue

        ratio = area / max(instance.area, 1)
        if ratio + 1.0e-12 < visible_ratio:
            continue

        included.append((instance, area, ratio))

    return included


def deduplicate_windows(windows: Iterable[Window]) -> list[Window]:
    best: dict[tuple[int, int, int, int], Window] = {}
    for window in windows:
        key = (window.x, window.y, window.w, window.h)
        previous = best.get(key)
        if previous is None or window.level > previous.level:
            best[key] = window
    return sorted(
        best.values(),
        key=lambda item: (item.y, item.x, -item.w, -item.h),
    )


def refine_window(
    window: Window,
    level: int,
    tile_sizes: list[int],
    overlap: float,
    instances: list[LocalInstance],
    visible_ratio: float,
    min_mask_area: int,
    max_instances: int,
    final_windows: list[Window],
    overloads: list[dict[str, Any]],
) -> None:
    included = included_instances(
        instances,
        window,
        visible_ratio,
        min_mask_area,
    )

    if len(included) <= max_instances:
        final_windows.append(window)
        return

    if level + 1 >= len(tile_sizes):
        overloads.append(
            {
                "x": window.x,
                "y": window.y,
                "w": window.w,
                "h": window.h,
                "instance_count": len(included),
                "message": (
                    "最小图块仍超过实例上限。请在 --tile-sizes 中增加更小尺寸。"
                ),
            }
        )
        return

    child_size = tile_sizes[level + 1]
    child_windows = make_windows_in_region(
        window,
        tile_size=child_size,
        overlap=overlap,
        level=level + 1,
        reason=f"refined_from_{window.w}x{window.h}",
    )

    # 极端情况下防止尺寸没有真正减小而无限递归。
    if (
        len(child_windows) == 1
        and child_windows[0].x == window.x
        and child_windows[0].y == window.y
        and child_windows[0].w == window.w
        and child_windows[0].h == window.h
    ):
        overloads.append(
            {
                "x": window.x,
                "y": window.y,
                "w": window.w,
                "h": window.h,
                "instance_count": len(included),
                "message": "子图块尺寸未减小，无法继续自适应切分。",
            }
        )
        return

    for child in child_windows:
        refine_window(
            child,
            level + 1,
            tile_sizes,
            overlap,
            instances,
            visible_ratio,
            min_mask_area,
            max_instances,
            final_windows,
            overloads,
        )


def centered_window(
    center_x: float,
    center_y: float,
    tile_size: int,
    image_w: int,
    image_h: int,
    level: int,
) -> Window:
    w = min(tile_size, image_w)
    h = min(tile_size, image_h)

    x = int(round(center_x - w / 2.0))
    y = int(round(center_y - h / 2.0))
    x = min(max(0, x), max(0, image_w - w))
    y = min(max(0, y), max(0, image_h - h))

    return Window(
        x=x,
        y=y,
        w=w,
        h=h,
        level=level,
        reason="coverage_rescue",
    )


def add_rescue_windows(
    windows: list[Window],
    instances: list[LocalInstance],
    tile_sizes: list[int],
    image_w: int,
    image_h: int,
    visible_ratio: float,
    rescue_visible_ratio: float,
    min_mask_area: int,
    max_instances: int,
) -> tuple[list[Window], list[int]]:
    windows = deduplicate_windows(windows)

    covered_ids: set[int] = set()
    for window in windows:
        for instance, _, _ in included_instances(
            instances,
            window,
            visible_ratio,
            min_mask_area,
        ):
            covered_ids.add(instance.ann_id)

    missing = [
        instance
        for instance in instances
        if instance.ann_id not in covered_ids
    ]

    for target in missing:
        candidate: Window | None = None

        # 从大到小尝试。大图块有利于保留完整颗粒；
        # 若实例过多，再逐步减小图块。
        for level, tile_size in enumerate(tile_sizes):
            window = centered_window(
                *target.center,
                tile_size=tile_size,
                image_w=image_w,
                image_h=image_h,
                level=level,
            )
            included = included_instances(
                instances,
                window,
                rescue_visible_ratio,
                min_mask_area,
            )
            included_ids = {item[0].ann_id for item in included}

            if (
                target.ann_id in included_ids
                and len(included) <= max_instances
            ):
                candidate = window
                break

        if candidate is not None:
            windows.append(candidate)
            windows = deduplicate_windows(windows)

    covered_after: set[int] = set()
    for window in windows:
        # 常规图块使用常规阈值；补救图块允许较低阈值。
        ratio = (
            rescue_visible_ratio
            if window.reason == "coverage_rescue"
            else visible_ratio
        )
        for instance, _, _ in included_instances(
            instances,
            window,
            ratio,
            min_mask_area,
        ):
            covered_after.add(instance.ann_id)

    missing_ids = [
        instance.ann_id
        for instance in instances
        if instance.ann_id not in covered_after
    ]
    return windows, missing_ids


def output_extension(source_path: Path, image_format: str) -> str:
    if image_format == "jpg":
        return ".jpg"
    if image_format == "png":
        return ".png"

    suffix = source_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return suffix
    return ".png"


def write_image(
    path: Path,
    image: np.ndarray,
    jpg_quality: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    success = cv2.imwrite(str(path), image, params)
    if not success:
        raise IOError(f"图像写入失败：{path}")


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(value) for value in rle["size"]],
        "counts": counts,
    }


def crop_instance_mask(
    instance: LocalInstance,
    window: Window,
) -> np.ndarray:
    tile_mask = np.zeros((window.h, window.w), dtype=np.uint8)
    slices = intersection_slice(instance, window)
    if slices is None:
        return tile_mask

    src_ys, src_xs, dst_ys, dst_xs = slices
    tile_mask[dst_ys, dst_xs] = instance.mask[src_ys, src_xs]
    return tile_mask


def safe_stem(file_name: str) -> str:
    stem = Path(file_name).stem
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in stem
    )
    return cleaned or "image"


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


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


def process_split(
    split: str,
    input_split_dir: Path,
    output_split_dir: Path,
    images_subdir: str,
    annotation_name: str,
    tile_sizes: list[int],
    overlap: float,
    max_instances: int,
    visible_ratio: float,
    rescue_visible_ratio: float,
    min_mask_area: int,
    keep_empty: bool,
    include_crowd: bool,
    image_format: str,
    jpg_quality: int,
    allow_missing_instances: bool,
    progress_every: int,
) -> dict[str, Any]:
    annotation_path = input_split_dir / annotation_name
    images_dir = input_split_dir / images_subdir

    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if not images_dir.is_dir():
        raise NotADirectoryError(images_dir)

    coco = load_json(annotation_path)
    source_images = coco.get("images", [])
    source_annotations = coco.get("annotations", [])

    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in source_annotations:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    output_images_dir = output_split_dir / images_subdir
    output_images_dir.mkdir(parents=True, exist_ok=True)

    new_images: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    next_image_id = 1
    next_annotation_id = 1
    tile_instance_counts: list[int] = []
    tile_size_counter: Counter[str] = Counter()
    source_valid_instance_total = 0
    source_covered_instance_total = 0
    duplicate_training_instances = 0
    rescue_tile_count = 0

    for source_index, image_info in enumerate(source_images, start=1):
        source_image_id = int(image_info["id"])
        source_file_name = str(image_info["file_name"])
        source_path = resolve_image_path(images_dir, source_file_name)

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            issue_rows.append(
                {
                    "source_image_id": source_image_id,
                    "source_file_name": source_file_name,
                    "source_annotation_id": "",
                    "issue": "unreadable_image",
                    "details": str(source_path),
                }
            )
            continue

        image_h, image_w = image.shape[:2]
        json_w = int(image_info.get("width", image_w))
        json_h = int(image_info.get("height", image_h))
        if json_w != image_w or json_h != image_h:
            issue_rows.append(
                {
                    "source_image_id": source_image_id,
                    "source_file_name": source_file_name,
                    "source_annotation_id": "",
                    "issue": "image_size_mismatch",
                    "details": (
                        f"json={json_w}x{json_h}; actual={image_w}x{image_h}"
                    ),
                }
            )

        instances: list[LocalInstance] = []
        skipped_ann_ids: list[int] = []

        for ann in anns_by_image.get(source_image_id, []):
            ann_id = int(ann["id"])
            if int(ann.get("iscrowd", 0)) == 1 and not include_crowd:
                skipped_ann_ids.append(ann_id)
                issue_rows.append(
                    {
                        "source_image_id": source_image_id,
                        "source_file_name": source_file_name,
                        "source_annotation_id": ann_id,
                        "issue": "crowd_annotation_skipped",
                        "details": "使用 --include-crowd 可保留。",
                    }
                )
                continue

            try:
                instance = build_local_instance(ann, image_w, image_h)
            except Exception as exc:
                instance = None
                issue_rows.append(
                    {
                        "source_image_id": source_image_id,
                        "source_file_name": source_file_name,
                        "source_annotation_id": ann_id,
                        "issue": "mask_decode_error",
                        "details": repr(exc),
                    }
                )

            if instance is None:
                skipped_ann_ids.append(ann_id)
                issue_rows.append(
                    {
                        "source_image_id": source_image_id,
                        "source_file_name": source_file_name,
                        "source_annotation_id": ann_id,
                        "issue": "invalid_or_empty_mask",
                        "details": "该标注未参与切块。",
                    }
                )
                continue

            instances.append(instance)

        source_valid_instance_total += len(instances)

        root = Window(
            x=0,
            y=0,
            w=image_w,
            h=image_h,
            level=-1,
            reason="full_image",
        )
        base_windows = make_windows_in_region(
            root,
            tile_size=tile_sizes[0],
            overlap=overlap,
            level=0,
            reason="base",
        )

        final_windows: list[Window] = []
        overloads: list[dict[str, Any]] = []

        for base_window in base_windows:
            refine_window(
                base_window,
                level=0,
                tile_sizes=tile_sizes,
                overlap=overlap,
                instances=instances,
                visible_ratio=visible_ratio,
                min_mask_area=min_mask_area,
                max_instances=max_instances,
                final_windows=final_windows,
                overloads=overloads,
            )

        final_windows = deduplicate_windows(final_windows)

        if overloads:
            for overload in overloads:
                issue_rows.append(
                    {
                        "source_image_id": source_image_id,
                        "source_file_name": source_file_name,
                        "source_annotation_id": "",
                        "issue": "tile_still_overloaded",
                        "details": json.dumps(
                            overload,
                            ensure_ascii=False,
                        ),
                    }
                )
            raise RuntimeError(
                f"{split}/{source_file_name} 的最小图块仍超过 "
                f"{max_instances} 个实例。请增加更小的 --tile-sizes。"
            )

        final_windows, missing_ids = add_rescue_windows(
            final_windows,
            instances,
            tile_sizes,
            image_w,
            image_h,
            visible_ratio,
            rescue_visible_ratio,
            min_mask_area,
            max_instances,
        )

        if missing_ids:
            for ann_id in missing_ids:
                issue_rows.append(
                    {
                        "source_image_id": source_image_id,
                        "source_file_name": source_file_name,
                        "source_annotation_id": ann_id,
                        "issue": "instance_not_covered",
                        "details": (
                            "常规图块和覆盖补救图块均未能保留该实例。"
                        ),
                    }
                )
            if not allow_missing_instances:
                raise RuntimeError(
                    f"{split}/{source_file_name} 仍有 "
                    f"{len(missing_ids)} 个实例未被任何图块覆盖。"
                    "可降低 --visible-ratio、增加重叠率或增加更小图块。"
                )

        source_ann_occurrences: Counter[int] = Counter()
        source_tile_count = 0

        for window_index, window in enumerate(final_windows):
            ratio_threshold = (
                rescue_visible_ratio
                if window.reason == "coverage_rescue"
                else visible_ratio
            )
            selected = included_instances(
                instances,
                window,
                ratio_threshold,
                min_mask_area,
            )

            if len(selected) > max_instances:
                raise RuntimeError(
                    f"内部审计失败：{split}/{source_file_name} 图块 "
                    f"{window} 含 {len(selected)} 个实例，超过 "
                    f"{max_instances}。"
                )
            if not selected and not keep_empty:
                continue

            crop = image[
                window.y : window.y + window.h,
                window.x : window.x + window.w,
            ].copy()

            extension = output_extension(source_path, image_format)
            tile_name = (
                f"{safe_stem(source_file_name)}"
                f"__src{source_image_id:06d}"
                f"__x{window.x:04d}_y{window.y:04d}"
                f"__w{window.w:04d}_h{window.h:04d}"
                f"__l{window.level}{extension}"
            )
            output_path = output_images_dir / tile_name
            write_image(output_path, crop, jpg_quality)

            tile_image_id = next_image_id
            next_image_id += 1
            source_tile_count += 1

            new_images.append(
                {
                    "id": tile_image_id,
                    "file_name": tile_name,
                    "width": int(window.w),
                    "height": int(window.h),
                    "source_image_id": source_image_id,
                    "source_file_name": source_file_name,
                    "tile_x": int(window.x),
                    "tile_y": int(window.y),
                    "tile_width": int(window.w),
                    "tile_height": int(window.h),
                    "tile_level": int(window.level),
                    "tile_reason": window.reason,
                }
            )

            source_ids_in_tile: list[int] = []

            for instance, clipped_pixel_area, visible in selected:
                tile_mask = crop_instance_mask(instance, window)
                area = int(tile_mask.sum())
                if area < min_mask_area:
                    continue

                rle = mask_to_rle(tile_mask)
                bbox = [
                    float(value)
                    for value in mask_utils.toBbox(rle).tolist()
                ]
                encoded_area = float(mask_utils.area(rle))

                new_annotations.append(
                    {
                        "id": next_annotation_id,
                        "image_id": tile_image_id,
                        "category_id": int(instance.category_id),
                        "segmentation": rle,
                        "area": encoded_area,
                        "bbox": bbox,
                        "iscrowd": int(instance.iscrowd),
                        "source_image_id": source_image_id,
                        "source_annotation_id": int(instance.ann_id),
                        "source_bbox": instance.source_bbox,
                        "visible_area_ratio": float(visible),
                        "is_truncated": bool(visible < 0.999999),
                    }
                )
                next_annotation_id += 1
                source_ann_occurrences[instance.ann_id] += 1
                source_ids_in_tile.append(instance.ann_id)

            actual_count = len(source_ids_in_tile)
            if actual_count > max_instances:
                raise RuntimeError(
                    f"写出后实例数审计失败：{tile_name}={actual_count}"
                )

            tile_instance_counts.append(actual_count)
            tile_size_counter[f"{window.w}x{window.h}"] += 1
            if window.reason == "coverage_rescue":
                rescue_tile_count += 1

            manifest_rows.append(
                {
                    "split": split,
                    "tile_image_id": tile_image_id,
                    "tile_file_name": tile_name,
                    "source_image_id": source_image_id,
                    "source_file_name": source_file_name,
                    "tile_x": window.x,
                    "tile_y": window.y,
                    "tile_width": window.w,
                    "tile_height": window.h,
                    "tile_level": window.level,
                    "tile_reason": window.reason,
                    "instance_count": actual_count,
                    "source_annotation_ids": "|".join(
                        str(value) for value in sorted(source_ids_in_tile)
                    ),
                }
            )

        covered_count = sum(
            1
            for instance in instances
            if source_ann_occurrences[instance.ann_id] > 0
        )
        source_covered_instance_total += covered_count
        duplicate_training_instances += sum(
            max(0, count - 1)
            for count in source_ann_occurrences.values()
        )

        for instance in instances:
            coverage_rows.append(
                {
                    "split": split,
                    "source_image_id": source_image_id,
                    "source_file_name": source_file_name,
                    "source_annotation_id": instance.ann_id,
                    "category_id": instance.category_id,
                    "source_mask_area": instance.area,
                    "tile_occurrence_count": source_ann_occurrences[
                        instance.ann_id
                    ],
                    "covered": int(
                        source_ann_occurrences[instance.ann_id] > 0
                    ),
                }
            )

        if progress_every > 0 and (
            source_index % progress_every == 0
            or source_index == len(source_images)
        ):
            print(
                f"[{split}] {source_index}/{len(source_images)} "
                f"原图；累计图块={len(new_images)}；"
                f"累计实例标注={len(new_annotations)}"
            )

    output_coco = {
        "info": {
            **coco.get("info", {}),
            "description": (
                f"Adaptive tiled dataset generated from {split}; "
                f"max_instances={max_instances}; "
                f"tile_sizes={tile_sizes}; overlap={overlap}; "
                f"visible_ratio={visible_ratio}"
            ),
        },
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": new_images,
        "annotations": new_annotations,
    }
    save_json(output_split_dir / annotation_name, output_coco)

    write_csv(
        output_split_dir / "tile_manifest.csv",
        manifest_rows,
        [
            "split",
            "tile_image_id",
            "tile_file_name",
            "source_image_id",
            "source_file_name",
            "tile_x",
            "tile_y",
            "tile_width",
            "tile_height",
            "tile_level",
            "tile_reason",
            "instance_count",
            "source_annotation_ids",
        ],
    )
    write_csv(
        output_split_dir / "source_instance_coverage.csv",
        coverage_rows,
        [
            "split",
            "source_image_id",
            "source_file_name",
            "source_annotation_id",
            "category_id",
            "source_mask_area",
            "tile_occurrence_count",
            "covered",
        ],
    )
    write_csv(
        output_split_dir / "conversion_issues.csv",
        issue_rows,
        [
            "source_image_id",
            "source_file_name",
            "source_annotation_id",
            "issue",
            "details",
        ],
    )

    missing_total = source_valid_instance_total - source_covered_instance_total
    summary = {
        "split": split,
        "source_image_count": len(source_images),
        "source_annotation_count": len(source_annotations),
        "source_valid_instance_count": source_valid_instance_total,
        "output_tile_count": len(new_images),
        "output_annotation_count": len(new_annotations),
        "covered_source_instance_count": source_covered_instance_total,
        "missing_source_instance_count": missing_total,
        "source_instance_coverage": (
            source_covered_instance_total / source_valid_instance_total
            if source_valid_instance_total
            else 1.0
        ),
        "duplicate_training_instance_occurrences": (
            duplicate_training_instances
        ),
        "rescue_tile_count": rescue_tile_count,
        "max_instances_limit": max_instances,
        "actual_max_instances_per_tile": (
            max(tile_instance_counts) if tile_instance_counts else 0
        ),
        "mean_instances_per_tile": (
            float(np.mean(tile_instance_counts))
            if tile_instance_counts
            else 0.0
        ),
        "median_instances_per_tile": (
            float(np.median(tile_instance_counts))
            if tile_instance_counts
            else 0.0
        ),
        "instances_per_tile_p90": percentile(tile_instance_counts, 90),
        "instances_per_tile_p95": percentile(tile_instance_counts, 95),
        "instances_per_tile_p99": percentile(tile_instance_counts, 99),
        "empty_tile_count": sum(
            1 for count in tile_instance_counts if count == 0
        ),
        "tile_size_counts": dict(tile_size_counter),
        "issue_count": len(issue_rows),
        "issue_breakdown": dict(
            Counter(str(row["issue"]) for row in issue_rows)
        ),
        "parameters": {
            "tile_sizes": tile_sizes,
            "overlap": overlap,
            "visible_ratio": visible_ratio,
            "rescue_visible_ratio": rescue_visible_ratio,
            "min_mask_area": min_mask_area,
            "keep_empty": keep_empty,
            "include_crowd": include_crowd,
            "image_format": image_format,
        },
        "outputs": {
            "coco_json": str(output_split_dir / annotation_name),
            "images_dir": str(output_images_dir),
            "tile_manifest": str(
                output_split_dir / "tile_manifest.csv"
            ),
            "coverage_csv": str(
                output_split_dir / "source_instance_coverage.csv"
            ),
            "issues_csv": str(
                output_split_dir / "conversion_issues.csv"
            ),
        },
    }
    save_json(output_split_dir / "tiling_summary.json", summary)

    if (
        summary["actual_max_instances_per_tile"] > max_instances
        or (
            summary["missing_source_instance_count"] > 0
            and not allow_missing_instances
        )
    ):
        raise RuntimeError(
            f"{split} 输出审计失败：{json.dumps(summary, ensure_ascii=False)}"
        )

    return summary


def main() -> None:
    args = parse_args()
    tile_sizes = parse_tile_sizes(args.tile_sizes)
    validate_args(args, tile_sizes)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    if dataset_root == output_root:
        raise ValueError("--output-root 不能与 --dataset-root 相同。")
    if dataset_root in output_root.parents:
        print(
            "[提示] 输出目录位于原始数据集目录内部。"
            "建议使用并列目录，避免后续误扫描。",
            file=sys.stderr,
        )

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"输出目录已存在：{output_root}\n"
                "确认无需保留后，添加 --overwrite。"
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    splits = [
        item.strip()
        for item in args.splits.split(",")
        if item.strip()
    ]
    if not splits:
        raise ValueError("--splits 不能为空。")

    all_summaries: dict[str, Any] = {}
    skipped_splits: list[str] = []

    print("=" * 72)
    print("COCO实例数据集自适应切块")
    print(f"输入目录：{dataset_root}")
    print(f"输出目录：{output_root}")
    print(f"图块尺寸：{tile_sizes}")
    print(f"实例上限：{args.max_instances}")
    print(f"重叠率：{args.overlap}")
    print(f"常规可见比例：{args.visible_ratio}")
    print(f"补救可见比例：{args.rescue_visible_ratio}")
    print("=" * 72)

    for split in splits:
        input_split_dir = dataset_root / split
        annotation_path = input_split_dir / args.annotation_name

        if not annotation_path.is_file():
            print(f"[跳过] {split}：未找到 {annotation_path}")
            skipped_splits.append(split)
            continue

        print(f"\n[开始] split={split}")
        summary = process_split(
            split=split,
            input_split_dir=input_split_dir,
            output_split_dir=output_root / split,
            images_subdir=args.images_subdir,
            annotation_name=args.annotation_name,
            tile_sizes=tile_sizes,
            overlap=args.overlap,
            max_instances=args.max_instances,
            visible_ratio=args.visible_ratio,
            rescue_visible_ratio=args.rescue_visible_ratio,
            min_mask_area=args.min_mask_area,
            keep_empty=args.keep_empty,
            include_crowd=args.include_crowd,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
            allow_missing_instances=args.allow_missing_instances,
            progress_every=args.progress_every,
        )
        all_summaries[split] = summary
        print(
            f"[完成] {split}：图块={summary['output_tile_count']}；"
            f"输出标注={summary['output_annotation_count']}；"
            f"最大实例数={summary['actual_max_instances_per_tile']}；"
            f"原实例覆盖率={summary['source_instance_coverage']:.6f}"
        )

    global_summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "processed_splits": list(all_summaries),
        "skipped_splits": skipped_splits,
        "splits": all_summaries,
    }
    save_json(output_root / "tiling_dataset_summary.json", global_summary)

    print("\n" + "=" * 72)
    print("[全部完成]")
    print(f"总汇总：{output_root / 'tiling_dataset_summary.json'}")
    for split, summary in all_summaries.items():
        print(
            f"  {split}: tiles={summary['output_tile_count']}, "
            f"max_instances={summary['actual_max_instances_per_tile']}, "
            f"coverage={summary['source_instance_coverage']:.6f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
