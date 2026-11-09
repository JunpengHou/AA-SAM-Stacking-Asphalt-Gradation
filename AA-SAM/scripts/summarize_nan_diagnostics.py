from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Aggregate-SAM3 NaN diagnostic JSONL and map COCO IDs to filenames."
    )
    parser.add_argument("--nan-jsonl", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def flatten_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "values" in value:
            return flatten_values(value["values"])
        result: list[Any] = []
        for item in value.values():
            result.extend(flatten_values(item))
        return result
    if isinstance(value, (list, tuple)):
        result: list[Any] = []
        for item in value:
            result.extend(flatten_values(item))
        return result
    return [value]


def first_nonempty(values: list[Any]) -> str:
    cleaned = [str(value) for value in values if value not in (None, "", -1, "-1")]
    return "|".join(cleaned)


def main() -> None:
    args = parse_args()
    nan_jsonl = args.nan_jsonl.expanduser().resolve()
    annotation_file = args.annotation_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not nan_jsonl.is_file():
        raise FileNotFoundError(nan_jsonl)
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)

    with annotation_file.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)

    image_map = {
        str(item["id"]): {
            "file_name": item.get("file_name", ""),
            "width": item.get("width"),
            "height": item.get("height"),
        }
        for item in coco.get("images", [])
    }

    records: list[dict[str, Any]] = []
    with nan_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                report = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "event": line_number,
                        "parse_error": str(exc),
                        "raw_line": raw_line[:500],
                    }
                )
                continue

            metadata = report.get("metadata", []) or []
            coco_ids: list[Any] = []
            original_ids: list[Any] = []
            object_ids: list[Any] = []
            for item in metadata:
                coco_ids.extend(flatten_values(item.get("coco_image_id")))
                original_ids.extend(flatten_values(item.get("original_image_id")))
                object_ids.extend(flatten_values(item.get("object_id")))

            coco_id_text = first_nonempty(coco_ids)
            original_id_text = first_nonempty(original_ids)
            object_id_text = first_nonempty(object_ids)

            filenames: list[str] = []
            dimensions: list[str] = []
            for coco_id in coco_ids:
                key = str(coco_id)
                info = image_map.get(key)
                if info is None:
                    continue
                filename = str(info.get("file_name", ""))
                if filename and filename not in filenames:
                    filenames.append(filename)
                dimension = f"{info.get('width')}x{info.get('height')}"
                if dimension not in dimensions:
                    dimensions.append(dimension)

            nonfinite_components: list[str] = []
            component_values: list[str] = []
            for name, info in (report.get("loss_components", {}) or {}).items():
                finite = info.get("finite")
                value = info.get("value")
                component_values.append(f"{name}={value}")
                if finite is False:
                    nonfinite_components.append(str(name))

            targets = report.get("targets", []) or []
            nonpositive_width = sum(
                int(item.get("nonpositive_width_count", 0) or 0) for item in targets
            )
            nonpositive_height = sum(
                int(item.get("nonpositive_height_count", 0) or 0) for item in targets
            )
            nonfinite_boxes = sum(
                int(item.get("nonfinite_box_count", 0) or 0) for item in targets
            )
            empty_masks = sum(
                int(item.get("empty_mask_count", 0) or 0) for item in targets
            )

            min_widths = [
                item.get("min_width")
                for item in targets
                if item.get("min_width") is not None
            ]
            min_heights = [
                item.get("min_height")
                for item in targets
                if item.get("min_height") is not None
            ]
            min_mask_pixels = [
                item.get("min_mask_pixels")
                for item in targets
                if item.get("min_mask_pixels") is not None
            ]

            records.append(
                {
                    "event": line_number,
                    "epoch": report.get("epoch"),
                    "step": report.get("step"),
                    "chunk_index": report.get("chunk_index"),
                    "reason": report.get("reason"),
                    "loss": report.get("loss"),
                    "coco_image_id": coco_id_text,
                    "original_image_id": original_id_text,
                    "file_name": "|".join(filenames),
                    "image_size": "|".join(dimensions),
                    "object_id": object_id_text,
                    "nonfinite_loss_components": "|".join(nonfinite_components),
                    "loss_components": "; ".join(component_values),
                    "nonpositive_width_count": nonpositive_width,
                    "nonpositive_height_count": nonpositive_height,
                    "nonfinite_box_count": nonfinite_boxes,
                    "empty_mask_count": empty_masks,
                    "minimum_width": min(min_widths) if min_widths else None,
                    "minimum_height": min(min_heights) if min_heights else None,
                    "minimum_mask_pixels": min(min_mask_pixels) if min_mask_pixels else None,
                    "nonfinite_gradient_parameters": "|".join(
                        report.get("nonfinite_gradient_parameters", []) or []
                    ),
                    "saved_augmented_images": "|".join(
                        report.get("saved_augmented_images", []) or []
                    ),
                    "parse_error": "",
                    "raw_line": "",
                }
            )

    events_path = output_dir / "nan_events.csv"
    fieldnames = sorted({key for record in records for key in record.keys()})
    with events_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    filename_counter = Counter()
    coco_counter = Counter()
    reason_counter = Counter()
    component_counter = Counter()

    for record in records:
        reason = record.get("reason")
        if reason:
            reason_counter[str(reason)] += 1

        for filename in str(record.get("file_name", "")).split("|"):
            if filename:
                filename_counter[filename] += 1

        for coco_id in str(record.get("coco_image_id", "")).split("|"):
            if coco_id:
                coco_counter[coco_id] += 1

        for component in str(record.get("nonfinite_loss_components", "")).split("|"):
            if component:
                component_counter[component] += 1

    repeated_path = output_dir / "repeated_nan_images.csv"
    with repeated_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file_name", "coco_image_id", "nan_event_count"],
        )
        writer.writeheader()

        all_keys = set(filename_counter) | set(coco_counter)
        for key in sorted(
            all_keys,
            key=lambda item: max(filename_counter[item], coco_counter[item]),
            reverse=True,
        ):
            writer.writerow(
                {
                    "file_name": key if key in filename_counter else "",
                    "coco_image_id": key if key in coco_counter else "",
                    "nan_event_count": max(filename_counter[key], coco_counter[key]),
                }
            )

    summary = {
        "nan_event_count": len(records),
        "reason_counts": dict(reason_counter),
        "nonfinite_loss_component_counts": dict(component_counter),
        "repeated_filename_counts": dict(filename_counter.most_common()),
        "repeated_coco_image_id_counts": dict(coco_counter.most_common()),
        "outputs": {
            "nan_events_csv": str(events_path),
            "repeated_nan_images_csv": str(repeated_path),
        },
    }

    summary_path = output_dir / "nan_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
