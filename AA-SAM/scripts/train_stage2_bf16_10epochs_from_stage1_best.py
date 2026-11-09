from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from typing import Any

# Windows DLL initialization workaround: Pillow first.
from PIL import Image  # noqa: F401

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BEST_METRICS = {
    "ap": ("coco_eval_segm_AP", "mask_AP"),
    "ap50": ("coco_eval_segm_AP_50", "mask_AP50"),
    "ap75": ("coco_eval_segm_AP_75", "mask_AP75"),
}
BEST_VAL_KEY = "aggregate_particle"
BEST_METER_NAME = "segmentation"
BEST_HELPER_MODULE = "aggregate_sam3_best_metric_generated"
STAGE2_MODEL_HELPER_MODULE = "aggregate_sam3_stage2_model_generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Stage-2 fine-tuning of Aggregate-SAM3 from a Stage-1 Trainer checkpoint.")
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        required=True,
        help=(
            "第一阶段Aggregate-SAM3最佳Trainer权重。推荐使用"
            "checkpoints/val_aggregate_particle_segmentation/mask_AP50.pt。"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-prompt",
        type=str,
        default="aggregate particle",
        help="检查COCO图像记录中的text_input；默认aggregate particle。",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help=(
            "SAM3视觉编码器输入分辨率。当前官方SAM3权重的ViT/RoPE固定对应"
            "1008×1008（patch_size=14，特征网格72×72），因此必须使用1008。"
        ),
    )
    parser.add_argument(
        "--freeze-mode",
        choices=["adapter", "vision_frozen", "full"],
        default="adapter",
    )
    parser.add_argument("--branch-base-channels", type=int, default=32)
    parser.add_argument(
        "--max-train-instances",
        type=int,
        default=180,
        help=(
            "每张训练图最多参与匹配和损失计算的实例数。SAM3当前解码器只有"
            "200个object queries，因此建议不超过180；默认180。该参数仅用于"
            "稳定性诊断，正式训练建议先将密集原图切块，而不是长期丢弃实例。"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--amp-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help=(
            "第二阶段推荐bfloat16；如出现连续NaN/Inf，可改为float32。"
            "本脚本禁止float16。"
        ),
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=0.05,
        help="梯度裁剪最大范数，稳定版默认0.05。",
    )
    parser.add_argument(
        "--best-metric",
        choices=sorted(BEST_METRICS),
        default="ap50",
        help=(
            "选择最佳权重的验证指标：ap=Mask mAP，ap50=Mask AP50，"
            "ap75=Mask AP75；默认ap50。"
        ),
    )
    parser.add_argument(
        "--best-max-dets",
        type=int,
        default=1000,
        help="验证时每张图最多保留的集料实例数，默认1000。",
    )
    parser.add_argument(
        "--best-checkpoint-name",
        type=str,
        default="best_checkpoint.pt",
        help="最佳权重的固定文件名，默认best_checkpoint.pt。",
    )
    parser.add_argument(
        "--allow-train-as-val",
        action="store_true",
        help="When val split is missing, reuse train split only for smoke testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate and validate the config; do not start training.",
    )
    return parser.parse_args()


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in folder.iterdir()
    )


def _json_candidates(split_root: Path, split_name: str) -> list[Path]:
    """
    Return possible COCO annotation filenames.

    Supported examples:
      train/aggregate_train_sam3.json
      val/aggregate_val_sam3.json
      test/aggregate_test_sam3.json
      train/_annotations.coco.json
      annotations/train.json
    """
    canonical_name = "val" if split_name == "valid" else split_name

    return [
        split_root / f"aggregate_{canonical_name}_sam3.json",
        split_root / "_annotations.coco.json",
        # 兼容先前由泥块数据转换脚本生成的旧文件名；内部类别仍以JSON为准。
        split_root / f"mud_{canonical_name}_sam3.json",
        split_root / "annotations.coco.json",
        split_root / "annotations.json",
        split_root / f"{canonical_name}.json",
        split_root / f"instances_{canonical_name}.json",
        split_root.parent / "annotations" / f"aggregate_{canonical_name}_sam3.json",
        split_root.parent / "annotations" / f"mud_{canonical_name}_sam3.json",
        split_root.parent / "annotations" / f"{canonical_name}.json",
        split_root.parent / f"aggregate_{canonical_name}_sam3.json",
        split_root.parent / f"{canonical_name}.json",
    ]


def resolve_split(dataset_root: Path, split_name: str) -> tuple[Path, Path] | None:
    """
    Resolve a dataset split while supporting both layouts:

    A) dataset/train/images/*.bmp + dataset/train/aggregate_train_sam3.json
    B) dataset/train/*.bmp        + dataset/train/aggregate_train_sam3.json
    """
    canonical_name = "val" if split_name == "valid" else split_name

    split_root_candidates = [
        dataset_root / canonical_name,
        dataset_root / "images" / canonical_name,
    ]

    for split_root in split_root_candidates:
        if not split_root.is_dir():
            continue

        ann_file = next(
            (
                path
                for path in _json_candidates(split_root, canonical_name)
                if path.is_file()
            ),
            None,
        )
        if ann_file is None:
            continue

        image_candidates = [
            split_root / "images",
            split_root,
            dataset_root / "images",
        ]

        for image_folder in image_candidates:
            if not image_folder.is_dir():
                continue

            # Prefer a folder that actually contains at least one image.
            # Empty validation/test folders are handled later by
            # --allow-train-as-val rather than being treated as train data.
            if _count_images(image_folder) > 0:
                return image_folder.resolve(), ann_file.resolve()

    return None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_coco(
    image_folder: Path,
    ann_file: Path,
    expected_prompt: str | None = "aggregate particle",
) -> dict[str, int]:
    """严格检查会导致匈牙利匹配代价矩阵出现NaN/Inf的COCO问题。"""
    with ann_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required = {"images", "annotations", "categories"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{ann_file}缺少COCO字段：{sorted(missing)}")

    errors: list[str] = []
    normalized_expected_prompt = (expected_prompt or "").strip().casefold()
    if normalized_expected_prompt:
        prompt_values = {
            str(item.get("text_input", "")).strip().casefold()
            for item in data["images"]
        }
        if prompt_values != {normalized_expected_prompt}:
            errors.append(
                "COCO images中的text_input与期望提示不一致："
                f"期望={expected_prompt!r}，实际={sorted(prompt_values)}"
            )

    image_by_id: dict[Any, dict[str, Any]] = {}
    missing_images: list[Any] = []

    for index, image in enumerate(data["images"]):
        image_id = image.get("id")
        filename = image.get("file_name")
        width = image.get("width")
        height = image.get("height")

        if image_id is None:
            errors.append(f"images[{index}]缺少id")
            continue
        if image_id in image_by_id:
            errors.append(f"重复的image id：{image_id}")
            continue
        image_by_id[image_id] = image

        if not filename:
            errors.append(f"image_id={image_id}缺少file_name")
            missing_images.append(filename)
        else:
            candidates = [
                image_folder / filename,
                image_folder / Path(filename).name,
            ]
            if not any(path.is_file() for path in candidates):
                missing_images.append(filename)

        if not _is_finite_number(width) or float(width) <= 0:
            errors.append(
                f"image_id={image_id}, file={filename}的width非法：{width}"
            )
        if not _is_finite_number(height) or float(height) <= 0:
            errors.append(
                f"image_id={image_id}, file={filename}的height非法：{height}"
            )

    if missing_images:
        sample = missing_images[:10]
        errors.append(
            f"有{len(missing_images)}个图像文件找不到，示例：{sample}"
        )

    category_ids = {
        category.get("id")
        for category in data["categories"]
        if category.get("id") is not None
    }
    annotation_ids: set[Any] = set()
    segmentation_count = 0

    for index, annotation in enumerate(data["annotations"]):
        ann_id = annotation.get("id", f"index_{index}")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")

        if ann_id in annotation_ids:
            errors.append(f"重复的annotation id：{ann_id}")
        annotation_ids.add(ann_id)

        image_info = image_by_id.get(image_id)
        if image_info is None:
            errors.append(
                f"annotation id={ann_id}引用不存在的image_id={image_id}"
            )
            continue

        filename = image_info.get("file_name")
        image_width = float(image_info.get("width", 0) or 0)
        image_height = float(image_info.get("height", 0) or 0)

        if category_ids and category_id not in category_ids:
            errors.append(
                f"annotation id={ann_id}, file={filename}的category_id={category_id}不存在"
            )

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(
                f"annotation id={ann_id}, file={filename}的bbox不是[x,y,w,h]：{bbox}"
            )
        elif not all(_is_finite_number(value) for value in bbox):
            errors.append(
                f"annotation id={ann_id}, file={filename}的bbox含NaN/Inf或非数字：{bbox}"
            )
        else:
            x, y, width, height = map(float, bbox)
            if width <= 0 or height <= 0:
                errors.append(
                    f"annotation id={ann_id}, file={filename}为零/负面积框：{bbox}"
                )
            if x < 0 or y < 0:
                errors.append(
                    f"annotation id={ann_id}, file={filename}的bbox左上角越界：{bbox}"
                )
            tolerance = 1.0
            if image_width > 0 and x + width > image_width + tolerance:
                errors.append(
                    f"annotation id={ann_id}, file={filename}的bbox超出图像宽度："
                    f"bbox={bbox}, image_width={image_width}"
                )
            if image_height > 0 and y + height > image_height + tolerance:
                errors.append(
                    f"annotation id={ann_id}, file={filename}的bbox超出图像高度："
                    f"bbox={bbox}, image_height={image_height}"
                )

        area = annotation.get("area")
        if not _is_finite_number(area) or float(area) <= 0:
            errors.append(
                f"annotation id={ann_id}, file={filename}的area非法：{area}"
            )

        segmentation = annotation.get("segmentation")
        if not segmentation:
            errors.append(
                f"annotation id={ann_id}, file={filename}的segmentation为空"
            )
            continue

        if isinstance(segmentation, list):
            valid_polygon = False
            for polygon_index, polygon in enumerate(segmentation):
                if not isinstance(polygon, list):
                    errors.append(
                        f"annotation id={ann_id}的polygon[{polygon_index}]不是列表"
                    )
                    continue
                if len(polygon) < 6 or len(polygon) % 2 != 0:
                    errors.append(
                        f"annotation id={ann_id}, file={filename}的polygon坐标数量非法："
                        f"len={len(polygon)}"
                    )
                    continue
                if not all(_is_finite_number(value) for value in polygon):
                    errors.append(
                        f"annotation id={ann_id}, file={filename}的polygon含NaN/Inf"
                    )
                    continue
                valid_polygon = True
            if valid_polygon:
                segmentation_count += 1
        elif isinstance(segmentation, dict):
            size = segmentation.get("size")
            counts = segmentation.get("counts")
            if (
                not isinstance(size, list)
                or len(size) != 2
                or not all(_is_finite_number(value) and float(value) > 0 for value in size)
                or counts is None
            ):
                errors.append(
                    f"annotation id={ann_id}, file={filename}的RLE segmentation非法"
                )
            else:
                segmentation_count += 1
        else:
            errors.append(
                f"annotation id={ann_id}, file={filename}的segmentation类型非法："
                f"{type(segmentation).__name__}"
            )

        if len(errors) >= 100:
            break

    if data["annotations"] and segmentation_count == 0:
        errors.append(f"{ann_file}没有可用的segmentation，无法训练实例分割。")

    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:30])
        extra = "" if len(errors) <= 30 else f"\n  ……另有{len(errors) - 30}项错误"
        raise ValueError(
            f"COCO数据严格检查失败：{ann_file}\n{preview}{extra}\n"
            "请重新生成或修复标注后再训练。"
        )

    return {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": len(data["categories"]),
        "segmentations": segmentation_count,
    }


def locate_official_sam3(sam3_root: Path) -> tuple[Path, Path, Path]:
    sam3_root = sam3_root.resolve()
    package_dir = sam3_root / "sam3"
    model_builder = package_dir / "model_builder.py"
    train_script = package_dir / "train" / "train.py"
    trainer_script = package_dir / "train" / "trainer.py"
    bpe_path = package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    missing = [
        path
        for path in [model_builder, train_script, trainer_script, bpe_path]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "官方SAM3源码不完整，缺少：\n"
            + "\n".join(str(path) for path in missing)
        )

    trainer_source = trainer_script.read_text(encoding="utf-8", errors="ignore")
    if "save_best_meters" not in trainer_source:
        raise RuntimeError(
            "当前官方SAM3 trainer.py不支持save_best_meters，无法在训练过程中"
            "自动保存最佳权重。请更新官方SAM3源码后重试。"
        )

    local_shadow = Path(__file__).resolve().parent / "sam3"
    if local_shadow.exists():
        raise RuntimeError(
            f"检测到会覆盖官方包的局部目录：{local_shadow}\n"
            "请将其删除或重命名，例如sam3_backup。"
        )

    return package_dir, train_script, bpe_path



def write_best_metric_helper(output_dir: Path) -> Path:
    """Generate an importable validation meter used by Hydra/SAM3 Trainer."""
    helper_dir = output_dir / "generated_helpers"
    helper_dir.mkdir(parents=True, exist_ok=True)
    helper_path = helper_dir / f"{BEST_HELPER_MODULE}.py"
    helper_code = textwrap.dedent(
        r'''from __future__ import annotations

import json
import logging
from pathlib import Path

from sam3.eval.coco_writer import PredictionDumper
from sam3.train.utils.distributed import is_main_process


class BestSegmentationPredictionDumper(PredictionDumper):
    """COCO segmentation evaluator exposing one maximized metric to Trainer."""

    def __init__(
        self,
        *args,
        metric_key: str = "coco_eval_segm_AP_50",
        output_key: str = "mask_AP50",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric_key = metric_key
        self.output_key = output_key

    @staticmethod
    def is_better(current: float, best: float) -> bool:
        return float(current) > float(best)

    def compute_synced(self):
        dumped_file = self.synchronize_between_processes()
        if not is_main_process():
            return {self.output_key: 0.0}

        dumped_file = Path(dumped_file)
        try:
            predictions = json.loads(dumped_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("无法读取验证预测文件%s：%s", dumped_file, exc)
            predictions = []

        # pycocotools.COCO.loadRes cannot evaluate an empty prediction list.
        if not predictions:
            metric_value = 0.0
            all_metrics = {self.metric_key: metric_value}
        else:
            all_metrics = {}
            for evaluator in self.pred_file_evaluators or []:
                all_metrics.update(evaluator.evaluate(dumped_file))
            metric_value = float(all_metrics.get(self.metric_key, 0.0))

        metrics_path = Path(self.dump_dir) / "latest_validation_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "selected_metric": self.metric_key,
                    "selected_value": metric_value,
                    "all_metrics": all_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {self.output_key: metric_value}
'''
    ).strip() + "\n"
    helper_path.write_text(helper_code, encoding="utf-8")
    return helper_dir



def write_stage2_model_helper(output_dir: Path) -> Path:
    """
    Generate a model builder for Stage-2 fine-tuning.

    Important:
      - base_checkpoint_path is still the official sam3.pt used to construct SAM3;
      - stage1_checkpoint_path is a Trainer checkpoint produced by Stage-1;
      - only checkpoint["model"] is inherited;
      - optimizer / LR scheduler are deliberately re-created for Stage-2.
    """
    helper_dir = output_dir / "generated_helpers"
    helper_dir.mkdir(parents=True, exist_ok=True)
    helper_path = helper_dir / f"{STAGE2_MODEL_HELPER_MODULE}.py"

    helper_code = textwrap.dedent(
        r"""from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image  # noqa: F401
import torch

from aggregate_sam3.model import build_aggregate_sam3_image_model


def _torch_load(path: str | Path):
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _extract_stage1_model_state(checkpoint: Any):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "第一阶段权重必须是SAM3 Trainer保存的dict checkpoint，"
            f"当前类型={type(checkpoint)!r}"
        )

    if "model" not in checkpoint:
        raise KeyError(
            "第一阶段checkpoint中没有'model'字段。"
            "请不要把官方sam3.pt误传给--stage1-checkpoint；"
            "应传入第一阶段的mask_AP50.pt或checkpoint_N.pt。"
        )

    state = checkpoint["model"]
    if not isinstance(state, dict):
        raise TypeError(
            "checkpoint['model']不是state_dict，"
            f"当前类型={type(state)!r}"
        )

    if state and all(
        str(key).startswith("module.") for key in state.keys()
    ):
        state = {
            str(key)[len("module."):]: value
            for key, value in state.items()
        }
    return state


def build_stage2_aggregate_sam3_image_model(
    bpe_path: str | None = None,
    device: str = "cpu",
    eval_mode: bool = False,
    base_checkpoint_path: str | None = None,
    stage1_checkpoint_path: str | None = None,
    load_from_HF: bool = False,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    freeze_mode: str = "adapter",
    branch_base_channels: int = 32,
    fuse_sam2_features: bool = False,
):
    if not base_checkpoint_path:
        raise ValueError("base_checkpoint_path不能为空。")
    if not stage1_checkpoint_path:
        raise ValueError("stage1_checkpoint_path不能为空。")

    # 1) Construct exactly the same Aggregate-SAM3 architecture as Stage-1.
    model = build_aggregate_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=False,
        checkpoint_path=base_checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
        freeze_mode=freeze_mode,
        branch_base_channels=branch_base_channels,
        fuse_sam2_features=fuse_sam2_features,
    )

    # 2) Replace its parameters with the learned Stage-1 parameters.
    checkpoint = _torch_load(stage1_checkpoint_path)
    state = _extract_stage1_model_state(checkpoint)

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if missing or unexpected:
        raise RuntimeError(
            "第一阶段权重与当前Aggregate-SAM3结构不完全一致。"
            "为避免错误微调，Stage-2已停止。\n"
            f"missing({len(missing)}): {missing[:30]}\n"
            f"unexpected({len(unexpected)}): {unexpected[:30]}"
        )

    source_epoch = checkpoint.get("epoch")
    source_steps = checkpoint.get("steps")
    source_best = checkpoint.get("best_meter_values", {})

    model._stage2_source_checkpoint = str(stage1_checkpoint_path)
    model._stage2_source_epoch = source_epoch
    model._stage2_source_steps = source_steps

    print(
        "[Stage-2初始化] 第一阶段模型参数加载成功："
        f"{stage1_checkpoint_path}"
    )
    print(
        "[Stage-2初始化] 来源checkpoint元数据："
        f"epoch={source_epoch}, steps={source_steps}"
    )
    print(
        "[Stage-2初始化] 第一阶段best_meter_values="
        f"{source_best}"
    )
    print(
        "[Stage-2初始化] 只继承model参数；"
        "AdamW、学习率调度器将在Stage-2重新初始化。"
    )

    del checkpoint, state

    if eval_mode:
        model.eval()
    return model
"""
    ).strip() + "\n"

    helper_path.write_text(helper_code, encoding="utf-8")
    return helper_path



def get_best_checkpoint_dir(args: argparse.Namespace) -> Path:
    """返回官方Trainer为最佳验证指标使用的检查点目录。"""
    return (
        args.output_dir
        / "checkpoints"
        / f"val_{BEST_VAL_KEY}_{BEST_METER_NAME}"
    )


def prepare_output_directories(args: argparse.Namespace) -> None:
    """提前创建训练器可能写入的目录，避免Windows下.tmp保存失败。"""
    checkpoint_dir = args.output_dir / "checkpoints"
    best_checkpoint_dir = get_best_checkpoint_dir(args)
    validation_dir = args.output_dir / "validation_predictions"
    log_dir = args.output_dir / "logs"
    tensorboard_dir = args.output_dir / "tensorboard"

    for folder in (
        checkpoint_dir,
        best_checkpoint_dir,
        validation_dir,
        log_dir,
        tensorboard_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    # 仅清理上次保存失败遗留的临时权重文件。
    for tmp_file in best_checkpoint_dir.glob("*.pt.tmp"):
        try:
            tmp_file.unlink()
            print(f"[清理临时文件] {tmp_file}")
        except OSError as exc:
            print(f"[警告] 无法删除临时文件{tmp_file}：{exc}")

    print(f"[检查点目录] {checkpoint_dir}")
    print(f"[最佳权重目录] {best_checkpoint_dir}")


def copy_best_checkpoint(args: argparse.Namespace) -> Path:
    _, output_key = BEST_METRICS[args.best_metric]
    checkpoint_dir = args.output_dir / "checkpoints"
    best_checkpoint_dir = get_best_checkpoint_dir(args)

    # 当前Trainer通常保存为：
    # checkpoints/val_aggregate_particle_segmentation/mask_AP50.pt
    # 同时兼容部分旧版Trainer的平面命名方式。
    source_candidates = [
        best_checkpoint_dir / f"{output_key}.pt",
        checkpoint_dir
        / f"val_{BEST_VAL_KEY}_{BEST_METER_NAME}_{output_key}.pt",
    ]

    source_path = next(
        (candidate for candidate in source_candidates if candidate.is_file()),
        None,
    )

    if source_path is None:
        recursive_candidates = sorted(
            path
            for path in checkpoint_dir.rglob("*.pt")
            if output_key.lower() in path.name.lower()
        )
        if recursive_candidates:
            source_path = recursive_candidates[0]

    requested_name = Path(args.best_checkpoint_name)
    if requested_name.name != args.best_checkpoint_name:
        raise ValueError("--best-checkpoint-name只能是文件名，不能包含目录。")
    if requested_name.suffix.lower() != ".pt":
        raise ValueError("--best-checkpoint-name必须以.pt结尾。")
    target_path = checkpoint_dir / requested_name.name

    if source_path is None or not source_path.is_file():
        candidates = sorted(checkpoint_dir.rglob("*.pt"))
        raise FileNotFoundError(
            "训练已结束，但没有找到最佳指标检查点。"
            f"预期位置：{[str(path) for path in source_candidates]}；"
            f"现有检查点：{[str(path) for path in candidates]}"
        )

    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)

    info = {
        "selection_metric": args.best_metric,
        "metric_tensor_name": BEST_METRICS[args.best_metric][0],
        "source_checkpoint": str(source_path),
        "best_checkpoint": str(target_path),
        "validation_metrics_file": str(
            args.output_dir
            / "validation_predictions"
            / "latest_validation_metrics.json"
        ),
    }
    try:
        import torch

        try:
            checkpoint = torch.load(
                source_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(source_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            info["epoch"] = checkpoint.get("epoch")
            info["steps"] = checkpoint.get("steps")
            info["best_meter_values"] = checkpoint.get(
                "best_meter_values",
                {},
            )
    except Exception as exc:
        info["checkpoint_metadata_warning"] = str(exc)

    info_path = checkpoint_dir / "best_checkpoint_info.json"
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[最佳权重] 指标={args.best_metric}")
    print(f"[最佳权重] 原始文件：{source_path}")
    print(f"[最佳权重] 已保存：{target_path}")
    print(f"[最佳权重] 信息文件：{info_path}")
    return target_path


def compose_transforms(resolution: int, training: bool) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = []

    if training:
        transforms.extend(
            [
                {
                    "_target_": (
                        "sam3.train.transforms.filter_query_transforms."
                        "FlexibleFilterFindGetQueries"
                    ),
                    "query_filter": {
                        "_target_": (
                            "sam3.train.transforms.filter_query_transforms."
                            "FilterCrowds"
                        )
                    },
                },
                {
                    "_target_": (
                        "sam3.train.transforms.point_sampling.RandomizeInputBbox"
                    ),
                    "box_noise_std": 0.1,
                    "box_noise_max": 20,
                },
            ]
        )

    transforms.append(
        {"_target_": "sam3.train.transforms.segmentation.DecodeRle"}
    )

    if training:
        resize_sizes = sorted(
            {
                max(480, int(resolution * ratio) // 14 * 14)
                for ratio in (0.67, 0.75, 0.83, 0.92, 1.0)
            }
        )
        transforms.append(
            {
                "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                "sizes": resize_sizes,
                "max_size": resolution,
                "consistent_transform": False,
            }
        )
        transforms.append(
            {
                "_target_": "sam3.train.transforms.basic_for_api.PadToSizeAPI",
                "size": resolution,
                "consistent_transform": False,
            }
        )
    else:
        transforms.append(
            {
                "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                "sizes": resolution,
                "max_size": resolution,
                "consistent_transform": False,
            }
        )
        # AGGREGATE_SAM3_VAL_PAD_TO_SIZE_FIX_V3
        transforms.append(
            {
                "_target_": "sam3.train.transforms.basic_for_api.PadToSizeAPI",
                "size": resolution,
                "consistent_transform": False,
            }
        )

    transforms.extend(
        [
            {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
            {
                "_target_": (
                    "sam3.train.transforms.filter_query_transforms."
                    "FlexibleFilterFindGetQueries"
                ),
                "query_filter": {
                    "_target_": (
                        "sam3.train.transforms.filter_query_transforms."
                        "FilterEmptyTargets"
                    )
                },
            },
            {
                "_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI",
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
        ]
    )

    return [
        {
            "_target_": "sam3.train.transforms.basic_for_api.ComposeAPI",
            "transforms": transforms,
        }
    ]


def build_loss_config() -> dict[str, Any]:
    return {
        "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",
        "normalization": "local",  # AGGREGATE_SAM3_SINGLE_GPU_LOCAL_LOSS_NORMALIZATION
        "matcher": {
            "_target_": "sam3.train.matcher.BinaryHungarianMatcherV2",
            "focal": True,
            "cost_class": 2.0,
            "cost_bbox": 5.0,
            "cost_giou": 2.0,
            "alpha": 0.25,
            "gamma": 2,
            "stable": False,
        },
        "o2m_weight": 2.0,
        "o2m_matcher": {
            "_target_": "sam3.train.matcher.BinaryOneToManyMatcher",
            "alpha": 0.3,
            "threshold": 0.4,
            "topk": 4,
        },
        "use_o2m_matcher_on_o2m_aux": False,
        "loss_fns_find": [
            {
                "_target_": "sam3.train.loss.loss_fns.Boxes",
                "weight_dict": {
                    "loss_bbox": 5.0,
                    "loss_giou": 2.0,
                },
            },
            {
                "_target_": "sam3.train.loss.loss_fns.IABCEMdetr",
                "weak_loss": False,
                "weight_dict": {
                    "loss_ce": 20.0,
                    "presence_loss": 20.0,
                },
                "pos_weight": 10.0,
                "alpha": 0.25,
                "gamma": 2,
                "use_presence": True,
                "pos_focal": False,
                "pad_n_queries": 200,
                "pad_scale_pos": 1.0,
            },
            {
                "_target_": "sam3.train.loss.loss_fns.Masks",
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "weight_dict": {
                    "loss_mask": 100.0,
                    "loss_dice": 10.0,
                },
                "compute_aux": False,
            },
        ],
        "loss_fn_semantic_seg": None,
        "scale_by_find_batch_size": True,
    }


def build_dataset_config(
    image_folder: Path,
    ann_file: Path,
    transforms: list[dict[str, Any]],
    batch_size: int,
    workers: int,
    training: bool,
    max_train_instances: int = 180,
) -> dict[str, Any]:
    return {
        "_target_": "sam3.train.data.torch_dataset.TorchDataset",
        "dataset": {
            "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
            "transforms": transforms,
            "load_segmentation": True,
            # The image decoder exposes 200 object queries. Feeding 839 targets
            # into one image is a capacity mismatch and can destabilize matching
            # and backward propagation. Limit training only; validation remains
            # untruncated for diagnostic evaluation.
            "max_ann_per_img": max_train_instances if training else 5000,
            "multiplier": 1,
            "max_train_queries": max_train_instances,
            "max_val_queries": 5000,
            "training": training,
            "use_caching": False,
            "img_folder": str(image_folder),
            "ann_file": str(ann_file),
            "coco_json_loader": {
                "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
                "include_negatives": True,
                "category_chunk_size": 1,
                "_partial_": True,
            },
        },
        "shuffle": training,
        "batch_size": batch_size if training else 1,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": training,
        "collate_fn": {
            "_target_": "sam3.train.data.collator.collate_fn_api",
            "_partial_": True,
            "repeats": 1,
            # Validation uses a named key so its COCO meter receives predictions.
            # DummyLoss is retained for validation; model selection uses Mask AP.
            "dict_key": "all" if training else BEST_VAL_KEY,
            "with_seg_masks": True,
        },
    }


def build_config(
    args: argparse.Namespace,
    train_split: tuple[Path, Path],
    val_split: tuple[Path, Path],
    bpe_path: Path,
) -> dict[str, Any]:
    train_images, train_json = train_split
    val_images, val_json = val_split
    output_dir = args.output_dir.resolve()

    lr_scheduler = {
        "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
        "base_lr": args.learning_rate,
        "timescale": max(20, args.epochs),
        "warmup_steps": min(20, max(1, args.epochs)),
        "cooldown_steps": min(20, max(1, args.epochs)),
    }

    config: dict[str, Any] = {
        "trainer": {
            "_target_": "aggregate_sam3.single_gpu_trainer.SingleGPUTrainer",
            "skip_saving_ckpts": False,
            "empty_gpu_mem_cache_after_eval": True,
            "skip_first_val": False,
            "max_epochs": args.epochs,
            "accelerator": "cuda",
            "seed_value": args.seed,
            "val_epoch_freq": args.val_every,
            "mode": "train",
            "gradient_accumulation_steps": args.grad_accum,
            "distributed": {
                # Windows does not provide NCCL. Gloo works for one-GPU local runs.
                "backend": "gloo",
                "find_unused_parameters": True,
                "gradient_as_bucket_view": True,
                # 单GPU兼容启动器会禁用DDP；此字段仅保留配置兼容性。
            },
            "loss": {
                "all": build_loss_config(),
                "default": {
                    "_target_": "sam3.train.loss.sam3_loss.DummyLoss"
                },
            },
            "data": {
                "train": build_dataset_config(
                    train_images,
                    train_json,
                    compose_transforms(args.resolution, training=True),
                    args.batch_size,
                    args.num_workers,
                    training=True,
                    max_train_instances=args.max_train_instances,
                ),
                "val": build_dataset_config(
                    val_images,
                    val_json,
                    compose_transforms(args.resolution, training=False),
                    1,
                    args.num_workers,
                    training=False,
                    max_train_instances=args.max_train_instances,
                ),
            },
            "model": {
                "_target_": (
                    f"{STAGE2_MODEL_HELPER_MODULE}."
                    "build_stage2_aggregate_sam3_image_model"
                ),
                "bpe_path": str(bpe_path),
                "device": "cpu",
                "eval_mode": False,
                "base_checkpoint_path": str(args.base_checkpoint.resolve()),
                "stage1_checkpoint_path": str(
                    args.stage1_checkpoint.resolve()
                ),
                "load_from_HF": False,
                "enable_segmentation": True,
                "enable_inst_interactivity": False,
                "compile": False,
                "freeze_mode": args.freeze_mode,
                "branch_base_channels": args.branch_base_channels,
                "fuse_sam2_features": False,
            },
            "meters": {
                "val": {
                    BEST_VAL_KEY: {
                        BEST_METER_NAME: {
                            "_target_": (
                                f"{BEST_HELPER_MODULE}."
                                "BestSegmentationPredictionDumper"
                            ),
                            "metric_key": BEST_METRICS[args.best_metric][0],
                            "output_key": BEST_METRICS[args.best_metric][1],
                            "iou_type": "segm",
                            "dump_dir": str(output_dir / "validation_predictions"),
                            "merge_predictions": True,
                            "gather_pred_via_filesys": False,
                            "maxdets": args.best_max_dets,
                            "postprocessor": {
                                "_target_": "sam3.eval.postprocessors.PostProcessImage",
                                "max_dets_per_img": args.best_max_dets,
                                "iou_type": "segm",
                                "to_cpu": True,
                                "use_original_ids": True,
                                "use_original_sizes_box": True,
                                "use_original_sizes_mask": True,
                                "convert_mask_to_rle": False,
                                "always_interpolate_masks_on_gpu": True,
                                "use_presence": True,
                                "detection_threshold": -1.0,
                            },
                            "pred_file_evaluators": [
                                {
                                    "_target_": (
                                        "sam3.eval.coco_eval_offline."
                                        "CocoEvaluatorOfflineWithPredFileEvaluators"
                                    ),
                                    "gt_path": str(val_json),
                                    "tide": False,
                                    "iou_type": "segm",
                                }
                            ],
                        }
                    }
                }
            },
            "optim": {
                "amp": {
                    "enabled": args.amp_dtype != "float32",
                    # Trainer在AMP关闭时仍需要合法dtype字符串。
                    "amp_dtype": (
                        args.amp_dtype
                        if args.amp_dtype != "float32"
                        else "bfloat16"
                    ),
                },
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "eps": 1.0e-6,
                    "betas": [0.9, 0.999],
                },
                "gradient_clip": {
                    "_target_": "sam3.train.optim.optimizer.GradientClipper",
                    "max_norm": args.gradient_clip,
                    "norm_type": 2,
                },
                "param_group_modifiers": [],
                "options": {
                    "lr": [{"scheduler": lr_scheduler}],
                    "weight_decay": [
                        {
                            "scheduler": {
                                "_target_": (
                                    "fvcore.common.param_scheduler."
                                    "ConstantParamScheduler"
                                ),
                                "value": args.weight_decay,
                            }
                        },
                        {
                            "scheduler": {
                                "_target_": (
                                    "fvcore.common.param_scheduler."
                                    "ConstantParamScheduler"
                                ),
                                "value": 0.0,
                            },
                            "param_names": ["*bias*"],
                            "module_cls_names": ["torch.nn.LayerNorm"],
                        },
                    ],
                },
            },
            "checkpoint": {
                "save_dir": str(output_dir / "checkpoints"),
                "save_freq": args.save_every,
                "save_list": [],
                "save_best_meters": [f"val_{BEST_VAL_KEY}/{BEST_METER_NAME}"],
                "skip_saving_parameters": [],
            },
            "logging": {
                "tensorboard_writer": {
                    "_target_": "sam3.train.utils.logger.make_tensorboard_logger",
                    "log_dir": str(output_dir / "tensorboard"),
                    "flush_secs": 30,
                    "should_log": True,
                },
                "wandb_writer": None,
                "log_dir": str(output_dir / "logs"),
                "log_freq": 1,
            },
        },
        "launcher": {
            "num_nodes": 1,
            "gpus_per_node": 1,
            "experiment_log_dir": str(output_dir),
            "multiprocessing_context": "spawn",
        },
        "submitit": {
            "account": None,
            "partition": None,
            "qos": None,
            "timeout_hour": 72,
            "use_cluster": False,
            "cpus_per_task": max(2, args.num_workers + 2),
            "port_range": [10000, 65000],
            "constraint": None,
        },
    }

    # AGGREGATE_SAM3_GRAD_ACCUM_CHUNKING_FIX
    # SAM3 Trainer在gradient_accumulation_steps > 1时要求训练批次为
    # 长度等于累积步数的list。普通collate_fn_api返回dict，因此需要
    # 使用collate_fn_api_with_chunking，并让DataLoader一次读取
    # micro_batch_size * grad_accum个样本后再分成多个微批次。
    if args.grad_accum > 1:
        train_data = config["trainer"]["data"]["train"]
        train_data["batch_size"] = args.batch_size * args.grad_accum
        train_data["collate_fn"] = {
            "_target_": (
                "sam3.train.data.collator."
                "collate_fn_api_with_chunking"
            ),
            "_partial_": True,
            "num_chunks": args.grad_accum,
            "repeats": 1,
            "dict_key": "all",
            "with_seg_masks": True,
        }

    return config


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.sam3_root = args.sam3_root.resolve()
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.stage1_checkpoint = args.stage1_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(
            f"官方SAM3基础权重不存在：{args.base_checkpoint}"
        )
    if not args.stage1_checkpoint.is_file():
        raise FileNotFoundError(
            f"第一阶段权重不存在：{args.stage1_checkpoint}"
        )
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("epochs、batch-size和grad-accum必须大于等于1。")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning-rate和gradient-clip必须大于0。")
    if not 1 <= args.max_train_instances <= 200:
        raise ValueError(
            "--max-train-instances必须位于1到200之间。"
            "建议使用180，为无目标查询预留容量。"
        )

    # AGGREGATE_SAM3_STAGE2_STABILITY_MODE
    # Stage-1 has already established a stable solution in FP32.
    # Stage-2 therefore permits BF16 AMP but retains conservative LR/clip guards.
    requested_lr = args.learning_rate
    requested_clip = args.gradient_clip

    if args.amp_dtype == "float16":
        raise ValueError(
            "Stage-2禁止float16：当前密集实例SAM3曾出现NaN/Inf风险。"
            "请使用bfloat16；若BF16不稳定则改用float32。"
        )

    if args.learning_rate > 1.0e-5:
        args.learning_rate = 1.0e-5
    if args.gradient_clip > 0.05:
        args.gradient_clip = 0.05

    if (
        requested_lr != args.learning_rate
        or requested_clip != args.gradient_clip
    ):
        print(
            "[Stage-2稳定保护] 已自动修正："
            f"lr {requested_lr}->{args.learning_rate}；"
            f"gradient_clip {requested_clip}->{args.gradient_clip}。"
        )

    print(
        "[Stage-2设置] "
        f"stage1={args.stage1_checkpoint}；"
        f"lr={args.learning_rate}；"
        f"amp_dtype={args.amp_dtype}；"
        "AdamW eps=1e-6；"
        f"gradient_clip={args.gradient_clip}；"
        f"epochs={args.epochs}。"
    )
    if args.resolution != 1008:
        raise ValueError(
            "当前官方SAM3图像模型必须使用 --resolution 1008。"
            f"当前传入的是 {args.resolution}。"
            "SAM3 ViT使用patch_size=14，1008对应72×72特征网格；"
            "其他分辨率会导致RoPE频率张量尺寸不一致并触发AssertionError。"
        )
    print(
        "[SAM3输入尺寸] resolution=1008；patch_size=14；"
        "ViT特征网格=72×72。"
    )
    if args.val_every < 1:
        raise ValueError("val-every必须大于等于1，否则无法选择最佳权重。")
    if args.best_max_dets < 1:
        raise ValueError("best-max-dets必须大于等于1。")
    if Path(args.best_checkpoint_name).name != args.best_checkpoint_name:
        raise ValueError("best-checkpoint-name只能是文件名，不能包含目录。")
    if not args.best_checkpoint_name.lower().endswith(".pt"):
        raise ValueError("best-checkpoint-name必须以.pt结尾。")

    _, official_train_script, bpe_path = locate_official_sam3(args.sam3_root)

    train_split = resolve_split(args.dataset_root, "train")
    val_split = resolve_split(args.dataset_root, "val")
    if val_split is None:
        val_split = resolve_split(args.dataset_root, "valid")

    if train_split is None:
        raise FileNotFoundError(
            f"无法在{args.dataset_root}中识别train图像目录和COCO标注文件。"
        )
    if val_split is None:
        if args.allow_train_as_val:
            print("[警告] 未检测到val/valid，当前仅为冒烟测试，复用train作为val。")
            val_split = train_split
        else:
            raise FileNotFoundError(
                "未检测到val或valid划分。正式训练必须提供独立验证集；"
                "仅测试流程时可添加--allow-train-as-val。"
            )

    train_stats = validate_coco(
        *train_split, expected_prompt=args.expected_prompt
    )
    val_stats = validate_coco(
        *val_split, expected_prompt=args.expected_prompt
    )

    if train_stats["images"] == 0:
        raise ValueError("训练集COCO文件中images为空，无法训练。")

    if val_stats["images"] == 0:
        if args.allow_train_as_val:
            print(
                "[警告] 验证集COCO文件为空；当前仅进行冒烟测试，"
                "临时复用训练集作为验证集。"
            )
            val_split = train_split
            val_stats = train_stats
        else:
            raise ValueError(
                "验证集COCO文件中images为空。正式训练请提供独立验证集；"
                "仅测试代码时可添加--allow-train-as-val。"
            )

    print(f"[数据集] train={train_stats}")
    print(f"[数据集] val={val_stats}")
    print(
        "[稳定性设置] "
        f"lr={args.learning_rate}, amp_dtype={args.amp_dtype}, "
        f"gradient_clip={args.gradient_clip}, "
        f"max_train_instances={args.max_train_instances}/200"
    )

    # Stage-2必须使用新的输出目录，避免官方Trainer自动恢复或覆盖Stage-1。
    existing_latest = args.output_dir / "checkpoints" / "checkpoint.pt"
    if existing_latest.is_file():
        raise RuntimeError(
            "Stage-2输出目录已经存在checkpoint.pt："
            f"{existing_latest}\n"
            "请换一个全新的--output-dir。"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    helper_dir = write_best_metric_helper(args.output_dir)
    stage2_helper_path = write_stage2_model_helper(args.output_dir)
    print(f"[最佳权重] 验证指标：{args.best_metric}")
    print(
        f"[最佳权重] 辅助模块："
        f"{helper_dir / (BEST_HELPER_MODULE + '.py')}"
    )
    print(f"[Stage-2模型] 辅助模块：{stage2_helper_path}")
    config = build_config(args, train_split, val_split, bpe_path)
    config_path = args.output_dir / "aggregate_sam3_generated.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )
    print(f"[配置] 已生成：{config_path}")

    stage2_info = {
        "stage": 2,
        "strategy": "model-only fine-tuning; optimizer and scheduler reinitialized",
        "official_base_checkpoint": str(args.base_checkpoint),
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "learning_rate": args.learning_rate,
        "amp_dtype": args.amp_dtype,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "num_workers": args.num_workers,
        "max_train_instances": args.max_train_instances,
        "resolution": args.resolution,
        "gradient_clip": args.gradient_clip,
        "best_metric": args.best_metric,
    }
    (args.output_dir / "stage2_training_info.json").write_text(
        json.dumps(stage2_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()

    # Windows 单机单 GPU 兼容设置。
    # 官方 SAM3 的 TensorBoard Logger 会读取分布式环境变量；
    # 非 torchrun 启动时这些变量通常不存在，LOCAL_RANK=None 会触发 int(None) 报错。
    # 使用 setdefault 可在普通单卡训练中提供安全默认值，同时不覆盖外部启动器已设置的值。
    env.setdefault("LOCAL_RANK", "0")
    env.setdefault("RANK", "0")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("LOCAL_WORLD_SIZE", "1")
    print(
        "[单GPU环境] "
        f"LOCAL_RANK={env['LOCAL_RANK']}, "
        f"RANK={env['RANK']}, "
        f"WORLD_SIZE={env['WORLD_SIZE']}, "
        f"LOCAL_WORLD_SIZE={env['LOCAL_WORLD_SIZE']}"
    )

    python_paths = [
        str(helper_dir),
        str(project_root),
        str(args.sam3_root),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
    env["PYTHONUTF8"] = "1"

    # Verify that the intended official package is imported.
    verify_code = (
        # This is a fresh Python process. Pillow must be loaded before
        # PyTorch/TorchVision/SAM3 to avoid PIL._imaging DLL conflicts
        # observed in this Windows environment.
        "from PIL import Image; "
        "import torch; "
        "import sam3; "
        "import aggregate_sam3; "
        f"import {BEST_HELPER_MODULE}; "
        f"import {STAGE2_MODEL_HELPER_MODULE}; "
        "from pathlib import Path; "
        "p=Path(sam3.__file__).resolve(); "
        "print('[Pillow优先导入] OK'); "
        "print('[SAM3导入路径]', p); "
        "print('[Aggregate-SAM3导入]', aggregate_sam3.__file__); "
        "print('[CUDA可用]', torch.cuda.is_available()); "
        "print('[BF16支持]', "
        "torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False); "
        f"expected=Path(r'{str(args.sam3_root)}').resolve(); "
        "assert expected in p.parents, "
        "f'导入了错误的sam3：{p}，期望位于{expected}'"
    )
    subprocess.run(
        [sys.executable, "-c", verify_code],
        check=True,
        cwd=project_root,
        env=env,
    )

    if args.dry_run:
        print("[完成] dry-run验证通过，最佳权重配置已生成，未启动训练。")
        return

    local_launcher = Path(__file__).resolve().parent / "launch_train_no_gloo_stable.py"
    if not local_launcher.is_file():
        raise FileNotFoundError(f"单卡启动脚本不存在：{local_launcher}")
    print(f"[单卡启动器] {local_launcher}")
    command = [
        sys.executable,
        str(local_launcher),
        "--config",
        str(config_path),
    ]
    print("[启动训练]", subprocess.list2cmdline(command))

    # 官方Trainer保存最佳权重时会先写入*.pt.tmp；
    # Windows下必须提前创建其父目录。
    prepare_output_directories(args)

    subprocess.run(command, check=True, cwd=project_root, env=env)
    copy_best_checkpoint(args)


if __name__ == "__main__":
    main()
