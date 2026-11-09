from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import types
from pathlib import Path
from typing import Any, Mapping

# Windows环境中保持Pillow优先于PyTorch导入。
from PIL import Image  # noqa: F401

import torch
import sam3.train.trainer as sam3_trainer_module
from sam3.train.trainer import Trainer
from sam3.train.utils.train_utils import get_amp_type


class SingleGPUTrainer(Trainer):
    """
    SAM3沥青集料训练的Windows单GPU兼容版本。

    该类保留官方SAM3的数据、模型、损失、优化器、AMP、日志和检查点流程，
    但在world_size=1时：
      1. 不初始化torch.distributed/Gloo；
      2. 不使用DistributedDataParallel；
      3. 为梯度累积提供no_sync空上下文；
      4. 将Trainer内部直接调用的dist.barrier临时改为空操作。

    仅适用于单机单GPU。不得用于多GPU或多节点训练。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._original_dist_barrier = sam3_trainer_module.dist.barrier
        original_initialized_check = (
            sam3_trainer_module.is_dist_avail_and_initialized
        )

        # 官方Trainer在__init__中强制断言进程组已初始化。
        # 单GPU兼容模式不创建进程组，因此仅在构造阶段绕过这一项断言。
        sam3_trainer_module.is_dist_avail_and_initialized = lambda: True

        # 官方val_epoch中存在直接的dist.barrier()调用。
        # 单进程下将其暂时改为空操作。
        sam3_trainer_module.dist.barrier = lambda *args, **kwargs: None

        try:
            super().__init__(*args, **kwargs)
        except Exception:
            sam3_trainer_module.dist.barrier = self._original_dist_barrier
            raise
        finally:
            sam3_trainer_module.is_dist_avail_and_initialized = (
                original_initialized_check
            )

        logging.info(
            "Windows single-GPU mode enabled: "
            "torch.distributed and DDP are disabled."
        )
        print(
            "[单GPU兼容模式] 已禁用Gloo和DDP；"
            "保留SAM3官方训练、损失、AMP、日志与检查点流程。"
        )

        # NaN批次防护：默认最多跳过10个异常训练迭代。
        # 若超过上限，说明不是单个坏样本，而是系统性数值问题，将停止训练。
        self.nan_batch_count = 0
        self.max_nan_batches = int(
            os.environ.get("AGGREGATE_SAM3_MAX_NAN_BATCHES", "10")
        )
        self.sanitized_gradient_event_count = 0
        self.max_sanitized_gradient_events = int(
            os.environ.get(
                "AGGREGATE_SAM3_MAX_SANITIZED_GRADIENT_EVENTS",
                "100",
            )
        )
        self.nan_diagnostic_dir = (
            Path(self.logging_conf.log_dir).resolve() / "nan_diagnostics"
        )
        self.nan_diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.nan_jsonl_path = self.nan_diagnostic_dir / "nan_batches.jsonl"

        print(
            "[NaN批次防护V2] 已启用；"
            f"诊断目录={self.nan_diagnostic_dir}；"
            f"最多整轮跳过={self.max_nan_batches}次；"
            f"集料分支梯度修复上限={self.max_sanitized_gradient_events}次。"
        )

    def _setup_device(self, accelerator: str) -> None:
        """Force a deterministic single-device setup."""

        self.local_rank = 0
        self.distributed_rank = 0

        if accelerator == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "配置要求使用CUDA，但当前PyTorch未检测到可用GPU。"
                )
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(0)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"不支持的accelerator：{accelerator}")

    def _setup_torch_dist_and_backend(
        self,
        cuda_conf,
        distributed_conf,
    ) -> None:
        """
        Configure CUDA numerical behavior without creating a process group.
        """

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = (
                cuda_conf.cudnn_deterministic
            )
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark

            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = 0

    def _setup_ddp_distributed_training(
        self,
        distributed_conf,
        accelerator: str,
    ) -> None:
        """
        Keep the model as a normal nn.Module instead of wrapping it in DDP.

        The official Trainer calls model.no_sync() during gradient
        accumulation, so a compatible empty context manager is attached.
        """

        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(
                f"模型必须是torch.nn.Module，当前为{type(self.model)!r}。"
            )

        if not hasattr(self.model, "no_sync"):
            def _no_sync(_model):
                return contextlib.nullcontext()

            self.model.no_sync = types.MethodType(
                _no_sync,
                self.model,
            )


    @staticmethod
    def _tensor_summary(value: Any) -> dict[str, Any]:
        """Return compact finite-value statistics without retaining GPU tensors."""

        if value is None:
            return {"present": False}

        if not torch.is_tensor(value):
            return {
                "present": True,
                "type": type(value).__name__,
                "repr": repr(value)[:500],
            }

        tensor = value.detach()
        result: dict[str, Any] = {
            "present": True,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
        }

        if tensor.numel() == 0:
            result["finite"] = True
            return result

        if tensor.is_floating_point() or tensor.is_complex():
            finite = torch.isfinite(tensor)
            result["finite"] = bool(finite.all().item())
            result["nonfinite_count"] = int((~finite).sum().item())
            if finite.any():
                finite_values = tensor[finite].float()
                result.update(
                    {
                        "min": float(finite_values.min().item()),
                        "max": float(finite_values.max().item()),
                        "mean": float(finite_values.mean().item()),
                    }
                )
        else:
            result["finite"] = True
            result["min"] = int(tensor.min().item())
            result["max"] = int(tensor.max().item())

        return result

    @staticmethod
    def _tensor_to_list(value: Any, max_items: int = 100) -> Any:
        if value is None:
            return None
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            items = flat[:max_items].tolist()
            if flat.numel() > max_items:
                return {
                    "values": items,
                    "truncated": True,
                    "total": int(flat.numel()),
                }
            return items
        if isinstance(value, (list, tuple)):
            return list(value)[:max_items]
        return value

    def _batch_summary(
        self,
        batch_dict: Mapping[str, Any],
        *,
        phase: str,
        chunk_index: int,
        loss_value: float | None = None,
        extra_losses: Mapping[str, Any] | None = None,
        reason: str = "nonfinite_loss",
    ) -> dict[str, Any]:
        """Collect image IDs, target geometry and mask statistics for one chunk."""

        if not isinstance(batch_dict, Mapping) or len(batch_dict) != 1:
            return {
                "phase": phase,
                "epoch": float(self.epoch),
                "step": int(self.steps.get(phase, -1)),
                "chunk_index": int(chunk_index),
                "reason": reason,
                "batch_type": type(batch_dict).__name__,
                "batch_repr": repr(batch_dict)[:1000],
            }

        key, datapoint = next(iter(batch_dict.items()))
        report: dict[str, Any] = {
            "phase": phase,
            "epoch": float(self.epoch),
            "step": int(self.steps.get(phase, -1)),
            "chunk_index": int(chunk_index),
            "dict_key": str(key),
            "reason": reason,
            "loss": loss_value,
            "find_text_batch": list(
                getattr(datapoint, "find_text_batch", []) or []
            ),
            "image_batch": self._tensor_summary(
                getattr(datapoint, "img_batch", None)
            ),
            "metadata": [],
            "targets": [],
        }

        for stage_id, metadata in enumerate(
            getattr(datapoint, "find_metadatas", []) or []
        ):
            report["metadata"].append(
                {
                    "stage_id": stage_id,
                    "coco_image_id": self._tensor_to_list(
                        getattr(metadata, "coco_image_id", None)
                    ),
                    "original_image_id": self._tensor_to_list(
                        getattr(metadata, "original_image_id", None)
                    ),
                    "original_category_id": self._tensor_to_list(
                        getattr(metadata, "original_category_id", None)
                    ),
                    "original_size": self._tensor_to_list(
                        getattr(metadata, "original_size", None)
                    ),
                    "object_id": self._tensor_to_list(
                        getattr(metadata, "object_id", None)
                    ),
                }
            )

        for stage_id, target in enumerate(
            getattr(datapoint, "find_targets", []) or []
        ):
            boxes = getattr(target, "boxes", None)
            target_report: dict[str, Any] = {
                "stage_id": stage_id,
                "num_boxes": self._tensor_to_list(
                    getattr(target, "num_boxes", None)
                ),
                "boxes": self._tensor_summary(boxes),
                "boxes_padded": self._tensor_summary(
                    getattr(target, "boxes_padded", None)
                ),
                "segments": self._tensor_summary(
                    getattr(target, "segments", None)
                ),
                "is_valid_segment": self._tensor_to_list(
                    getattr(target, "is_valid_segment", None)
                ),
            }

            if torch.is_tensor(boxes) and boxes.numel() > 0:
                boxes_2d = boxes.detach().float().reshape(-1, 4)
                widths = boxes_2d[:, 2]
                heights = boxes_2d[:, 3]
                target_report["nonpositive_width_count"] = int(
                    (widths <= 0).sum().item()
                )
                target_report["nonpositive_height_count"] = int(
                    (heights <= 0).sum().item()
                )
                target_report["nonfinite_box_count"] = int(
                    (~torch.isfinite(boxes_2d).all(dim=1)).sum().item()
                )
                target_report["min_width"] = float(widths.min().item())
                target_report["min_height"] = float(heights.min().item())

            segments = getattr(target, "segments", None)
            if torch.is_tensor(segments) and segments.numel() > 0:
                masks = segments.detach().bool().reshape(
                    segments.shape[0], -1
                )
                mask_areas = masks.sum(dim=1)
                target_report["empty_mask_count"] = int(
                    (mask_areas == 0).sum().item()
                )
                target_report["min_mask_pixels"] = int(
                    mask_areas.min().item()
                )
                target_report["max_mask_pixels"] = int(
                    mask_areas.max().item()
                )

            report["targets"].append(target_report)

        if extra_losses is not None:
            report["loss_components"] = {}
            for loss_name, value in extra_losses.items():
                if torch.is_tensor(value):
                    scalar = (
                        float(value.detach().float().mean().item())
                        if value.numel() > 0
                        else 0.0
                    )
                    report["loss_components"][str(loss_name)] = {
                        "value": scalar,
                        "finite": math.isfinite(scalar),
                        "tensor": self._tensor_summary(value),
                    }
                else:
                    try:
                        scalar = float(value)
                    except (TypeError, ValueError):
                        scalar = None
                    report["loss_components"][str(loss_name)] = {
                        "value": scalar,
                        "finite": (
                            math.isfinite(scalar)
                            if scalar is not None
                            else None
                        ),
                    }

        return report

    def _save_augmented_images(
        self,
        batch_dict: Mapping[str, Any],
        *,
        chunk_index: int,
    ) -> list[str]:
        """Save normalized augmented tensors as PNG for visual inspection."""

        try:
            from PIL import Image as PILImage
            import numpy as np
        except Exception as exc:
            logging.warning("Unable to import image diagnostic dependencies: %s", exc)
            return []

        if not isinstance(batch_dict, Mapping) or len(batch_dict) != 1:
            return []

        _, datapoint = next(iter(batch_dict.items()))
        images = getattr(datapoint, "img_batch", None)
        if not torch.is_tensor(images) or images.ndim != 4:
            return []

        saved: list[str] = []
        for image_index, tensor in enumerate(images.detach().cpu()[:4]):
            image = tensor.float()
            # Current training normalization is mean=std=0.5.
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = (
                image.permute(1, 2, 0).numpy() * 255.0
            ).round().astype(np.uint8)

            path = self.nan_diagnostic_dir / (
                f"epoch_{int(self.epoch):04d}_"
                f"step_{int(self.steps.get('train', -1)):08d}_"
                f"chunk_{chunk_index}_img_{image_index}.png"
            )
            PILImage.fromarray(image).save(path)
            saved.append(str(path))

        return saved

    def _write_nan_report(
        self,
        report: dict[str, Any],
        batch_dict: Mapping[str, Any],
        *,
        chunk_index: int,
    ) -> None:
        report["saved_augmented_images"] = self._save_augmented_images(
            batch_dict,
            chunk_index=chunk_index,
        )
        with self.nan_jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(report, ensure_ascii=False, allow_nan=True) + "\n"
            )

    def _register_bad_iteration(
        self,
        report: dict[str, Any],
        batch_dict: Mapping[str, Any],
        *,
        chunk_index: int,
    ) -> None:
        self.nan_batch_count += 1
        self._write_nan_report(
            report,
            batch_dict,
            chunk_index=chunk_index,
        )

        image_ids = [
            item.get("original_image_id")
            for item in report.get("metadata", [])
        ]
        logging.error(
            "[NaN批次防护] 已跳过异常迭代；epoch=%s step=%s chunk=%s "
            "original_image_id=%s；诊断文件=%s；累计=%s/%s",
            report.get("epoch"),
            report.get("step"),
            chunk_index,
            image_ids,
            self.nan_jsonl_path,
            self.nan_batch_count,
            self.max_nan_batches,
        )
        print(
            "[NaN批次防护] 已跳过异常迭代；"
            f"original_image_id={image_ids}；"
            f"详情={self.nan_jsonl_path}"
        )

        if self.nan_batch_count > self.max_nan_batches:
            raise FloatingPointError(
                "异常批次数量超过安全上限："
                f"{self.nan_batch_count}>{self.max_nan_batches}。"
                f"请检查 {self.nan_jsonl_path} 和COCO标注。"
            )

    def _nonfinite_gradient_info(self) -> list[dict[str, Any]]:
        """Collect every parameter containing NaN/Inf gradient values."""

        bad: list[dict[str, Any]] = []
        for name, parameter in self.model.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue

            finite = torch.isfinite(grad)
            if bool(finite.all().item()):
                continue

            nonfinite_count = int((~finite).sum().item())
            finite_count = int(finite.sum().item())
            item: dict[str, Any] = {
                "name": name,
                "shape": list(grad.shape),
                "dtype": str(grad.dtype),
                "numel": int(grad.numel()),
                "nonfinite_count": nonfinite_count,
                "finite_count": finite_count,
            }

            if finite_count > 0:
                finite_values = grad.detach()[finite].float()
                item.update(
                    {
                        "finite_min": float(finite_values.min().item()),
                        "finite_max": float(finite_values.max().item()),
                        "finite_abs_max": float(
                            finite_values.abs().max().item()
                        ),
                    }
                )
            bad.append(item)

        return bad

    @staticmethod
    def _is_aggregate_branch_parameter(name: str) -> bool:
        return name.startswith(
            "backbone.vision_backbone.aggregate_branch."
        )

    def _sanitize_aggregate_branch_gradients(
        self,
        bad_gradient_info: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Repair only known custom aggregate-branch gradients.

        NaN/Inf elements are set to zero. Finite elements are retained and
        softly bounded before the official global norm clip is applied.
        """

        info_by_name = {
            str(item["name"]): item for item in bad_gradient_info
        }
        sanitized_parameter_count = 0
        sanitized_element_count = 0

        for name, parameter in self.model.named_parameters():
            item = info_by_name.get(name)
            if item is None:
                continue
            grad = parameter.grad
            if grad is None:
                continue

            nonfinite_mask = ~torch.isfinite(grad)
            count = int(nonfinite_mask.sum().item())
            if count <= 0:
                continue

            # Preserve all valid gradient entries. Only invalid entries are
            # removed, then a conservative element-wise bound is applied.
            grad.nan_to_num_(
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            grad.clamp_(min=-1.0, max=1.0)

            sanitized_parameter_count += 1
            sanitized_element_count += count

        return {
            "sanitized_parameter_count": sanitized_parameter_count,
            "sanitized_element_count": sanitized_element_count,
        }

    def _register_sanitized_gradient_event(
        self,
        report: dict[str, Any],
        batch_dict: Mapping[str, Any],
        *,
        chunk_index: int,
    ) -> None:
        self.sanitized_gradient_event_count += 1
        report["sanitized_gradient_event_count"] = (
            self.sanitized_gradient_event_count
        )
        self._write_nan_report(
            report,
            batch_dict,
            chunk_index=chunk_index,
        )

        logging.warning(
            "[集料分支梯度修复] epoch=%s step=%s；"
            "异常参数=%s；异常元素=%s；累计=%s/%s；详情=%s",
            report.get("epoch"),
            report.get("step"),
            report.get("sanitized_parameter_count"),
            report.get("sanitized_element_count"),
            self.sanitized_gradient_event_count,
            self.max_sanitized_gradient_events,
            self.nan_jsonl_path,
        )
        print(
            "[集料分支梯度修复] 已将aggregate_branch中的NaN/Inf梯度"
            "元素置零，保留本轮其他有效梯度；"
            f"累计={self.sanitized_gradient_event_count}/"
            f"{self.max_sanitized_gradient_events}。"
        )

        if (
            self.sanitized_gradient_event_count
            > self.max_sanitized_gradient_events
        ):
            raise FloatingPointError(
                "集料分支非有限梯度修复次数超过安全上限："
                f"{self.sanitized_gradient_event_count}>"
                f"{self.max_sanitized_gradient_events}。"
                f"请检查 {self.nan_jsonl_path}。"
            )

    def _run_step(
        self,
        batch,
        phase,
        loss_mts,
        extra_loss_mts,
        raise_on_error: bool = True,
    ):
        """
        Official SAM3 forward/backward with bad-batch isolation.

        A non-finite loss causes the entire accumulated iteration to be
        discarded. Gradients are cleared, exact image IDs and target geometry
        are written to nan_diagnostics/nan_batches.jsonl, and training proceeds
        unless the configured safety limit is exceeded.
        """

        self.optim.zero_grad(set_to_none=True)

        if self.gradient_accumulation_steps > 1:
            if not isinstance(batch, list):
                raise TypeError(
                    f"Expected a list of batches, got {type(batch)!r}"
                )
            if len(batch) != self.gradient_accumulation_steps:
                raise ValueError(
                    "Gradient accumulation batch count mismatch: "
                    f"expected={self.gradient_accumulation_steps}, "
                    f"actual={len(batch)}"
                )
            accum_steps = len(batch)
        else:
            accum_steps = 1
            batch = [batch]

        diagnostic_batches: list[Mapping[str, Any]] = []

        for chunk_index, chunked_batch in enumerate(batch):
            diagnostic_batches.append(chunked_batch)
            ddp_context = (
                self.model.no_sync()
                if chunk_index < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                # _step() calls popitem(), so pass a shallow copy and retain the
                # original dictionary for diagnostics.
                step_batch = (
                    dict(chunked_batch)
                    if isinstance(chunked_batch, Mapping)
                    else chunked_batch
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    loss_dict, batch_size, extra_losses = self._step(
                        step_batch,
                        self.model,
                        phase,
                    )

                if len(loss_dict) != 1:
                    raise RuntimeError(
                        f"Expected one core loss, got keys={list(loss_dict)}"
                    )
                loss_key, loss = loss_dict.popitem()
                loss_value = float(loss.detach().float().item())

                extra_nonfinite = False
                for extra_value in extra_losses.values():
                    if torch.is_tensor(extra_value):
                        if not torch.isfinite(extra_value).all():
                            extra_nonfinite = True
                            break
                    else:
                        try:
                            if not math.isfinite(float(extra_value)):
                                extra_nonfinite = True
                                break
                        except (TypeError, ValueError):
                            continue

                if not math.isfinite(loss_value) or extra_nonfinite:
                    report = self._batch_summary(
                        chunked_batch,
                        phase=phase,
                        chunk_index=chunk_index,
                        loss_value=loss_value,
                        extra_losses=extra_losses,
                        reason=(
                            "nonfinite_core_loss"
                            if not math.isfinite(loss_value)
                            else "nonfinite_loss_component"
                        ),
                    )
                    self.optim.zero_grad(set_to_none=True)
                    self._register_bad_iteration(
                        report,
                        chunked_batch,
                        chunk_index=chunk_index,
                    )
                    return

                self.scaler.scale(loss).backward()
                loss_mts[loss_key].update(loss_value, batch_size)

                for extra_loss_key, extra_loss in extra_losses.items():
                    if extra_loss_key not in extra_loss_mts:
                        # Reuse the official AverageMeter class through the
                        # current meter object's type to avoid version coupling.
                        meter_cls = type(next(iter(loss_mts.values())))
                        extra_loss_mts[extra_loss_key] = meter_cls(
                            extra_loss_key,
                            self.device,
                            ":.2e",
                        )
                    extra_loss_mts[extra_loss_key].update(
                        float(extra_loss.detach().float().item()),
                        batch_size,
                    )

        bad_gradient_info = self._nonfinite_gradient_info()
        if bad_gradient_info:
            batch_for_report = diagnostic_batches[-1]
            bad_names = [
                str(item["name"]) for item in bad_gradient_info
            ]
            outside_aggregate_branch = [
                name
                for name in bad_names
                if not self._is_aggregate_branch_parameter(name)
            ]

            report = self._batch_summary(
                batch_for_report,
                phase=phase,
                chunk_index=len(diagnostic_batches) - 1,
                reason=(
                    "sanitized_aggregate_branch_gradient"
                    if not outside_aggregate_branch
                    else "nonfinite_gradient_outside_aggregate_branch"
                ),
            )
            report["nonfinite_gradient_parameters"] = bad_names
            report["nonfinite_gradient_details"] = bad_gradient_info
            report["outside_aggregate_branch_parameters"] = (
                outside_aggregate_branch
            )

            if not outside_aggregate_branch:
                sanitize_result = (
                    self._sanitize_aggregate_branch_gradients(
                        bad_gradient_info
                    )
                )
                report.update(sanitize_result)

                # Verify that every invalid value was actually repaired.
                remaining = self._nonfinite_gradient_info()
                if remaining:
                    report["remaining_nonfinite_gradients"] = remaining
                    self.optim.zero_grad(set_to_none=True)
                    self._register_bad_iteration(
                        report,
                        batch_for_report,
                        chunk_index=len(diagnostic_batches) - 1,
                    )
                    return

                self._register_sanitized_gradient_event(
                    report,
                    batch_for_report,
                    chunk_index=len(diagnostic_batches) - 1,
                )
                # Do not return: official Trainer will now apply global
                # gradient clipping and optimizer.step() to the repaired
                # gradients and all other valid gradients.
            else:
                # Unknown/systemic failure: discard the entire iteration.
                self.optim.zero_grad(set_to_none=True)
                self._register_bad_iteration(
                    report,
                    batch_for_report,
                    chunk_index=len(diagnostic_batches) - 1,
                )
                return

    def run(self):
        """
        Run training/validation and restore the original distributed barrier.
        """

        try:
            return super().run()
        finally:
            sam3_trainer_module.dist.barrier = (
                self._original_dist_barrier
            )
