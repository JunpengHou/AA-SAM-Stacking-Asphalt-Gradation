from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# IMPORTANT: initialize Pillow before importing PyTorch in this environment.
from PIL import Image  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    """Convolution + GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        norm_groups = min(32, out_channels)
        while out_channels % norm_groups != 0:
            norm_groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )


class DepthwiseResidualBlock(nn.Module):
    """Lightweight residual block for local aggregate-particle texture and boundary cues."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        norm_groups = min(32, channels)
        while channels % norm_groups != 0:
            norm_groups -= 1

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.pw1 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm1(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.norm2(x)
        x = x * self.layer_scale.view(1, -1, 1, 1)
        return residual + x


def _normalized_sobel_edge(image: torch.Tensor) -> torch.Tensor:
    """Return a normalized one-channel Sobel magnitude map for boundary cues."""

    gray = (
        0.2989 * image[:, 0:1]
        + 0.5870 * image[:, 1:2]
        + 0.1140 * image[:, 2:3]
    )
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = image.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    grad_x = F.conv2d(gray, kernel_x, padding=1)
    grad_y = F.conv2d(gray, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
    scale = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    return magnitude / scale


class AggregateFeatureBranch(nn.Module):
    """
    Aggregate-particle feature branch.

    It produces four feature levels that emphasize:
    1. thin mortar gaps and small aggregate boundaries;
    2. local texture/color differences between aggregate and asphalt mortar;
    3. medium-scale particle shape;
    4. coarse mixture context.

    A normalized Sobel magnitude map is concatenated with RGB before the
    learnable stem, which improves sensitivity to narrow particle boundaries
    without modifying the official SAM3 backbone.
    """

    def __init__(
        self,
        out_channels: int = 256,
        base_channels: int = 32,
        blocks_per_stage: Sequence[int] = (2, 2, 3, 2),
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 4:
            raise ValueError("blocks_per_stage 必须包含4个整数。")

        self.stem = nn.Sequential(
            ConvNormAct(4, base_channels, kernel_size=3, stride=2),
            ConvNormAct(base_channels, base_channels, kernel_size=3, stride=1),
            DepthwiseResidualBlock(base_channels),
        )

        stage_channels = (
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 8,
        )

        in_ch = base_channels
        self.stages = nn.ModuleList()
        self.projections = nn.ModuleList()

        for stage_idx, (out_ch, num_blocks) in enumerate(
            zip(stage_channels, blocks_per_stage)
        ):
            dilation = (1, 1, 2, 3)[stage_idx]
            blocks = [ConvNormAct(in_ch, out_ch, kernel_size=3, stride=2)]
            blocks.extend(
                DepthwiseResidualBlock(out_ch, dilation=dilation)
                for _ in range(num_blocks)
            )
            self.stages.append(nn.Sequential(*blocks))
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(out_ch, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                )
            )
            in_ch = out_ch

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "集料分支输入必须为[B, 3, H, W]，"
                f"当前形状为{tuple(image.shape)}。"
            )

        features: list[torch.Tensor] = []
        edge = _normalized_sobel_edge(image)
        x = self.stem(torch.cat([image, edge], dim=1))
        for stage, projection in zip(self.stages, self.projections):
            x = stage(x)
            features.append(projection(x))
        return features


class GatedResidualFusion(nn.Module):
    """
    Fuse one SAM3 feature level and one aggregate-branch feature level.

    The output starts close to the original SAM3 feature because the final
    residual convolution and the learnable scale are initialized near zero.
    """

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.aggregate_align = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

        # Make the initial model behave almost exactly like pretrained SAM3.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        sam_feature: torch.Tensor,
        aggregate_feature: torch.Tensor,
    ) -> torch.Tensor:
        aggregate_feature = F.interpolate(
            aggregate_feature,
            size=sam_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        aggregate_feature = self.aggregate_align(aggregate_feature)
        merged = torch.cat([sam_feature, aggregate_feature], dim=1)
        gate = self.gate(merged)
        residual = self.residual(merged)
        return sam_feature + self.scale * gate * residual


def _extract_image_tensor(samples: Any) -> torch.Tensor:
    """Extract a BCHW image tensor from the input accepted by SAM3."""

    if torch.is_tensor(samples):
        if samples.ndim == 4:
            return samples
        if samples.ndim == 3:
            return samples.unsqueeze(0)

    tensors = getattr(samples, "tensors", None)
    if torch.is_tensor(tensors):
        return tensors

    if isinstance(samples, (list, tuple)):
        if len(samples) == 1 and torch.is_tensor(samples[0]) and samples[0].ndim == 4:
            return samples[0]
        if samples and all(torch.is_tensor(item) and item.ndim == 3 for item in samples):
            return torch.stack(list(samples), dim=0)

    raise TypeError(
        "无法从SAM3视觉骨干输入中提取图像张量。"
        f"输入类型：{type(samples)!r}"
    )


def _unwrap_feature(feature: Any) -> tuple[torch.Tensor, Any]:
    """Support both plain tensors and SAM3 NestedTensor-like objects."""

    if torch.is_tensor(feature):
        return feature, None

    tensors = getattr(feature, "tensors", None)
    if torch.is_tensor(tensors):
        return tensors, feature

    raise TypeError(f"不支持的SAM3特征类型：{type(feature)!r}")


def _rewrap_feature(tensor: torch.Tensor, template: Any) -> Any:
    if template is None:
        return tensor

    mask = getattr(template, "mask", None)
    try:
        return type(template)(tensor, mask)
    except Exception:
        # Older/newer SAM3 variants may use a mutable container.
        if hasattr(template, "tensors"):
            template.tensors = tensor
            return template
        raise


def _cast_feature_dtype(feature: Any, dtype: torch.dtype) -> Any:
    """Cast floating tensors while preserving SAM3 NestedTensor-like wrappers."""

    if torch.is_tensor(feature):
        return feature.to(dtype=dtype) if feature.is_floating_point() else feature

    tensors = getattr(feature, "tensors", None)
    if torch.is_tensor(tensors):
        cast_tensor = (
            tensors.to(dtype=dtype) if tensors.is_floating_point() else tensors
        )
        return _rewrap_feature(cast_tensor, feature)

    return feature


def _cast_feature_sequence_dtype(
    features: Sequence[Any] | None,
    dtype: torch.dtype,
) -> Sequence[Any] | None:
    if features is None:
        return None
    return [_cast_feature_dtype(feature, dtype) for feature in features]


class AggregateFusionVisionBackbone(nn.Module):
    """
    Wrap the official SAM3 vision neck.

    Forward flow:
        RGB -> official SAM3 ViT/neck -> SAM3 feature pyramid
        RGB -> aggregate feature branch     -> aggregate feature pyramid
        both pyramids -> gated residual fusion
        fused pyramid -> original SAM3 transformer encoder and decoder
    """

    def __init__(
        self,
        base_backbone: nn.Module,
        feature_dim: int = 256,
        branch_base_channels: int = 32,
        fuse_sam2_features: bool = False,
    ) -> None:
        super().__init__()
        self.base_backbone = base_backbone
        self.aggregate_branch = AggregateFeatureBranch(
            out_channels=feature_dim,
            base_channels=branch_base_channels,
        )
        self.fusion_blocks = nn.ModuleList(
            GatedResidualFusion(feature_dim) for _ in range(4)
        )
        self.fuse_sam2_features = fuse_sam2_features

    @property
    def trunk(self) -> Any:
        """Keep compatibility with code that accesses vision_backbone.trunk."""

        return getattr(self.base_backbone, "trunk", None)

    @property
    def position_encoding(self) -> Any:
        return getattr(self.base_backbone, "position_encoding", None)

    def _fuse_feature_list(
        self,
        sam_features: Sequence[Any] | None,
        aggregate_features: Sequence[torch.Tensor],
    ) -> Sequence[Any] | None:
        if sam_features is None:
            return None

        fused: list[Any] = []
        num_levels = len(sam_features)

        for level, sam_feature in enumerate(sam_features):
            sam_tensor, template = _unwrap_feature(sam_feature)
            aggregate_index = min(level, len(aggregate_features) - 1)
            fusion_index = min(level, len(self.fusion_blocks) - 1)
            fused_tensor = self.fusion_blocks[fusion_index](
                sam_tensor,
                aggregate_features[aggregate_index],
            )
            fused.append(_rewrap_feature(fused_tensor, template))

        if num_levels == 0:
            return sam_features
        return fused

    def forward(self, samples: Any):
        # AGGREGATE_SAM3_FROZEN_BASE_LOCAL_BF16_FIX
        #
        # 官方SAM3冻结ViT使用的融合MLP/addmm_act路径会产生BFloat16激活。
        # 当全局AMP关闭、权重保持Float32时，直接前向会出现：
        #   mat1 and mat2 must have the same dtype,
        #   but got BFloat16 and Float
        #
        # 正确策略：
        # 1. 冻结的官方视觉骨干：no_grad + 局部bfloat16 autocast；
        # 2. 骨干输出：立即转换回Float32；
        # 3. 可训练集料分支、融合模块、解码器和损失：继续使用Float32。
        #
        # 这样既兼容官方SAM3视觉骨干，又避免全局bfloat16训练导致Loss=NaN。
        image = _extract_image_tensor(samples)

        base_is_frozen = not any(
            parameter.requires_grad
            for parameter in self.base_backbone.parameters()
        )

        if base_is_frozen:
            self.base_backbone.eval()
            use_cuda_bf16 = image.is_cuda and torch.cuda.is_bf16_supported()
            with torch.no_grad():
                with torch.autocast(
                    device_type=image.device.type,
                    dtype=torch.bfloat16,
                    enabled=use_cuda_bf16,
                ):
                    base_output = self.base_backbone(samples)
        else:
            base_output = self.base_backbone(samples)

        if not isinstance(base_output, (tuple, list)):
            raise TypeError(
                "SAM3视觉骨干返回值应为tuple/list，"
                f"当前类型为{type(base_output)!r}。"
            )

        aggregate_features = self.aggregate_branch(image.float())

        # Official Sam3DualViTDetNeck:
        # (sam3_features, sam3_pos, sam2_features, sam2_pos)
        if len(base_output) == 4:
            sam3_features, sam3_pos, sam2_features, sam2_pos = base_output

            # The trainable fusion/decoder path is intentionally Float32.
            sam3_features = _cast_feature_sequence_dtype(
                sam3_features,
                torch.float32,
            )
            sam3_pos = _cast_feature_sequence_dtype(
                sam3_pos,
                torch.float32,
            )
            sam2_features = _cast_feature_sequence_dtype(
                sam2_features,
                torch.float32,
            )
            sam2_pos = _cast_feature_sequence_dtype(
                sam2_pos,
                torch.float32,
            )

            sam3_features = self._fuse_feature_list(
                sam3_features,
                aggregate_features,
            )
            if self.fuse_sam2_features:
                sam2_features = self._fuse_feature_list(
                    sam2_features,
                    aggregate_features,
                )
            return sam3_features, sam3_pos, sam2_features, sam2_pos

        # Fail explicitly for an incompatible future interface.
        raise RuntimeError(
            "当前代码适配官方Sam3DualViTDetNeck的4项输出接口，"
            f"但检测到{len(base_output)}项输出。"
            "请确认使用的是SAM3图像模型，而非SAM3.1三头视频模型。"
        )


@dataclass(frozen=True)
class TrainableSummary:
    trainable: int
    total: int

    @property
    def ratio(self) -> float:
        return self.trainable / max(self.total, 1)


def _set_freeze_mode(model: nn.Module, freeze_mode: str) -> TrainableSummary:
    """
    freeze_mode:
      - adapter: train aggregate branch/fusion + SAM3 decoder/segmentation output heads.
      - vision_frozen: freeze original SAM3 vision and language backbones.
      - full: train all parameters.
    """

    freeze_mode = freeze_mode.lower()
    valid = {"adapter", "vision_frozen", "full"}
    if freeze_mode not in valid:
        raise ValueError(f"freeze_mode必须属于{sorted(valid)}，当前为{freeze_mode!r}。")

    if freeze_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True

    elif freeze_mode == "vision_frozen":
        for name, parameter in model.named_parameters():
            frozen = (
                "backbone.vision_backbone.base_backbone." in name
                or "backbone.language_backbone." in name
            )
            parameter.requires_grad = not frozen

    else:  # adapter
        trainable_keywords = (
            "aggregate_branch.",
            "fusion_blocks.",
            "transformer.decoder.",
            "segmentation_head.",
            "dot_prod_scoring.",
            "class_embed",
            "bbox_embed",
            "presence",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(key in name for key in trainable_keywords)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("冻结策略导致可训练参数为0，请检查SAM3版本及参数名称。")
    return TrainableSummary(trainable=trainable, total=total)


def _filtered_call(function, **kwargs):
    """Pass only arguments supported by the installed SAM3 builder version."""

    signature = inspect.signature(function)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return function(**accepted)


def build_aggregate_sam3_image_model(
    bpe_path: str | None = None,
    device: str = "cpu",
    eval_mode: bool = False,
    checkpoint_path: str | None = None,
    load_from_HF: bool = False,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
    freeze_mode: str = "vision_frozen",
    branch_base_channels: int = 32,
    fuse_sam2_features: bool = False,
):
    """
    Build an official SAM3 image model, add a aggregate feature branch, and replace
    the official visual neck with a feature-fusion wrapper.

    The original SAM3 transformer encoder, transformer decoder, box head and
    segmentation head are retained. Therefore the fused features are decoded
    by the official SAM3 decoder.
    """

    try:
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        raise RuntimeError(
            "无法导入官方SAM3。请确保官方仓库已完整安装，"
            "且项目中不存在scripts/sam3等同名残缺目录。"
        ) from exc

    model = _filtered_call(
        build_sam3_image_model,
        bpe_path=bpe_path,
        device=device,
        eval_mode=eval_mode,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        enable_inst_interactivity=enable_inst_interactivity,
        compile=compile,
    )

    backbone = getattr(model, "backbone", None)
    if backbone is None or not hasattr(backbone, "vision_backbone"):
        raise AttributeError(
            "当前SAM3模型没有model.backbone.vision_backbone，"
            "请确认使用官方SAM3图像模型。"
        )

    if isinstance(backbone.vision_backbone, AggregateFusionVisionBackbone):
        raise RuntimeError("模型已包含集料颗粒特征分支，禁止重复包装。")

    backbone.vision_backbone = AggregateFusionVisionBackbone(
        base_backbone=backbone.vision_backbone,
        feature_dim=256,
        branch_base_channels=branch_base_channels,
        fuse_sam2_features=fuse_sam2_features,
    )

    summary = _set_freeze_mode(model, freeze_mode)
    print(
        "[Aggregate-SAM3] 模型构建完成；"
        f"冻结模式={freeze_mode}；"
        f"可训练参数={summary.trainable:,}/{summary.total:,}"
        f"（{summary.ratio:.2%}）"
    )

    if eval_mode:
        model.eval()
    return model
