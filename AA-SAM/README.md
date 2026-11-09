# Preserved SAM3 segmentation implementation

The model source in AA-SAM/ implements normalized Sobel input, four-level AFEB, depthwise residual blocks, GroupNorm/GELU, layer scale 1e-4 and gated residual fusion initialized as an identity (scale 0.05, zero final convolution). aggregate_sam3/ exposes this source under the namespace used by existing training scripts.

This is the actual existing SAM3 implementation. It is not silently renamed as original SAM. The original-SAM backbone/prompt interface, historical training checkpoint provenance, original-SAM training, YOLOv5-Seg/Mask R-CNN training and executable five-way ablation training remain unverified or absent. Ablation YAML is a protocol description, not evidence of completed experiments.

Training code and validation-only checkpoint sweeps are preserved. Paths are required CLI arguments; use --help. Full training needs an independently installed official SAM3 checkout, compatible weights and private dataset. Do not infer successful GPU training from component unit tests. The upstream SAM License applies where appropriate; see ../LICENSE_SCOPE.md.
