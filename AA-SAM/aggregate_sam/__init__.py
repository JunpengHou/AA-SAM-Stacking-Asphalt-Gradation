"""Compatibility namespace for the preserved SAM3 experiment, not original SAM.
Model source remains in AA-SAM/AA-SAM; no weights are migrated or rewritten.
"""
from pathlib import Path
__path__.append(str(Path(__file__).resolve().parent.parent / "AA-SAM"))
from .model import AggregateFeatureBranch, AggregateFusionVisionBackbone, build_aggregate_sam3_image_model
__all__ = ["AggregateFeatureBranch", "AggregateFusionVisionBackbone", "build_aggregate_sam3_image_model"]
