"""Strict manifests, original-slab splits, and physical crop geometry."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .provenance import sha256

COMMON = ["image_id", "original_image_id", "specimen_id", "image_path",
          "sha256", "width_mm", "height_mm"]
REQUIRED = {
    "segmentation": COMMON + ["slab_id", "split", "mask_path"],
    "gradation": COMMON + ["gradation_id", "mask_path", "label_source", "reference_id"],
    "field": COMMON + ["road_id", "cross_section_id", "point_id", "patch_id", "mask_path"],
}


def validate_manifest(frame, kind, verify_files=False, root=Path(".")):
    missing = set(REQUIRED[kind]) - set(frame)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    df = frame.copy()
    for col in REQUIRED[kind]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing manifest value: {col}")
    if df.empty or df.image_id.duplicated().any():
        raise ValueError("Manifest must be nonempty with unique image_id")
    if not df.sha256.astype(str).str.fullmatch(r"[a-fA-F0-9]{64}").all():
        raise ValueError("sha256 must be a full hexadecimal digest")
    for col in ["width_mm", "height_mm"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] <= 0).any():
            raise ValueError("Physical FOV must be finite and positive")
    if kind == "segmentation":
        if not set(df.split) <= {"train", "val", "test"}:
            raise ValueError("Invalid segmentation split")
        for key in ["slab_id", "original_image_id", "specimen_id", "sha256"]:
            if df.groupby(key).split.nunique().gt(1).any():
                raise ValueError(f"Segmentation split leakage: {key}")
    if kind == "gradation" and df.groupby("specimen_id").gradation_id.nunique().gt(1).any():
        raise ValueError("Specimen belongs to multiple gradations")
    if verify_files:
        for row in df.itertuples():
            path = Path(root) / row.image_path
            if sha256(path).lower() != row.sha256.lower():
                raise ValueError(f"Hash mismatch: {row.image_id}")
            if not (Path(root) / row.mask_path).is_file():
                raise ValueError(f"Missing mask: {row.image_id}")
    return df


def read_manifest(path, kind, verify_files=False):
    path = Path(path)
    return validate_manifest(pd.read_csv(path, dtype=str, keep_default_na=False), kind,
                             verify_files, path.parent)


def assert_independent(segmentation, gradation):
    for key in ["image_id", "original_image_id", "specimen_id", "sha256"]:
        overlap = set(segmentation[key].str.lower()) & set(gradation[key].str.lower())
        if overlap:
            raise ValueError(f"Segmentation/gradation overlap in {key}: {sorted(overlap)[:5]}")


def assign_slab_splits(frame, counts=(28, 6, 6), seed=2026):
    slabs = np.array(sorted(frame.slab_id.unique()))
    if len(slabs) != sum(counts) or min(counts) < 1:
        raise ValueError("Actual slab count must match explicit split counts")
    np.random.default_rng(seed).shuffle(slabs)
    assignment = dict(zip(slabs, np.repeat(["train", "val", "test"], counts)))
    return frame.assign(split=frame.slab_id.map(assignment))


def physical_windows(width_px, height_px, width_mm, height_mm, size_mm=150., step_mm=50.):
    values = [width_px, height_px, width_mm, height_mm, size_mm, step_mm]
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError("Crop geometry must be finite and positive")
    if size_mm > min(width_mm, height_mm):
        raise ValueError("Crop exceeds physical FOV")
    def axis(px, mm):
        starts = np.arange(int(np.floor((mm-size_mm)/step_mm + 1e-9))+1)*step_mm
        coords = np.column_stack([starts, starts+size_mm])*px/mm
        if not np.allclose(coords, np.rint(coords), atol=1e-8, rtol=0):
            raise ValueError("Physical crop does not align to integer pixels")
        return np.rint(coords).astype(int)
    return [(x0,y0,x1,y1) for y0,y1 in axis(height_px,height_mm)
            for x0,x1 in axis(width_px,width_mm)]


def resize_labels(labels, size):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or labels.min() < 0:
        raise ValueError("Expected nonnegative integer instance map")
    if labels.max() > np.iinfo(np.int32).max:
        raise ValueError("Label ID exceeds supported int32")
    out = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(size, Image.Resampling.NEAREST))
    if not set(np.unique(out)) <= set(np.unique(labels)):
        raise AssertionError("Nearest-neighbor transform corrupted label IDs")
    return out
