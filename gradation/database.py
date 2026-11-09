"""One-to-one labelled database construction without sample dropping."""
import numpy as np
import pandas as pd
from .feature47 import HEADERS
TARGETS = ["P0.075","P2.36","P4.75","P9.5","P13.2","P16"]
IDS = ["image_id","specimen_id","gradation_id"]


def align_tables(tables):
    reference = None
    aligned = []
    for table in tables:
        if table.image_id.isna().any() or table.image_id.duplicated().any():
            raise ValueError("Duplicate/missing image_id")
        table = table.sort_values("image_id").reset_index(drop=True)
        if reference is not None:
            if set(table.image_id) != set(reference.image_id):
                raise ValueError("Unpaired database image IDs; silent drops are forbidden")
            for col in ["specimen_id","gradation_id",*TARGETS,"feature_schema_version","feature_undefined_policy","feature_particles_policy"]:
                if col.startswith("feature_") and col not in table and col not in reference:continue
                if col not in table or col not in reference or not table[col].equals(reference[col]):
                    raise ValueError(f"Unpaired database metadata/labels: {col}")
        reference = table
        aligned.append(table)
    return aligned


def build_database(features, metadata, labels):
    for name,frame in [("features",features),("metadata",metadata),("labels",labels)]:
        if frame.image_id.isna().any() or frame.image_id.duplicated().any():
            raise ValueError(f"{name}: unique image_id required")
    if not set(features.image_id) == set(metadata.image_id) == set(labels.image_id):
        raise ValueError("Input image IDs differ; no inner join or silent sample deletion")
    required = [*IDS,"label_source","reference_id"]
    if set(required)-set(metadata) or set(TARGETS)-set(labels) or set(HEADERS)-set(features):
        raise ValueError("Database columns missing")
    out = metadata[required].merge(features[["image_id",*HEADERS]],on="image_id",validate="one_to_one")
    out = out.merge(labels[["image_id",*TARGETS]],on="image_id",validate="one_to_one")
    if out[required].isna().any().any() or out[required].astype(str).apply(lambda c:c.str.strip().eq("")).any().any():
        raise ValueError("Missing provenance fields")
    numeric = out[[*HEADERS,*TARGETS]].apply(pd.to_numeric,errors="raise")
    if np.isinf(numeric).any().any() or not np.isfinite(numeric[TARGETS]).all().all():
        raise ValueError("Nonfinite database values")
    y = numeric[TARGETS].to_numpy()
    if (y<0).any() or (y>100).any() or (np.diff(y,axis=1)<0).any():
        raise ValueError("References must be nondecreasing percentages in [0,100]")
    if out.groupby("specimen_id").gradation_id.nunique().gt(1).any():
        raise ValueError("Specimen spans gradations")
    if out.groupby("gradation_id")[TARGETS].nunique().gt(1).any().any():
        raise ValueError("Design-level evaluation requires constant references within each design")
    out[[*HEADERS,*TARGETS]] = numeric
    return out.sort_values(IDS).reset_index(drop=True)
