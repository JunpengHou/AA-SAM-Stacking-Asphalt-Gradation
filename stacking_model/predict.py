"""Predict with final sieve-specific models on image-level feature records."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
if __package__:
    from .features import transform
    from .train import TARGETS
else:
    from features import transform
    from train import TARGETS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path);ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--targets",default=",".join(TARGETS))
    args=ap.parse_args()
    df=pd.read_excel(args.input) if args.input.suffix.lower()==".xlsx" else pd.read_csv(args.input)
    if "image_id" not in df or df.image_id.isna().any() or df.image_id.duplicated().any():
        ap.error("A unique image_id is required for each image.")
    out=df[["image_id"]].copy()
    if args.out.exists():ap.error("Output exists; choose a new file")
    targets=args.targets.split(",")
    if not targets or set(targets)-set(TARGETS) or len(set(targets)) != len(targets):ap.error("Invalid targets")
    for target in targets:
        path=args.run/"final"/target/"model.joblib"
        if not path.exists():ap.error(f"Missing requested final model: {target}")
        bundle=joblib.load(path)
        schema=bundle.get("feature_schema_version","unversioned-legacy")
        if schema!="unversioned-legacy":
            if "feature_schema_version" not in df or not df.feature_schema_version.eq(schema).all():
                ap.error(f"Feature schema mismatch for {target}: expected {schema}")
        for field,expected in bundle.get("feature_policy",{}).items():
            if field not in df or not df[field].eq(expected).all():ap.error(f"Feature policy mismatch: {field}")
        x=df[bundle["features"]].apply(pd.to_numeric,errors="raise").to_numpy(float)
        if np.isinf(x).any():raise ValueError("Infinite input")
        z=transform(x,bundle["median"],bundle["columns"])
        base=np.column_stack([m.predict(z) for m in bundle["base"].values()])
        prediction=bundle["meta"].predict(base)
        if not np.isfinite(prediction).all():raise ValueError("Nonfinite prediction")
        out[target]=prediction
    if len(out.columns)==1:ap.error("No final models found.")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_excel(args.out,index=False)
if __name__=="__main__":main()
