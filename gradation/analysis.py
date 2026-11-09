"""Descriptive eta squared and equal-weight field hierarchy."""
import numpy as np
import pandas as pd
from .database import TARGETS


def eta_squared(frame, names, group="gradation_id"):
    rows=[]
    if frame[group].isna().any():raise ValueError("Missing group")
    for name in names:
        values=pd.to_numeric(frame[name],errors="raise")
        valid=values.notna()
        data=pd.DataFrame({"value":values[valid],"group":frame.loc[valid,group]})
        total=float(((data.value-data.value.mean())**2).sum())
        between=sum(len(part)*(part.value.mean()-data.value.mean())**2 for _,part in data.groupby("group"))
        rows.append(dict(feature=name,eta_squared=float(between/total) if total else np.nan,
                         n_defined=len(data),n_missing=int((~valid).sum())))
    return pd.DataFrame(rows)


def regression_metrics(y,p,min_target=1e-8):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if y.shape!=p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid regression pairs")
    error=y-p
    return dict(MSE=float(np.mean(error**2)),MAE=float(np.mean(np.abs(error))),
                RMSE=float(np.sqrt(np.mean(error**2))),
                MAPE=float(np.mean(np.abs(error/y))*100) if (np.abs(y)>min_target).all() else None,
                MAPE_undefined_count=int((np.abs(y)<=min_target).sum()),
                R2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)) if np.ptp(y)>0 else None)


def field_validation(patches,cores,expected_patches=4,expected_points=4):
    keys=["road_id","cross_section_id","point_id"]
    needed=[*keys,"patch_id",*TARGETS]
    if set(needed)-set(patches) or set(keys+TARGETS)-set(cores):raise ValueError("Missing field columns")
    if patches[needed].isna().any().any() or cores[keys+TARGETS].isna().any().any():raise ValueError("Missing field values")
    if patches.duplicated(keys+["patch_id"]).any() or cores.duplicated(keys).any():raise ValueError("Duplicate field records")
    if not np.isfinite(patches[TARGETS].to_numpy(float)).all() or not np.isfinite(cores[TARGETS].to_numpy(float)).all():raise ValueError("Nonfinite field value")
    if (patches.groupby(keys).size()!=expected_patches).any():raise ValueError("Incomplete patch-to-point hierarchy")
    points=patches.groupby(keys,as_index=False)[TARGETS].mean()
    if (points.groupby(keys[:2]).size()!=expected_points).any():raise ValueError("Incomplete point-to-section hierarchy")
    paired=points.merge(cores,on=keys,how="outer",suffixes=("_pred","_ref"),indicator=True,validate="one_to_one")
    if not paired._merge.eq("both").all():raise ValueError("Missing paired field core")
    sections=paired.drop(columns="_merge").groupby(keys[:2],as_index=False).mean(numeric_only=True)
    rows=[]
    for level,table in [("point",paired),("cross_section",sections)]:
        for target in TARGETS:
            rows.append(dict(level=level,sieve=target,n_pairs=len(table),
                        **regression_metrics(table[target+"_ref"],table[target+"_pred"])))
    return points,sections,pd.DataFrame(rows)


def cross_section_metrics(sections):
    """Manuscript p22: one road-section metric across six sieve pairs."""
    keys=["road_id","cross_section_id"]
    rows=[]
    for record in sections.to_dict(orient="records"):
        y=[record[t+"_ref"] for t in TARGETS]
        p=[record[t+"_pred"] for t in TARGETS]
        rows.append({**{k:record[k] for k in keys},"n_sieves":len(TARGETS),**regression_metrics(y,p)})
    return pd.DataFrame(rows)
