"""Versioned adapter over the preserved 47-descriptor implementation."""
import csv
import importlib.util
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from .provenance import manifest, sha256, write_json

_SOURCE = Path(__file__).resolve().parents[1] / "features" / "extract_47_features.py"
_spec = importlib.util.spec_from_file_location("_legacy_feature47", _SOURCE)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
SCHEMA_VERSION = "47-v2-coarse2.36"
HEADERS = _legacy.HEADERS


def measure(labels, width_mm, height_mm, particles="all", undefined="zero"):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not labels.size or labels.dtype.kind not in "iu" or labels.min() < 0:
        raise ValueError("Labels must be a nonempty 2D nonnegative integer map")
    if particles not in {"all", "interior"} or undefined not in {"nan", "zero"}:
        raise ValueError("Unknown particle/undefined policy")
    if not all(np.isfinite(v) and v > 0 for v in [width_mm, height_mm]):
        raise ValueError("Physical FOV must be finite and positive")
    # The preserved legacy entry remains 4.75 mm. This private module is isolated.
    _legacy.COARSE_MM = 2.36
    values, qc = _legacy.measure(labels, width_mm, height_mm, particles)
    result = dict(zip(HEADERS, values))
    if undefined == "nan":
        denominators = {"Aavg": result["N"], "Ac_avg": result["Nc"],
                        "Af_avg": result["Nf"], "Racf": result["Af"],
                        "Ras": width_mm*height_mm-result["A"], "Aedge": result["Nedge"]}
        for tag in _legacy.TAGS:
            denominators["Rn"+tag] = result["N"]
            denominators["Ra"+tag] = result["A"]
            denominators["A"+tag+"_avg"] = result["N"+tag]
        for name, denominator in denominators.items():
            if denominator == 0:
                result[name] = np.nan
        if result["N"] < 2:
            result["Davg"] = result["Dmin"] = np.nan
    h,w = labels.shape
    rows = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        region = labels == label_id
        ys,xs = np.nonzero(region)
        area = len(xs)*width_mm/w*height_mm/h
        edge = bool(((xs==0)|(xs==w-1)|(ys==0)|(ys==h-1)).any())
        rows.append({"instance_id": int(label_id), "area_px":len(xs), "area_mm2":area,
                     "diameter_mm":np.sqrt(4*area/np.pi), "border":edge,
                     "components_8":ndimage.label(region, np.ones((3,3)))[1],
                     "included": particles == "all" or not edge})
    return result, rows, {"schema_version":SCHEMA_VERSION, "undefined_policy":undefined,
                         "source_instances":qc[4], "included_instances":result["N"],
                         "undefined_count":sum(pd.isna(v) for v in result.values())}


def read_labels(path):
    """Read a label map or a JSON descriptor for disjoint binary instance masks."""
    path=Path(path)
    if path.suffix.lower()!=".json":return _legacy.read_mask(path),[path]
    descriptor=json.loads(path.read_text(encoding="utf-8"))
    shape=descriptor.get("shape")
    if not isinstance(shape,list) or len(shape)!=2 or any(not isinstance(v,int) or v<1 for v in shape):
        raise ValueError("Mask-set descriptor requires positive integer [height,width] shape")
    labels=np.zeros(shape,dtype=np.int32);inputs=[path]
    for label_id,relative in enumerate(descriptor["masks"],1):
        source=path.parent/relative
        with _legacy.Image.open(source) as image:
            if image.n_frames!=1:raise ValueError("Mask-set images must be single-page")
            mask=np.asarray(image)
        if mask.shape!=tuple(shape) or not set(np.unique(mask)) <= {0,1,255}:
            raise ValueError("Expected binary instance masks with the declared shape")
        mask=mask>0
        if not mask.any():raise ValueError("Empty member of mask set; use empty list for zero instances")
        if (mask & (labels>0)).any():raise ValueError("Overlapping ground-truth masks require explicit correction")
        labels[mask]=label_id;inputs.append(source)
    return labels,inputs


def qa_preview(labels):
    """Deterministic instance colors; QA visualization never changes labels."""
    ids,inverse=np.unique(labels,return_inverse=True)
    palette=np.zeros((len(ids),3),dtype=np.uint8)
    for i,label_id in enumerate(ids):
        if label_id:palette[i]=np.random.default_rng(int(label_id)).integers(50,256,3)
    return _legacy.Image.fromarray(palette[inverse].reshape(*labels.shape,3))


def build_features(frame, root, output, undefined="zero", particles="all"):
    """Stream instance detail and previews; mark incomplete runs on any failure."""
    output=Path(output)
    if output.exists():raise FileExistsError("Choose a new features output directory")
    if frame.empty or frame.image_id.duplicated().any():raise ValueError("Nonempty unique image IDs required")
    output.mkdir(parents=True);(output/"qa").mkdir()
    incomplete=output/".incomplete";incomplete.write_text("Feature run is incomplete",encoding="utf-8")
    records,qa,inputs=[],[],[_SOURCE,Path(__file__)]
    fields=["image_id","instance_id","area_px","area_mm2","diameter_mm","border","components_8","included"]
    with (output/"instances.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in frame.itertuples():
            path=Path(root)/row.mask_path
            try:
                labels,source_paths=read_labels(path)
                features,detail,audit=measure(labels,float(row.width_mm),float(row.height_mm),particles,undefined)
            except (OSError,ValueError,KeyError) as error:
                pd.DataFrame([dict(image_id=row.image_id,error=str(error))]).to_csv(output/"errors.csv",index=False)
                raise
            records.append({"image_id":row.image_id,**features})
            for record in detail:writer.writerow({"image_id":row.image_id,**record})
            preview_name=f"{len(records):05d}_"+re.sub(r"[^A-Za-z0-9_.-]","_",str(row.image_id))[:80]+".png"
            qa_preview(labels).save(output/"qa"/preview_name)
            qa.append({"image_id":row.image_id,"mask_sha256":sha256(path),"qa_path":"qa/"+preview_name,**audit})
            inputs.extend(source_paths)
    table=pd.DataFrame(records,columns=["image_id",*HEADERS])
    table.to_csv(output/"features.csv",index=False)
    table.to_excel(output/"features.xlsx",index=False)
    pd.DataFrame(qa).to_csv(output/"qa.csv",index=False)
    provenance=manifest(inputs,{"schema_version":SCHEMA_VERSION,"undefined":undefined,
                              "particles":particles,"coarse_mm":2.36})
    provenance["feature_table_sha256"]=sha256(output/"features.csv")
    provenance["status"]="complete"
    write_json(output/"manifest.json",provenance)
    incomplete.unlink()
    return table
