"""Explicitly synthetic CPU-only end-to-end validation; never research results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from .provenance import sha256,write_json
from .database import TARGETS


def run_smoke(root,targets=None):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError("Synthetic output already exists")
    root.mkdir(parents=True)
    (root/"SYNTHETIC_ONLY.txt").write_text("Generated fixtures and dummy checkpoint identity, not trained SAM weights or paper results.\n",encoding="utf-8")
    rng=np.random.default_rng(17)
    rows=[];labels=[]
    for group in range(10):
        for index in range(3):
            image_id=f"synthetic-g{group:02}-i{index}"
            mask=np.zeros((64,64),dtype=np.uint16)
            for particle in range(4+group):
                y,x=rng.integers(0,59,size=2);size=int(rng.integers(1,5))
                mask[y:y+size,x:x+size]=particle+1
            image=rng.integers(0,256,size=(64,64,3),dtype=np.uint8)
            Image.fromarray(mask).save(root/f"{image_id}.tif")
            Image.fromarray(image).save(root/f"{image_id}.png")
            rows.append(dict(image_id=image_id,original_image_id=image_id,specimen_id=image_id,
                gradation_id=f"g{group:02}",image_path=f"{image_id}.png",mask_path=f"{image_id}.tif",
                sha256=sha256(root/f"{image_id}.png"),width_mm=150,height_mm=150,
                label_source="synthetic-design",reference_id=f"synthetic-reference-{group}"))
            labels.append(dict(image_id=image_id,**dict(zip(TARGETS,np.arange(1.,7.)*10+group))))
    pd.DataFrame(rows).to_csv(root/"gradation.csv",index=False)
    pd.DataFrame(labels).to_csv(root/"labels.csv",index=False)
    seg=[]
    for i,split in enumerate(["train","val","test"]):
        image_id=f"synthetic-development-{i}"
        Image.fromarray(rng.integers(0,256,(16,16,3),dtype=np.uint8)).save(root/f"{image_id}.png")
        Image.fromarray(np.zeros((16,16),dtype=np.uint16)).save(root/f"{image_id}.tif")
        seg.append(dict(image_id=image_id,original_image_id=image_id,specimen_id=image_id,slab_id=image_id,
            split=split,image_path=f"{image_id}.png",mask_path=f"{image_id}.tif",sha256=sha256(root/f"{image_id}.png"),width_mm=150,height_mm=150))
    pd.DataFrame(seg).to_csv(root/"segmentation.csv",index=False)
    (root/"dummy_checkpoint.bin").write_bytes(b"SYNTHETIC IDENTITY ONLY - NOT A MODEL")
    write_json(root/"frozen.json",dict(synthetic=True,training_dataset="segmentation-development",selection_split="val",
                checkpoint_sha256=sha256(root/"dummy_checkpoint.bin"),segmentation_manifest_sha256=sha256(root/"segmentation.csv")))
    runs=[]
    def command(name,config):
        config["synthetic"]=True
        path=root/(name+".yaml")
        path.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
        argv=[sys.executable,"-m","gradation.cli",name,"--config",str(path)]
        start=time.perf_counter()
        result=subprocess.run(argv,text=True,capture_output=True)
        (root/(name+".log")).write_text(result.stdout+result.stderr,encoding="utf-8")
        runs.append(dict(command=argv,returncode=result.returncode,elapsed_seconds=time.perf_counter()-start))
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    command("prepare-seg",dict(segmentation_manifest="segmentation.csv",gradation_manifest="gradation.csv",output="validated"))
    command("build-features",dict(manifest="gradation.csv",undefined="zero",output="features"))
    command("build-database",dict(segmentation_manifest="segmentation.csv",gradation_manifest="gradation.csv",features="features/features.csv",feature_manifest="features/manifest.json",labels="labels.csv",frozen_model="frozen.json",checkpoint="dummy_checkpoint.bin",output="database"))
    selected=targets or TARGETS
    command("run-logo",dict(input="database/database.csv",output="stacking",settings=dict(targets=",".join(selected),models="DT,SVR",rules="all",trials=4,folds=2,max_base=2,shap_samples=2,permutations=1,pilot_trees=2,seed=17,paper_layout=False)))
    argv=[sys.executable,"-m","stacking_model.predict",str(root/"database/database.csv"),"--run",str(root/"stacking"),"--out",str(root/"predictions.xlsx"),"--targets",",".join(selected)]
    start=time.perf_counter();result=subprocess.run(argv,text=True,capture_output=True)
    (root/"predict.log").write_text(result.stdout+result.stderr,encoding="utf-8")
    runs.append(dict(command=argv,returncode=result.returncode,elapsed_seconds=time.perf_counter()-start))
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    command("compare-databases",dict(databases={"fixture_a":"database/database.csv","fixture_b":"database/database.csv"},output="comparison"))
    command("make-report",dict(run="stacking",output="report"))
    write_json(root/"smoke_results.json",dict(synthetic=True,research_replication=False,n_designs=10,n_images=30,targets=selected,runs=runs))
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    print(run_smoke(args.out))

if __name__=="__main__":main()
