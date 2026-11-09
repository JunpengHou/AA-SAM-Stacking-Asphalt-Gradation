"""Compare SAM/AA-SAM/YOLOv5-Seg/Mask R-CNN candidates on identical GT.
This evaluates provided predictions; it does not invent missing baseline trainers.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args();path=args.config.resolve();config=yaml.safe_load(path.read_text(encoding="utf-8"))
    output=(path.parent/config["output"]).resolve()
    if output.exists():raise FileExistsError(output)
    identities=None;configs=[]
    for model,relative in config["models"].items():
        cpath=(path.parent/relative).resolve();c=yaml.safe_load(cpath.read_text(encoding="utf-8"))
        table=pd.read_csv(cpath.parent/c["samples"])
        if table.image_id.duplicated().any():raise ValueError("Duplicate comparison IDs")
        from hashlib import sha256
        pairs=sorted((str(row.image_id),sha256((cpath.parent/row.truth_path).read_bytes()).hexdigest()) for row in table.itertuples())
        if identities is not None and pairs!=identities:raise ValueError("Comparison GT/IDs are not exactly paired")
        identities=pairs;configs.append((model,cpath,c))
    if not configs:raise ValueError("No models supplied")
    output.mkdir(parents=True);rows=[]
    for index,(model,cpath,c) in enumerate(configs):
        source_samples=pd.read_csv(cpath.parent/c["samples"])
        for column in ["truth_path","candidates_path"]:
            source_samples[column]=source_samples[column].map(lambda value:str((cpath.parent/value).resolve()))
        normalized=output/f"samples_{index:02}.csv"
        source_samples.to_csv(normalized,index=False)
        c["samples"]=str(normalized)
        c["output"]=str(output/f"model_{index:02}")
        generated=output/f"model_{index:02}.yaml"
        generated.write_text(yaml.safe_dump(c),encoding="utf-8")
        subprocess.run([sys.executable,"-m","gradation.cli","eval-seg","--config",str(generated)],check=True)
        metrics=json.loads((Path(c["output"])/"segmentation_metrics.json").read_text())
        rows.append(dict(model=model,**{k:v for k,v in metrics.items() if not isinstance(v,dict)}))
    pd.DataFrame(rows).to_csv(output/"comparison.csv",index=False)

if __name__=="__main__":main()
