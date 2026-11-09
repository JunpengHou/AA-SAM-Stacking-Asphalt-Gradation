"""Config-driven entry points; all relative paths resolve from the config file."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml
from .provenance import manifest, write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare-seg","build-features","build-database","run-logo","compare-databases","validate-field","eval-seg","make-report"])
    parser.add_argument("--config",required=True,type=Path)
    args=parser.parse_args(argv)
    config_path=args.config.resolve()
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config,dict):parser.error("Config must be a mapping")
    input_paths=[config_path]
    def path(key):
        value=(config_path.parent/str(config[key])).resolve()
        if value.is_file():input_paths.append(value)
        return value
    def table(key):
        p=path(key)
        return pd.read_excel(p) if p.suffix==".xlsx" else pd.read_csv(p)
    output=path("output")
    if output.exists():raise FileExistsError(f"Choose a new output directory: {output}")
    started=time.perf_counter()
    command=args.command
    if command=="prepare-seg":
        from .data import read_manifest,assert_independent
        seg=read_manifest(path("segmentation_manifest"),"segmentation",config.get("verify_files",True))
        grad=read_manifest(path("gradation_manifest"),"gradation",config.get("verify_files",True))
        assert_independent(seg,grad)
        output.mkdir(parents=True)
        seg.to_csv(output/"segmentation.csv",index=False);grad.to_csv(output/"gradation.csv",index=False)
    elif command=="build-features":
        from .data import read_manifest
        from .feature47 import build_features
        frame=read_manifest(path("manifest"),config.get("kind","gradation"),config.get("verify_files",True))
        build_features(frame,path("manifest").parent,output,config.get("undefined","zero"),config.get("particles","all"))
    elif command=="build-database":
        from .data import read_manifest,assert_independent
        from .database import build_database
        from .provenance import sha256
        seg=read_manifest(path("segmentation_manifest"),"segmentation",config.get("verify_files",True))
        grad=read_manifest(path("gradation_manifest"),"gradation",config.get("verify_files",True))
        assert_independent(seg,grad)
        frozen=json.loads(path("frozen_model").read_text(encoding="utf-8"))
        if frozen.get("training_dataset")!="segmentation-development" or frozen.get("selection_split")!="val":
            raise ValueError("Frozen model must declare segmentation-only training and validation-only selection")
        if frozen.get("checkpoint_sha256") != sha256(path("checkpoint")):
            raise ValueError("Frozen checkpoint hash mismatch")
        from .feature47 import SCHEMA_VERSION
        feature_provenance=json.loads(path("feature_manifest").read_text(encoding="utf-8"))
        if (feature_provenance.get("status")!="complete"
                or feature_provenance.get("settings",{}).get("schema_version")!=SCHEMA_VERSION
                or feature_provenance.get("feature_table_sha256")!=sha256(path("features"))):
            raise ValueError("Feature version/hash mismatch; legacy features must be recomputed")
        if frozen.get("segmentation_manifest_sha256")!=sha256(path("segmentation_manifest")):
            raise ValueError("Frozen model segmentation manifest identity mismatch")
        result=build_database(table("features"),grad,table("labels"))
        result["feature_schema_version"]=SCHEMA_VERSION
        result["feature_undefined_policy"]=feature_provenance["settings"]["undefined"]
        result["feature_particles_policy"]=feature_provenance["settings"]["particles"]
        result["segmentation_checkpoint_sha256"]=frozen["checkpoint_sha256"]
        output.mkdir(parents=True)
        result.to_csv(output/"database.csv",index=False);result.to_excel(output/"database.xlsx",index=False)
        write_json(output/"frozen_model.json",frozen)
    elif command=="run-logo":
        cli=[sys.executable,"-m","stacking_model.train",str(path("input")),"--out",str(output)]
        allowed={"models","targets","rules","trials","folds","max_base","shap_samples","permutations","pilot_trees","seed","mape_min","skip_final","paper_layout","final_models"}
        settings=config.get("settings",{})
        if set(settings)-allowed:raise ValueError("Unknown stacking settings")
        for key,value in settings.items():
            if isinstance(value,bool):
                if value:cli.append("--"+key.replace("_","-"))
            else:cli.extend(["--"+key.replace("_","-"),str(value)])
        subprocess.run(cli,check=True)
    elif command=="compare-databases":
        from .database import align_tables
        from .analysis import eta_squared
        inputs=config["databases"]
        if len(inputs)<2:raise ValueError("Need at least two aligned sources")
        paths=[(config_path.parent/p).resolve() for p in inputs.values()]
        input_paths.extend(paths)
        frames=align_tables([pd.read_csv(p) for p in paths])
        from .feature47 import HEADERS
        output.mkdir(parents=True)
        rows=[]
        for name,frame in zip(inputs,frames):
            eta=eta_squared(frame,HEADERS)
            categories={feature:("count" if i<16 else "area" if i<45 else "spatial") for i,feature in enumerate(HEADERS)}
            eta["category"]=eta.feature.map(categories)
            eta.to_csv(output/f"{name}_eta_squared.csv",index=False)
            summary=eta.groupby("category").eta_squared.agg(["mean","count","size"]).reset_index()
            summary.rename(columns={"mean":"mean_eta_squared","count":"n_defined","size":"n_features"}).to_csv(output/f"{name}_eta_categories.csv",index=False)
            rows.append(dict(source=name,n_images=len(frame),zero_instances=int(frame.N.eq(0).sum()),
                             feature_missing_cells=int(frame[HEADERS].isna().sum().sum())))
        pd.DataFrame(rows).to_csv(output/"paired_sources.csv",index=False)
    elif command=="validate-field":
        from .analysis import field_validation, cross_section_metrics
        points,sections,metrics=field_validation(table("patch_predictions"),table("cores"),config.get("expected_patches",4),config.get("expected_points",4))
        output.mkdir(parents=True)
        table("patch_predictions").to_csv(output/"patches.csv",index=False)
        points.to_csv(output/"points.csv",index=False);sections.to_csv(output/"cross_sections.csv",index=False)
        metrics.to_csv(output/"metrics_by_sieve.csv",index=False)
        cross_section_metrics(sections).to_csv(output/"cross_section_metrics.csv",index=False)
    elif command=="eval-seg":
        from .segmentation import evaluate_instances
        from .feature47 import _legacy
        samples=[]
        for row in table("samples").itertuples():
            candidate_path=(config_path.parent/row.candidates_path).resolve()
            truth_path=(config_path.parent/row.truth_path).resolve()
            input_paths.extend([candidate_path,truth_path])
            with np.load(candidate_path,allow_pickle=False) as data:
                samples.append(dict(image_id=row.image_id,truth=_legacy.read_mask(truth_path),
                                    masks=data["masks"].copy(),scores=data["scores"].copy()))
        if config.get("split") not in {"val","test"}:raise ValueError("Declare evaluation split")
        if config["split"]=="test" and not config.get("frozen_checkpoint_sha256"):
            raise ValueError("Independent test requires a frozen checkpoint identity")
        result=evaluate_instances(samples,**config.get("protocol",{}))
        output.mkdir(parents=True);write_json(output/"segmentation_metrics.json",result)
    elif command=="make-report":
        source=path("run")
        files=sorted(source.rglob("*.csv"))
        if not files:raise ValueError("No computed CSV artifacts in the run")
        input_paths.extend(files)
        output.mkdir(parents=True)
        if config.get("figures",True):
            from .reporting import make_figures
            make_figures(source,output/"figures")
        text=["# Run artifact index", "", "Computed artifacts only; no claim of manuscript replication.", ""]
        text.extend(f"- `{p.relative_to(source).as_posix()}`" for p in files)
        (output/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    # Every successful CLI records its config hash and actual elapsed time.
    code_paths=[*Path(__file__).parent.glob("*.py")]
    write_json(output/"command_manifest.json",manifest(list(dict.fromkeys(input_paths+code_paths)),dict(command=command,config=config,elapsed_seconds=time.perf_counter()-started)))
    print(output)
    return 0

if __name__=="__main__":raise SystemExit(main())
