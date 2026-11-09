"""Inference adapter for the preserved SAM3 model; no original-SAM substitution.
Real checkpoint inference remains unverified until matching weights are supplied.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
from .segmentation import instances_from_candidates
from .provenance import sha256,write_json


def export_predictions(inputs,output,predict,checkpoint_hash,score_threshold=.5,nms_iou=.7,min_area=1):
    """Backend contract: predict(PIL RGB) -> (NxHxW bool masks, N finite scores)."""
    output=Path(output)
    if output.exists():raise FileExistsError("Choose a new inference output directory")
    inputs=[Path(p) for p in inputs]
    if not inputs:raise ValueError("No input images")
    output.mkdir(parents=True);(output/"masks").mkdir();(output/"candidates").mkdir()
    marker=output/".incomplete";marker.write_text("Incomplete inference",encoding="utf-8")
    records=[]
    for index,path in enumerate(inputs):
        with Image.open(path) as source:image=source.convert("RGB")
        masks,scores=predict(image)
        masks=np.asarray(masks,bool);scores=np.asarray(scores,float)
        if masks.shape[1:]!=(image.height,image.width):raise ValueError("Candidate image dimensions do not match source")
        labels,instances=instances_from_candidates(masks,scores,score_threshold,nms_iou,min_area)
        name=f"{index:05d}"
        Image.fromarray(labels).save(output/"masks"/(name+".tif"))
        np.savez_compressed(output/"candidates"/(name+".npz"),masks=masks,scores=scores,checkpoint_sha256=np.asarray(checkpoint_hash))
        records.append(dict(image_id=name,source_filename=path.name,source_sha256=sha256(path),mask_path="masks/"+name+".tif",candidates_path="candidates/"+name+".npz",n_instances=len(instances),checkpoint_sha256=checkpoint_hash))
    pd.DataFrame(records).to_csv(output/"predictions.csv",index=False)
    write_json(output/"manifest.json",dict(backend="preserved-SAM3",checkpoint_sha256=checkpoint_hash,score_threshold=score_threshold,nms_iou=nms_iou,min_area=min_area,n_images=len(records)))
    marker.unlink()
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--sam3-root",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path,help="Full trained augmented-model state, not base SAM3 weights")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--prompt",default="aggregate particle")
    parser.add_argument("--score-threshold",type=float,default=.5)
    parser.add_argument("--nms-iou",type=float,default=.7)
    parser.add_argument("--min-area",type=int,default=1)
    args=parser.parse_args(argv)
    sys.path.insert(0,str(args.sam3_root.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"AA-SAM"))
    import torch
    from aggregate_sam3.model import build_aggregate_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    bpe=args.sam3_root/"sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model=build_aggregate_sam3_image_model(bpe_path=str(bpe),device="cpu",checkpoint_path=None,load_from_HF=False,eval_mode=True,freeze_mode="adapter")
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    state=checkpoint.get("model",checkpoint.get("state_dict",checkpoint))
    model.load_state_dict(state,strict=True)
    model=model.to(args.device).eval()
    processor=Sam3Processor(model,device=args.device,confidence_threshold=0.)
    def predict(image):
        with torch.inference_mode():
            state=processor.set_image(image)
            result=processor.set_text_prompt(args.prompt,state)
        return result["masks"].squeeze(1).cpu().numpy(),result["scores"].cpu().numpy()
    files=[args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff"})
    export_predictions(files,args.output,predict,sha256(args.checkpoint),args.score_threshold,args.nms_iou,args.min_area)
    return 0

if __name__=="__main__":raise SystemExit(main())
