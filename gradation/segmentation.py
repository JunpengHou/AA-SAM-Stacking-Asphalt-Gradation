"""Model-independent instance postprocessing and explicit evaluation protocol.
AP50 is all-points interpolated mask PR area (not COCO 101-point AP).
No SAM/SAM3 architecture or weights are silently substituted here.
"""
import numpy as np
from scipy import ndimage


def instances_from_candidates(masks, scores, score_threshold=.5, nms_iou=.7, min_area=1):
    masks = np.asarray(masks,dtype=bool)
    scores = np.asarray(scores,dtype=float)
    if masks.ndim != 3 or scores.ndim!=1 or len(scores)!=len(masks) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError("Expected NxHxW candidates with finite confidence scores")
    if not 0 <= score_threshold <= 1 or not 0 <= nms_iou <= 1 or min_area < 1:
        raise ValueError("Invalid filtering protocol")
    labels = np.zeros(masks.shape[1:],dtype=np.int32)
    accepted, records = [], []
    for i in np.argsort(-scores,kind="stable"):
        region=masks[i]; area=int(region.sum())
        if scores[i] < score_threshold or area < min_area: continue
        duplicate=False
        for previous in accepted:
            union=np.logical_or(region,previous).sum()
            if union and (region & previous).sum()/union > nms_iou:
                duplicate=True;break
        if duplicate:continue
        visible=region & (labels==0)
        if visible.sum() < min_area:continue
        instance_id=len(records)+1
        labels[visible]=instance_id
        accepted.append(region)
        records.append(dict(instance_id=instance_id,score=float(scores[i]),area_px=int(visible.sum()),
                            candidate_index=int(i),overlap_removed_px=area-int(visible.sum())))
    return labels,records


def boundary_f1(pred, truth, tolerance=2):
    if tolerance < 0 or int(tolerance) != tolerance:raise ValueError("Invalid boundary tolerance")
    pred,truth=np.asarray(pred,bool),np.asarray(truth,bool)
    if pred.shape!=truth.shape or pred.ndim!=2:raise ValueError("Boundary shape mismatch")
    pb=pred & ~ndimage.binary_erosion(pred,structure=np.ones((3,3)),border_value=0)
    tb=truth & ~ndimage.binary_erosion(truth,structure=np.ones((3,3)),border_value=0)
    if not pb.any() or not tb.any():return float(not pb.any() and not tb.any())
    precision=np.mean(ndimage.distance_transform_edt(~tb)[pb] <= tolerance)
    recall=np.mean(ndimage.distance_transform_edt(~pb)[tb] <= tolerance)
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.


def evaluate_instances(samples, score_threshold=.5, match_iou=.5, tolerance=2, max_dets=1000):
    if not 0<=score_threshold<=1 or not 0<match_iou<=1 or max_dets<1:
        raise ValueError("Invalid evaluation protocol")
    detections=[];ngt=0;seen=set()
    for sample in samples:
        image_id=sample["image_id"]
        if image_id in seen:raise ValueError("Repeated evaluation image")
        seen.add(image_id)
        gt=np.asarray(sample["truth"])
        masks=np.asarray(sample["masks"],bool);scores=np.asarray(sample["scores"],float)
        if gt.ndim!=2 or gt.dtype.kind not in "iu" or gt.min()<0 or scores.ndim!=1 or masks.ndim!=3 or masks.shape[1:]!=gt.shape or len(scores)!=len(masks):
            raise ValueError("Invalid evaluation arrays")
        if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError("Invalid confidence")
        truths=[gt==i for i in np.unique(gt) if i>0];ngt+=len(truths);used=set()
        for index in np.argsort(-scores,kind="stable")[:max_dets]:
            region=masks[index];best=-1;best_iou=0.
            for j,truth in enumerate(truths):
                if j in used:continue
                union=(region|truth).sum();iou=(region&truth).sum()/union if union else 0.
                if iou>best_iou:best,best_iou=j,iou
            match=best>=0 and best_iou>=match_iou
            bf=boundary_f1(region,truths[best],tolerance) if match else 0.
            if match:used.add(best)
            detections.append((float(scores[index]),int(match),bf))
    detections.sort(key=lambda r:-r[0])
    tp=np.cumsum([r[1] for r in detections]);fp=np.arange(1,len(tp)+1)-tp
    recalls=tp/ngt if ngt else np.zeros(len(tp))
    precisions=tp/np.maximum(tp+fp,1)
    r=np.r_[0.,recalls,1.];p=np.r_[0.,precisions,0.]
    p=np.maximum.accumulate(p[::-1])[::-1]
    changed=np.flatnonzero(r[1:]!=r[:-1])+1
    ap=float(np.sum((r[changed]-r[changed-1])*p[changed])) if ngt else None
    selected=[row for row in detections if row[0]>=score_threshold]
    true=sum(row[1] for row in selected);false=len(selected)-true;miss=ngt-true
    precision=true/len(selected) if selected else 0.;recall=true/ngt if ngt else 0.
    return {"Precision":precision,"Recall":recall,
            "F1":2*precision*recall/(precision+recall) if precision+recall else 0.,
            "Boundary_F1_matched":float(np.mean([row[2] for row in selected if row[1]])) if true else 0.,
            "Boundary_F1_GT_weighted":sum(row[2] for row in selected)/ngt if ngt else None,
            "Mask_AP50_all_points":ap if np.isclose(match_iou,.5) else None,
            "Mask_AP_at_IoU_all_points":ap,"TP":true,"FP":false,"FN":miss,
            "PR":dict(recall=recalls.tolist(),precision=precisions.tolist()),
            "protocol":dict(score_threshold=score_threshold,match_iou=match_iou,
                            boundary_tolerance_px=tolerance,boundary_connectivity=8,boundary_outside_background=True,max_dets=max_dets,
                            ap="all-points interpolated mask PR area; not COCO AP")}
