#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Batch-evaluate Aggregate-SAM3 checkpoint_*.pt files on one COCO validation split.

Metrics:
- COCO Mask mAP, AP50, AP75, AP_small/AP_medium/AP_large
- instance mIoU, Dice, Precision, Recall, F1
- Boundary F1
- Small / Medium / Large recall
- total/trainable parameter count
- forward-only speed and whole validation-pipeline speed

The script reuses the training run's aggregate_sam3_generated.yaml so model
architecture, validation preprocessing and SAM3 postprocessing stay aligned with
training. It does NOT retrain the model.
'''
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# On this Windows/SAM3 setup Pillow should be imported before torch/torchvision.
from PIL import Image  # noqa: F401

import numpy as np
import yaml

CHECKPOINT_RE = re.compile(r'^checkpoint_(\d+)\.pt$', re.IGNORECASE)

CORE_COLUMNS = [
    'epoch', 'checkpoint', 'status',
    'Mask_mAP', 'Mask_AP50', 'Mask_AP75',
    'mIoU', 'Dice', 'Precision', 'Recall', 'F1', 'Boundary_F1',
    'Small_Recall', 'Medium_Recall', 'Large_Recall',
    'TP', 'FP', 'FN', 'matched_instance_count', 'gt_instance_count',
    'pred_instance_count_after_score_filter',
    'Mask_AP_small', 'Mask_AP_medium', 'Mask_AP_large',
    'GT_mIoU_including_misses', 'GT_Dice_including_misses',
    'GT_Boundary_F1_including_misses',
    'score_threshold', 'match_iou', 'small_area_max', 'medium_area_max',
    'total_parameters', 'trainable_parameters', 'trainable_ratio',
    'forward_ms_per_image', 'forward_fps',
    'validation_pipeline_ms_per_image', 'validation_pipeline_fps',
    'validation_seconds', 'validation_image_count',
    'checkpoint_epoch_metadata', 'checkpoint_steps_metadata', 'error',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='批量评估 Aggregate-SAM3 每个 epoch 的 checkpoint。',
    )
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--sam3-root', type=Path, required=True)
    p.add_argument('--prompt', default='aggregate particle')
    p.add_argument('--training-config', type=Path, default=None)
    p.add_argument('--project-root', type=Path, default=None)
    p.add_argument('--launcher', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--split', choices=['val', 'valid', 'test'], default='val')
    p.add_argument(
        '--epochs', default='',
        help='指定 epoch，如 13,14,15 或 1,5,10-20；空=全部 checkpoint_N.pt。',
    )
    p.add_argument('--start-epoch', type=int, default=None)
    p.add_argument('--end-epoch', type=int, default=None)
    p.add_argument('--every', type=int, default=1)
    p.add_argument('--score-threshold', type=float, default=0.50)
    p.add_argument('--match-iou', type=float, default=0.50)
    p.add_argument('--boundary-tolerance', type=int, default=2)
    p.add_argument('--small-area-max', type=float, default=32.0 * 32.0)
    p.add_argument('--medium-area-max', type=float, default=96.0 * 96.0)
    p.add_argument('--max-dets', type=int, default=1000)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--stop-on-error', action='store_true')
    p.add_argument(
        '--keep-predictions', action='store_true',
        help='保留每个 checkpoint 的预测 JSON；默认计算完指标后删除以节省空间。',
    )
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def ensure_dependencies() -> None:
    missing = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append('torch')
    try:
        import pycocotools  # noqa: F401
    except Exception:
        missing.append('pycocotools')
    try:
        import cv2  # noqa: F401
    except Exception:
        missing.append('cv2/opencv-python')
    if missing:
        raise RuntimeError('缺少依赖：' + ', '.join(missing))


def resolve_annotation(split_root: Path, split_name: str) -> Path | None:
    for name in [
        '_annotations.coco.json', 'annotations.coco.json', 'annotations.json',
        f'aggregate_{split_name}_sam3.json', f'{split_name}.json',
        f'instances_{split_name}.json',
    ]:
        path = split_root / name
        if path.is_file():
            return path
    return None


def resolve_split(dataset_root: Path, split_name: str) -> tuple[Path, Path]:
    canonical = 'val' if split_name == 'valid' else split_name
    names = [canonical]
    if canonical == 'val':
        names.append('valid')
    for name in names:
        for split_root in [dataset_root / name, dataset_root / 'images' / name]:
            if not split_root.is_dir():
                continue
            image_dir = split_root / 'images'
            if not image_dir.is_dir():
                image_dir = split_root
            ann = resolve_annotation(split_root, name)
            if ann is None:
                ann = resolve_annotation(dataset_root, name)
            if ann is not None:
                return image_dir.resolve(), ann.resolve()
    raise FileNotFoundError(f'无法在 {dataset_root} 中解析 {split_name} 数据。')


def validate_coco_prompt(image_dir: Path, ann_file: Path, prompt: str) -> dict[str, Any]:
    data = json.loads(ann_file.read_text(encoding='utf-8'))
    images = data.get('images', [])
    anns = data.get('annotations', [])
    if not images:
        raise ValueError(f'COCO images 为空：{ann_file}')
    prompts = {str(x.get('text_input', '')) for x in images}
    bad = sorted(x for x in prompts if x != prompt)
    if bad:
        raise ValueError(f'验证集 text_input 不一致：期望={prompt!r}，实际={bad[:10]}')
    missing = []
    for image in images:
        fn = str(image.get('file_name', ''))
        if fn and not (image_dir / fn).is_file():
            missing.append(fn)
    if missing:
        raise FileNotFoundError(f'验证图像缺失 {len(missing)} 张，示例：{missing[:10]}')
    return {'images': len(images), 'annotations': len(anns), 'prompts': sorted(prompts)}


def parse_epoch_expression(text: str) -> set[int]:
    out: set[int] = set()
    if not text.strip():
        return out
    for token in text.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a, b = token.split('-', 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(token))
    return out


def discover_checkpoints(
    folder: Path,
    wanted: set[int],
    start: int | None,
    end: int | None,
    every: int,
) -> list[tuple[int, Path]]:
    items = []
    anchor = start if start is not None else 1
    for path in folder.iterdir():
        if not path.is_file():
            continue
        m = CHECKPOINT_RE.match(path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if wanted and epoch not in wanted:
            continue
        if start is not None and epoch < start:
            continue
        if end is not None and epoch > end:
            continue
        if every > 1 and (epoch - anchor) % every != 0:
            continue
        items.append((epoch, path.resolve()))
    return sorted(items)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'YAML 根节点不是字典：{path}')
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=140)


def json_clone(data: Any) -> Any:
    return json.loads(json.dumps(data))


def write_eval_helper(helper_dir: Path) -> Path:
    helper_dir.mkdir(parents=True, exist_ok=True)
    path = helper_dir / 'aggregate_sam3_eval_generated.py'
    helper = r'''from __future__ import annotations
import json
import time
import types
from pathlib import Path
from PIL import Image  # noqa: F401
import torch
from aggregate_sam3.single_gpu_trainer import SingleGPUTrainer


def _load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


class ModelOnlyCheckpointInitializer:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = str(checkpoint_path)

    def __call__(self, model):
        ckpt = _load(self.checkpoint_path)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        if not isinstance(state, dict):
            raise TypeError(f'checkpoint model state type={type(state)!r}')
        if state and all(str(k).startswith('module.') for k in state):
            state = {str(k)[7:]: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError("Checkpoint architecture mismatch; refusing partial evaluation: "
                               + str((missing, unexpected)))
        if missing or unexpected:
            raise RuntimeError(
                'checkpoint 与当前模型结构不一致\n'
                f'missing({len(missing)}): {missing[:30]}\n'
                f'unexpected({len(unexpected)}): {unexpected[:30]}'
            )
        if isinstance(ckpt, dict):
            model._evaluation_checkpoint_epoch = ckpt.get('epoch')
            model._evaluation_checkpoint_steps = ckpt.get('steps')
        return model


class EvaluationSingleGPUTrainer(SingleGPUTrainer):
    def __init__(self, *args, eval_stats_path: str, **kwargs):
        self.eval_stats_path = str(eval_stats_path)
        self._forward_seconds = 0.0
        self._forward_images = 0
        self._forward_calls = 0
        super().__init__(*args, **kwargs)

    def _write_stats(self, validation_seconds: float):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        payload = {
            'total_parameters': int(total),
            'trainable_parameters': int(trainable),
            'trainable_ratio': float(trainable / total) if total else 0.0,
            'forward_seconds': float(self._forward_seconds),
            'forward_images': int(self._forward_images),
            'forward_calls': int(self._forward_calls),
            'validation_seconds': float(validation_seconds),
            'checkpoint_epoch_metadata': getattr(self.model, '_evaluation_checkpoint_epoch', None),
            'checkpoint_steps_metadata': getattr(self.model, '_evaluation_checkpoint_steps', None),
        }
        if self._forward_seconds > 0 and self._forward_images > 0:
            payload['forward_ms_per_image'] = 1000.0 * self._forward_seconds / self._forward_images
            payload['forward_fps'] = self._forward_images / self._forward_seconds
        else:
            payload['forward_ms_per_image'] = None
            payload['forward_fps'] = None
        p = Path(self.eval_stats_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def run(self):
        original_forward = self.model.forward
        trainer = self

        def timed_forward(_model, *args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = original_forward(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            trainer._forward_seconds += time.perf_counter() - t0
            n = 1
            if args:
                try:
                    n = len(args[0].img_batch)
                except Exception:
                    n = 1
            trainer._forward_images += int(n)
            trainer._forward_calls += 1
            return output

        self.model.forward = types.MethodType(timed_forward, self.model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return super().run()
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            self.model.forward = original_forward
            self._write_stats(elapsed)
'''
    path.write_text(helper, encoding='utf-8')
    return path


def set_val_paths(config: dict[str, Any], image_dir: Path, ann_file: Path, workers: int | None) -> None:
    val = config['trainer']['data']['val']
    ds = val['dataset']
    ds['img_folder'] = str(image_dir)
    ds['ann_file'] = str(ann_file)
    ds['training'] = False
    ds['max_ann_per_img'] = max(int(ds.get('max_ann_per_img', 5000)), 5000)
    ds['max_val_queries'] = max(int(ds.get('max_val_queries', 5000)), 5000)
    val['batch_size'] = 1
    val['shuffle'] = False
    val['drop_last'] = False
    if workers is not None:
        val['num_workers'] = int(workers)
    val.setdefault('collate_fn', {})['dict_key'] = 'aggregate_particle'
    val['collate_fn']['with_seg_masks'] = True


def build_meter(ann_file: Path, dump_dir: Path, max_dets: int) -> dict[str, Any]:
    return {
        'val': {
            'aggregate_particle': {
                'segmentation': {
                    '_target_': 'sam3.eval.coco_writer.PredictionDumper',
                    'iou_type': 'segm',
                    'dump_dir': str(dump_dir),
                    'merge_predictions': True,
                    'gather_pred_via_filesys': False,
                    'maxdets': int(max_dets),
                    'postprocessor': {
                        '_target_': 'sam3.eval.postprocessors.PostProcessImage',
                        'max_dets_per_img': int(max_dets),
                        'iou_type': 'segm',
                        'to_cpu': True,
                        'use_original_ids': True,
                        'use_original_sizes_box': True,
                        'use_original_sizes_mask': True,
                        'convert_mask_to_rle': False,
                        'always_interpolate_masks_on_gpu': True,
                        'use_presence': True,
                        'detection_threshold': -1.0,
                    },
                    'pred_file_evaluators': [{
                        '_target_': 'sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators',
                        'gt_path': str(ann_file),
                        'tide': False,
                        'iou_type': 'segm',
                    }],
                }
            }
        }
    }


def prepare_eval_config(
    template: dict[str, Any], checkpoint: Path, image_dir: Path, ann_file: Path,
    eval_root: Path, max_dets: int, workers: int | None,
) -> dict[str, Any]:
    cfg = json_clone(template)
    tr = cfg['trainer']
    tr['_target_'] = 'aggregate_sam3_eval_generated.EvaluationSingleGPUTrainer'
    tr['eval_stats_path'] = str(eval_root / 'model_eval_stats.json')
    tr['mode'] = 'val'
    tr['skip_first_val'] = False
    tr['skip_saving_ckpts'] = True
    tr['empty_gpu_mem_cache_after_eval'] = True
    set_val_paths(cfg, image_dir, ann_file, workers)
    pred_dir = eval_root / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    tr['meters'] = build_meter(ann_file, pred_dir, max_dets)

    ck = tr.setdefault('checkpoint', {})
    ck['save_dir'] = str(eval_root / '_empty_checkpoint_dir')
    ck['save_freq'] = 0
    ck['save_list'] = []
    ck['save_best_meters'] = []
    ck['skip_saving_parameters'] = []
    ck['resume_from'] = None
    ck['model_weight_initializer'] = {
        '_target_': 'aggregate_sam3_eval_generated.ModelOnlyCheckpointInitializer',
        'checkpoint_path': str(checkpoint),
    }
    empty = Path(ck['save_dir'])
    if empty.exists():
        shutil.rmtree(empty)
    empty.mkdir(parents=True, exist_ok=True)

    log = tr.setdefault('logging', {})
    log['log_dir'] = str(eval_root / 'logs')
    if isinstance(log.get('tensorboard_writer'), dict):
        log['tensorboard_writer']['log_dir'] = str(eval_root / 'tensorboard')
        log['tensorboard_writer']['should_log'] = False
    log['wandb_writer'] = None
    if isinstance(cfg.get('launcher'), dict):
        cfg['launcher']['experiment_log_dir'] = str(eval_root)
        cfg['launcher']['num_nodes'] = 1
        cfg['launcher']['gpus_per_node'] = 1
    if isinstance(cfg.get('submitit'), dict):
        cfg['submitit']['use_cluster'] = False
    return cfg


def stream_process(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8', errors='replace') as log:
        proc = subprocess.Popen(
            command, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        return int(proc.wait())


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch
    try:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location='cpu')
        if isinstance(ckpt, dict):
            out = {'epoch': ckpt.get('epoch'), 'steps': ckpt.get('steps')}
            del ckpt
            return out
    except Exception as exc:
        return {'error': repr(exc)}
    return {}


def read_last_json_line(path: Path) -> dict[str, Any]:
    last = {}
    if not path.is_file():
        return last
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            value = json.loads(line.strip())
        except Exception:
            continue
        if isinstance(value, dict):
            last = value
    return last


def extract_coco_metrics(stats: dict[str, Any]) -> dict[str, float]:
    mapping = {
        'Mask_mAP': 'coco_eval_segm_AP',
        'Mask_AP50': 'coco_eval_segm_AP_50',
        'Mask_AP75': 'coco_eval_segm_AP_75',
        'Mask_AP_small': 'coco_eval_segm_AP_small',
        'Mask_AP_medium': 'coco_eval_segm_AP_medium',
        'Mask_AP_large': 'coco_eval_segm_AP_large',
    }
    out = {}
    for name, suffix in mapping.items():
        for key, value in stats.items():
            k = str(key).replace('\\', '/')
            if k.endswith(suffix):
                try:
                    out[name] = float(value)
                    break
                except Exception:
                    pass
    return out


def normalize_rle(rle: dict[str, Any]) -> dict[str, Any]:
    out = {'size': [int(rle['size'][0]), int(rle['size'][1])], 'counts': rle['counts']}
    if isinstance(out['counts'], str):
        out['counts'] = out['counts'].encode('ascii')
    return out


def boundary(mask: np.ndarray) -> np.ndarray:
    import cv2
    x = np.asarray(mask, dtype=np.uint8)
    if x.ndim == 3:
        x = x[..., 0]
    if not x.any():
        return np.zeros_like(x, dtype=bool)
    eroded = cv2.erode(x, np.ones((3, 3), np.uint8), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (x > 0) & (eroded == 0)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int) -> float:
    """Euclidean boundary distance, matching revised manuscript p10 (not square dilation)."""
    from scipy.ndimage import distance_transform_edt
    if tolerance < 0:
        raise ValueError("Boundary tolerance must be nonnegative")
    pb, gb = boundary(pred), boundary(gt)
    if not pb.any() or not gb.any():
        return float(not pb.any() and not gb.any())
    bp = float(np.mean(distance_transform_edt(~gb)[pb] <= tolerance))
    br = float(np.mean(distance_transform_edt(~pb)[gb] <= tolerance))
    return 2.0 * bp * br / (bp + br) if bp + br else 0.0


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def custom_metrics(
    ann_file: Path, pred_file: Path, score_thr: float, match_iou: float,
    boundary_tolerance: int, small_max: float, medium_max: float, max_dets: int,
) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils

    coco = COCO(str(ann_file))
    preds_raw = json.loads(pred_file.read_text(encoding='utf-8'))
    if not isinstance(preds_raw, list):
        raise ValueError('预测JSON必须是list。')

    gt = defaultdict(list)
    size_total = {'small': 0, 'medium': 0, 'large': 0}
    total_gt = 0
    for ann in coco.dataset.get('annotations', []):
        if int(ann.get('iscrowd', 0)):
            continue
        rle = normalize_rle(coco.annToRLE(ann))
        area = float(ann.get('area', mask_utils.area(rle)))
        group = 'small' if area < small_max else ('medium' if area < medium_max else 'large')
        gt[(int(ann['image_id']), int(ann['category_id']))].append(
            {'rle': rle, 'area': area, 'group': group}
        )
        total_gt += 1
        size_total[group] += 1

    pred = defaultdict(list)
    for p in preds_raw:
        score = float(p.get('score', 0.0))
        if score < score_thr or 'segmentation' not in p:
            continue
        pred[(int(p['image_id']), int(p['category_id']))].append(
            {'score': score, 'rle': normalize_rle(p['segmentation'])}
        )
    for key in pred:
        pred[key].sort(key=lambda x: x['score'], reverse=True)
        pred[key] = pred[key][:max_dets]

    tp = fp = fn = 0
    total_pred = 0
    ious, dices, bf1s = [], [], []
    size_match = {'small': 0, 'medium': 0, 'large': 0}

    for key in set(gt) | set(pred):
        gs, ps = gt.get(key, []), pred.get(key, [])
        total_pred += len(ps)
        if not gs:
            fp += len(ps)
            continue
        if not ps:
            fn += len(gs)
            continue
        gr = [x['rle'] for x in gs]
        pr = [x['rle'] for x in ps]
        matrix = np.asarray(mask_utils.iou(pr, gr, [0] * len(gr)), dtype=np.float64)
        used = set()
        for pi in range(len(ps)):
            best_g, best_i = None, -1.0
            for gi in range(len(gs)):
                if gi in used:
                    continue
                val = float(matrix[pi, gi])
                if val > best_i:
                    best_i, best_g = val, gi
            if best_g is not None and best_i >= match_iou:
                used.add(best_g)
                tp += 1
                ious.append(best_i)
                dices.append(2.0 * best_i / (1.0 + best_i) if best_i > 0 else 0.0)
                size_match[gs[best_g]['group']] += 1
                pm = mask_utils.decode(pr[pi])
                gm = mask_utils.decode(gr[best_g])
                bf1s.append(boundary_f1(pm, gm, boundary_tolerance))
                del pm, gm
            else:
                fp += 1
        fn += len(gs) - len(used)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    miou = float(np.mean(ious)) if ious else 0.0
    dice = float(np.mean(dices)) if dices else 0.0
    bf1 = float(np.mean(bf1s)) if bf1s else 0.0

    return {
        'mIoU': miou,
        'Dice': dice,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Boundary_F1': bf1,
        'Small_Recall': safe_div(size_match['small'], size_total['small']),
        'Medium_Recall': safe_div(size_match['medium'], size_total['medium']),
        'Large_Recall': safe_div(size_match['large'], size_total['large']),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
        'matched_instance_count': int(tp),
        'gt_instance_count': int(total_gt),
        'pred_instance_count_after_score_filter': int(total_pred),
        'GT_mIoU_including_misses': safe_div(sum(ious), total_gt),
        'GT_Dice_including_misses': safe_div(sum(dices), total_gt),
        'GT_Boundary_F1_including_misses': safe_div(sum(bf1s), total_gt),
        'small_gt_count': int(size_total['small']),
        'medium_gt_count': int(size_total['medium']),
        'large_gt_count': int(size_total['large']),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    x = json.loads(path.read_text(encoding='utf-8'))
    return x if isinstance(x, dict) else {}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def find_prediction(root: Path) -> Path | None:
    d = root / 'predictions'
    for name in ['coco_predictions_segm.json', 'coco_predictions_segm_0.json']:
        p = d / name
        if p.is_file():
            return p
    if d.is_dir():
        matches = sorted(d.glob('*predictions*segm*.json'))
        return matches[0] if matches else None
    return None


def make_row(
    epoch: int, checkpoint: Path, metrics: dict[str, Any], model: dict[str, Any],
    meta: dict[str, Any], image_count: int, args: argparse.Namespace,
    status: str, error: str = '',
) -> dict[str, Any]:
    row = {k: '' for k in CORE_COLUMNS}
    row.update({
        'epoch': epoch, 'checkpoint': str(checkpoint), 'status': status,
        'score_threshold': args.score_threshold, 'match_iou': args.match_iou,
        'small_area_max': args.small_area_max, 'medium_area_max': args.medium_area_max,
        'validation_image_count': image_count,
        'checkpoint_epoch_metadata': meta.get('epoch', ''),
        'checkpoint_steps_metadata': meta.get('steps', ''), 'error': error,
    })
    row.update(metrics)
    for key in [
        'total_parameters', 'trainable_parameters', 'trainable_ratio',
        'forward_ms_per_image', 'forward_fps', 'validation_seconds',
    ]:
        if key in model:
            row[key] = model[key]
    sec = model.get('validation_seconds')
    if sec and image_count:
        row['validation_pipeline_ms_per_image'] = 1000.0 * float(sec) / image_count
        row['validation_pipeline_fps'] = image_count / float(sec)
    return row


def fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    extra = sorted({k for r in rows for k in r if k not in CORE_COLUMNS})
    return CORE_COLUMNS + extra


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = fieldnames(rows)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=names, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def num(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def choose_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        'Mask_mAP', 'Mask_AP50', 'Mask_AP75', 'mIoU', 'Dice', 'Precision',
        'Recall', 'F1', 'Boundary_F1', 'Small_Recall', 'Medium_Recall',
        'Large_Recall', 'GT_mIoU_including_misses', 'GT_Dice_including_misses',
        'GT_Boundary_F1_including_misses', 'forward_fps',
    ]
    out = {}
    ok = [r for r in rows if r.get('status') == 'ok']
    for metric in metrics:
        candidates = [(num(r.get(metric)), r) for r in ok]
        candidates = [(v, r) for v, r in candidates if v is not None]
        if candidates:
            value, row = max(candidates, key=lambda x: x[0])
            out[metric] = {
                'epoch': int(row['epoch']), 'value': float(value),
                'checkpoint': row['checkpoint'],
            }
    return out


def definitions(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'Mask_mAP': 'SAM3官方COCO segmentation AP@[0.50:0.95]。',
        'Mask_AP50': 'SAM3官方COCO segmentation AP@0.50。',
        'Mask_AP75': 'SAM3官方COCO segmentation AP@0.75。',
        'Precision/Recall/F1': (
            f'score>={args.score_threshold}；预测按置信度降序与同类别未匹配GT一对一匹配；'
            f'Mask IoU>={args.match_iou} 为TP。'
        ),
        'mIoU': 'TP匹配实例对的Mask IoU平均值。',
        'Dice': 'TP匹配实例对的Dice平均值。',
        'Boundary_F1': f'TP匹配实例对的边界F1平均值，容差±{args.boundary_tolerance}px。',
        'GT_*_including_misses': '以全部GT为分母，FN按0分计入。',
        'Small_Recall': f'GT area < {args.small_area_max:g} px^2。',
        'Medium_Recall': f'{args.small_area_max:g} <= GT area < {args.medium_area_max:g} px^2。',
        'Large_Recall': f'GT area >= {args.medium_area_max:g} px^2。',
        'forward_ms_per_image': 'CUDA同步的model.forward纯前向耗时/图，不含DataLoader和COCO后处理。',
        'validation_pipeline_ms_per_image': '整个验证流程耗时/图，包含读取、forward、后处理和指标同步。',
        'checkpoint_epoch_mapping': '官方Trainer先保存checkpoint_(epoch+1).pt，再验证当前Trainer/epoch，因此checkpoint_N通常对应日志Trainer/epoch=N-1。',
        'important': '最佳权重建议仍以验证集Mask AP50为主；其它指标用于完整性能分析。',
    }


def write_excel(path: Path, rows: list[dict[str, Any]], best: dict[str, Any], defs: dict[str, Any]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except Exception:
        return False
    wb = Workbook()
    ws = wb.active
    ws.title = 'checkpoint_metrics'
    names = fieldnames(rows)
    ws.append(names)
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in rows:
        ws.append([row.get(k, '') for k in names])
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    for i, name in enumerate(names, 1):
        ws.column_dimensions[get_column_letter(i)].width = max(12, min(34, len(name) + 3))

    bs = wb.create_sheet('best_checkpoints')
    bs.append(['metric', 'epoch', 'value', 'checkpoint'])
    for c in bs[1]:
        c.font = Font(bold=True)
    for metric, item in best.items():
        bs.append([metric, item['epoch'], item['value'], item['checkpoint']])

    ds = wb.create_sheet('definitions')
    ds.append(['item', 'definition'])
    for c in ds[1]:
        c.font = Font(bold=True)
    for k, v in defs.items():
        ds.append([k, str(v)])
    ds.column_dimensions['A'].width = 40
    ds.column_dimensions['B'].width = 110
    wb.save(path)
    return True


def plots(output: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[提示] matplotlib不可用，跳过曲线：{exc}')
        return
    metrics = [
        'Mask_mAP', 'Mask_AP50', 'Mask_AP75', 'mIoU', 'Dice', 'Precision',
        'Recall', 'F1', 'Boundary_F1', 'Small_Recall', 'Medium_Recall',
        'Large_Recall', 'GT_mIoU_including_misses', 'forward_ms_per_image',
    ]
    curve_dir = output / 'curves'
    curve_dir.mkdir(parents=True, exist_ok=True)
    ok = sorted([r for r in rows if r.get('status') == 'ok'], key=lambda r: int(r['epoch']))
    for metric in metrics:
        xy = [(int(r['epoch']), num(r.get(metric))) for r in ok]
        xy = [(x, y) for x, y in xy if y is not None]
        if not xy:
            continue
        x, y = zip(*xy)
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(x, y, marker='o', linewidth=1.4, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs Epoch')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(curve_dir / f'{metric}.png', dpi=220, bbox_inches='tight')
        plt.close(fig)


def existing_rows(output: Path) -> dict[int, dict[str, Any]]:
    out = {}
    for p in output.glob('checkpoint_*/metrics.json') if output.is_dir() else []:
        try:
            payload = load_json(p)
            if payload.get('status') == 'ok' and isinstance(payload.get('table_row'), dict):
                out[int(payload['epoch'])] = payload['table_row']
        except Exception:
            pass
    return out


def main() -> None:
    args = parse_args()
    ensure_dependencies()
    if not 0 <= args.score_threshold <= 1:
        raise ValueError('--score-threshold必须在[0,1]。')
    if not 0 < args.match_iou <= 1:
        raise ValueError('--match-iou必须在(0,1]。')
    if args.boundary_tolerance < 0:
        raise ValueError('--boundary-tolerance不能小于0。')
    if args.small_area_max <= 0 or args.medium_area_max <= args.small_area_max:
        raise ValueError('颗粒面积阈值设置错误。')
    if args.max_dets < 1 or args.every < 1:
        raise ValueError('--max-dets和--every必须>=1。')
    if args.epochs and (args.start_epoch is not None or args.end_epoch is not None):
        raise ValueError('--epochs不能和--start-epoch/--end-epoch同时使用。')

    if args.split == 'test':
        raise ValueError(
            "This checkpoint-sweep tool is validation-only. Freeze one checkpoint using val, "
            "then evaluate it once with the independent-test adapter. Test-based checkpoint selection is forbidden."
        )
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    sam3_root = args.sam3_root.expanduser().resolve()
    for p in [checkpoint_dir, dataset_root, sam3_root]:
        if not p.is_dir():
            raise NotADirectoryError(p)

    script_dir = Path(__file__).resolve().parent
    project_root = args.project_root.expanduser().resolve() if args.project_root else script_dir.parent
    launcher = args.launcher.expanduser().resolve() if args.launcher else script_dir / 'launch_train_no_gloo_stable.py'
    training_config = (
        args.training_config.expanduser().resolve()
        if args.training_config else checkpoint_dir.parent / 'aggregate_sam3_generated.yaml'
    )
    output = args.output_dir.expanduser().resolve() if args.output_dir else checkpoint_dir.parent / 'checkpoint_evaluation'
    output.mkdir(parents=True, exist_ok=True)

    if not launcher.is_file():
        raise FileNotFoundError(f'找不到启动器：{launcher}')
    if not training_config.is_file():
        raise FileNotFoundError(f'找不到训练配置：{training_config}')
    if not (project_root / 'aggregate_sam3').is_dir():
        raise FileNotFoundError(f'找不到项目包：{project_root / "aggregate_sam3"}')

    image_dir, ann_file = resolve_split(dataset_root, args.split)
    ds_stats = validate_coco_prompt(image_dir, ann_file, args.prompt.strip())
    wanted = parse_epoch_expression(args.epochs)
    checkpoints = discover_checkpoints(
        checkpoint_dir, wanted, args.start_epoch, args.end_epoch, args.every
    )
    if not checkpoints:
        raise FileNotFoundError('没有找到符合条件的 checkpoint_N.pt。')

    template = load_yaml(training_config)
    helper_dir = output / 'generated_helpers'
    helper_path = write_eval_helper(helper_dir)
    defs = definitions(args)
    save_json(output / 'metric_definitions.json', defs)

    print('=' * 88)
    print('Aggregate-SAM3 checkpoint 批量评估')
    print(f'checkpoint目录：{checkpoint_dir}')
    print(f'训练配置：{training_config}')
    print(f'验证图像：{image_dir}')
    print(f'验证COCO：{ann_file}')
    print(f'验证数据：{ds_stats}')
    print(f'项目根目录：{project_root}')
    print(f'官方SAM3：{sam3_root}')
    print(f'输出：{output}')
    print(f'辅助模块：{helper_path}')
    print('epoch：' + ','.join(str(e) for e, _ in checkpoints))
    print(f'自定义匹配：score>={args.score_threshold}, IoU>={args.match_iou}')
    print('=' * 88)
    print('[重要] 请不要和正在运行的训练同时执行本脚本；两者会争用同一块GPU。')

    env = os.environ.copy()
    env.setdefault('LOCAL_RANK', '0')
    env.setdefault('RANK', '0')
    env.setdefault('WORLD_SIZE', '1')
    env.setdefault('LOCAL_WORLD_SIZE', '1')
    env['PYTHONUTF8'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(
        x for x in [str(helper_dir), str(project_root), str(sam3_root), env.get('PYTHONPATH', '')] if x
    )

    rows_by_epoch = existing_rows(output)

    for idx, (epoch, checkpoint) in enumerate(checkpoints, 1):
        root = output / f'checkpoint_{epoch:03d}'
        result_file = root / 'metrics.json'
        if not args.overwrite and result_file.is_file():
            old = load_json(result_file)
            if old.get('status') == 'ok' and isinstance(old.get('table_row'), dict):
                rows_by_epoch[epoch] = old['table_row']
                print(f'[跳过] epoch={epoch} 已评估；需要重算时添加 --overwrite。')
                continue
        if args.overwrite and root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print('\n' + '#' * 88)
        print(f'[{idx}/{len(checkpoints)}] epoch={epoch}  {checkpoint.name}')
        print('#' * 88)
        # 不在父进程额外torch.load 3~4GB checkpoint；
        # 元数据由验证子进程在本来就要加载权重时顺带读取。
        meta = {}

        try:
            cfg = prepare_eval_config(
                template, checkpoint, image_dir, ann_file, root, args.max_dets, args.num_workers
            )
            config_path = root / 'config_eval.yaml'
            save_yaml(config_path, cfg)
            if args.dry_run:
                row = make_row(epoch, checkpoint, {}, {}, meta, ds_stats['images'], args, 'dry_run')
                rows_by_epoch[epoch] = row
                save_json(result_file, {'status': 'dry_run', 'epoch': epoch, 'table_row': row})
                continue

            cmd = [sys.executable, str(launcher), '--config', str(config_path)]
            print('[启动验证]', subprocess.list2cmdline(cmd))
            rc = stream_process(cmd, project_root, env, root / 'console.log')
            if rc != 0:
                raise RuntimeError(f'验证子进程退出码={rc}，见 {root / "console.log"}')

            pred_file = find_prediction(root)
            if pred_file is None:
                raise FileNotFoundError('找不到 coco_predictions_segm.json。')
            val_stats = read_last_json_line(root / 'logs' / 'val_stats.json')
            official = extract_coco_metrics(val_stats)
            custom = custom_metrics(
                ann_file, pred_file, args.score_threshold, args.match_iou,
                args.boundary_tolerance, args.small_area_max,
                args.medium_area_max, args.max_dets,
            )
            metrics = {**official, **custom}
            model_stats = load_json(root / 'model_eval_stats.json')
            meta = {
                'epoch': model_stats.get('checkpoint_epoch_metadata'),
                'steps': model_stats.get('checkpoint_steps_metadata'),
            }
            row = make_row(
                epoch, checkpoint, metrics, model_stats, meta,
                ds_stats['images'], args, 'ok'
            )
            rows_by_epoch[epoch] = row
            save_json(result_file, {
                'status': 'ok', 'epoch': epoch, 'checkpoint': str(checkpoint),
                'checkpoint_metadata': meta, 'official_coco_metrics': official,
                'custom_instance_metrics': custom, 'model_and_speed': model_stats,
                'table_row': row,
            })
            print(
                '[结果] '
                f'AP50={row.get("Mask_AP50")} | AP75={row.get("Mask_AP75")} | '
                f'mIoU={row.get("mIoU"):.6f} | Dice={row.get("Dice"):.6f} | '
                f'P={row.get("Precision"):.6f} | R={row.get("Recall"):.6f} | '
                f'F1={row.get("F1"):.6f} | BF1={row.get("Boundary_F1"):.6f}'
            )
            if not args.keep_predictions:
                try:
                    pred_file.unlink()
                    print('[磁盘清理] 已删除预测JSON；metrics.json仍完整保留结果。')
                except Exception as exc:
                    print(f'[提示] 删除预测JSON失败：{exc}')
        except Exception as exc:
            error = repr(exc)
            print(f'[失败] epoch={epoch}: {error}', file=sys.stderr)
            row = make_row(epoch, checkpoint, {}, {}, meta, ds_stats['images'], args, 'failed', error)
            rows_by_epoch[epoch] = row
            save_json(result_file, {
                'status': 'failed', 'epoch': epoch, 'checkpoint': str(checkpoint),
                'error': error, 'table_row': row,
            })
            if args.stop_on_error:
                raise

        current = [rows_by_epoch[e] for e in sorted(rows_by_epoch)]
        write_csv(output / 'checkpoint_metrics.csv', current)

    rows = [rows_by_epoch[e] for e in sorted(rows_by_epoch)]
    write_csv(output / 'checkpoint_metrics.csv', rows)
    best = choose_best(rows)
    save_json(output / 'best_checkpoints.json', best)
    excel_ok = write_excel(output / 'checkpoint_metrics.xlsx', rows, best, defs)
    if not args.no_plots:
        plots(output, rows)

    print('\n' + '=' * 88)
    print('[全部完成]')
    print(f'CSV：{output / "checkpoint_metrics.csv"}')
    if excel_ok:
        print(f'Excel：{output / "checkpoint_metrics.xlsx"}')
    else:
        print('[提示] 未安装openpyxl，因此只生成CSV；可 conda install openpyxl。')
    print(f'最佳权重：{output / "best_checkpoints.json"}')
    if not args.no_plots:
        print(f'曲线：{output / "curves"}')
    if 'Mask_AP50' in best:
        x = best['Mask_AP50']
        print(f'[AP50最佳] epoch={x["epoch"]}, AP50={x["value"]:.6f}')
        print(f'             {x["checkpoint"]}')
    print('=' * 88)


if __name__ == '__main__':
    main()
