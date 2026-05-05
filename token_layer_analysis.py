import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ensure registries are populated
import proxyclip_segmentor  # noqa: F401
import custom_datasets  # noqa: F401

from mmengine.config import Config
from mmengine.runner import Runner


@dataclass
class LayerAgg:
    # Pixel-level metrics (standard segmentation semantics)
    px_correct: torch.Tensor  # [L]
    px_total: torch.Tensor  # [L]
    px_inter: torch.Tensor  # [L, C]
    px_union: torch.Tensor  # [L, C]

    # Token-level convergence dynamics
    first_hist: torch.Tensor  # [L]
    converge_hist: torch.Tensor  # [L]
    stable_ratio_final_hist: torch.Tensor  # [L]
    stable_ratio_gt_hist: torch.Tensor  # [L]
    valid_tokens: int
    pred_equal_pixels: int
    pred_total_pixels: int


def _first_and_converge_layers(
    correct_mat: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute first-correct layer and convergence layer for each token.

    Args:
        correct_mat: [L, N] bool (already masked with valid)
        valid: [N] bool
    Returns:
        first: [N] int64, -1 if never correct.
        converge: [N] int64, -1 if never converged.

    Strict converge definition (per token): the earliest layer l such that
    it is correct at layer l AND remains correct for all subsequent layers.
    """
    if correct_mat.dtype != torch.bool:
        correct_mat = correct_mat.to(torch.bool)
    L, N = correct_mat.shape

    valid = valid.to(torch.bool)
    cm = correct_mat & valid.unsqueeze(0)

    any_correct = cm.any(dim=0)
    first = torch.full((N,), -1, device=cm.device, dtype=torch.long)
    if any_correct.any():
        first[any_correct] = cm[:, any_correct].float().argmax(dim=0)

    # strict suffix-all correctness
    cm_int = cm.to(torch.int32)
    suffix_all = torch.flip(torch.cumprod(torch.flip(cm_int, dims=[0]), dim=0), dims=[0]).to(torch.bool)  # [L,N]
    converge_ok = suffix_all

    any_conv = converge_ok.any(dim=0)
    converge = torch.full((N,), -1, device=cm.device, dtype=torch.long)
    if any_conv.any():
        converge[any_conv] = converge_ok[:, any_conv].float().argmax(dim=0)

    return first, converge


def _stable_layer_by_ratio(
    correct_mat: torch.Tensor,
    valid: torch.Tensor,
    ratio_threshold: float,
) -> torch.Tensor:
    """Earliest layer where suffix correctness ratio reaches threshold.

    Args:
        correct_mat: [L, N] bool
        valid: [N] bool
        ratio_threshold: in [0, 1]
    Returns:
        stable: [N] int64, -1 if never reaches threshold.
    """
    if correct_mat.dtype != torch.bool:
        correct_mat = correct_mat.to(torch.bool)

    valid = valid.to(torch.bool)
    cm = correct_mat & valid.unsqueeze(0)

    L = cm.shape[0]
    suffix_hits = torch.flip(torch.cumsum(torch.flip(cm.float(), dims=[0]), dim=0), dims=[0])
    suffix_len = torch.arange(L, 0, -1, device=cm.device, dtype=torch.float32).unsqueeze(1)
    suffix_ratio = suffix_hits / suffix_len
    meets = suffix_ratio >= float(ratio_threshold)

    stable = torch.full((cm.shape[1],), -1, device=cm.device, dtype=torch.long)
    for l in range(L):
        ok = (stable < 0) & meets[l] & valid
        stable[ok] = l
    return stable


def _resolve_token_grid(model, img_hw: Tuple[int, int]) -> Tuple[int, int]:
    """Infer native token grid in forward_feature before upsampling.

    This mirrors proxyclip_segmentor.forward_feature spatial logic so layer
    logits can be reshaped without touching model code.
    """
    h, w = img_hw
    if model.vfm_model == 'sam':
        # SAM image encoder runs on 1024x1024 in forward_feature.
        return (64, 64)
    if model.vfm_model == 'dino':
        patch = model.vfm.patch_embed.patch_size
        return (h // patch, w // patch)
    if model.vfm_model == 'dinov2':
        patch = model.vfm.patch_embed.patch_size
        return (h // patch[0], w // patch[1])
    if model.vfm_model == 'mae':
        patch = model.vfm.patch_embed.patch_size
        return (model.slide_crop // patch[0], model.slide_crop // patch[1])

    clip_patch_h, clip_patch_w = model.clip.visual.patch_size
    return (h // clip_patch_h, w // clip_patch_w)


def _seg_logits_to_pred(model, seg_logits: torch.Tensor) -> torch.Tensor:
    """Reproduce proxyclip_segmentor.postprocess_result without mutating samples.

    Args:
        seg_logits: [B, Q, H, W] query-level logits before logit_scale and softmax.
    Returns:
        pred: [B, H, W] class prediction map.
    """
    batch = seg_logits.shape[0]
    preds: List[torch.Tensor] = []
    num_cls, num_queries = max(model.query_idx) + 1, len(model.query_idx)
    cls_index = None
    if num_cls != num_queries:
        cls_index = F.one_hot(model.query_idx, num_classes=num_cls).T.view(num_cls, num_queries, 1, 1)

    for i in range(batch):
        cur = seg_logits[i] * model.logit_scale
        cur = cur.softmax(0)
        if cls_index is not None:
            cur = (cur.unsqueeze(0) * cls_index).max(1)[0]
        score, pred = cur.max(0)
        pred = pred.clone()
        pred[score < model.prob_thd] = 0
        preds.append(pred)
    return torch.stack(preds, dim=0)


def _stack_pred_from_samples(data_samples) -> torch.Tensor:
    return torch.stack([ds.pred_sem_seg.data.squeeze(0) for ds in data_samples], dim=0)


@torch.no_grad()
def _forward_feature_with_layer_logits(
    model,
    img: torch.Tensor,
    logit_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Tuple[int, int]]:
    """Run model.forward_feature and capture per-layer token logits as side-channel.

    Returns:
        final_logits: [B,Q,H,W] exactly from model.forward_feature
        layer_logits_native: list [B,Q,I,J] per layer before upsampling
        layer_logits_up: list [B,Q,H,W] per layer after same upsampling rule as final
        token_size: native token grid (I,J)
    """
    if isinstance(img, list):
        img = img[0]

    token_size = _resolve_token_grid(model, (img.shape[-2], img.shape[-1]))
    clip_patch_h, clip_patch_w = model.clip.visual.patch_size
    clip_token_size = (img.shape[-2] // clip_patch_h, img.shape[-1] // clip_patch_w)
    cap: Dict[str, List[torch.Tensor]] = {}
    orig_encode_image = model.clip.encode_image

    def _wrapped_encode_image(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['return_layer_tokens'] = True
        final_tokens, layer_tokens = orig_encode_image(*args, **kwargs)
        cap['layer_tokens'] = layer_tokens
        return final_tokens

    model.clip.encode_image = _wrapped_encode_image
    try:
        final_logits = model.forward_feature(img, logit_size=logit_size)
    finally:
        model.clip.encode_image = orig_encode_image

    if 'layer_tokens' not in cap:
        raise RuntimeError('Failed to capture layer tokens from clip.encode_image.')

    layer_logits_native: List[torch.Tensor] = []
    layer_logits_up: List[torch.Tensor] = []
    out_hw = (int(final_logits.shape[-2]), int(final_logits.shape[-1]))

    for tokens in cap['layer_tokens']:
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        logits = tokens @ model.query_features.T
        
        n_tokens = tokens.shape[1]
        if n_tokens == token_size[0] * token_size[1]:
            native_size = token_size
        elif n_tokens == clip_token_size[0] * clip_token_size[1]:
            native_size = clip_token_size
        else:
            raise RuntimeError(
                f'Unexpected token count {n_tokens}; expected '
                f'{token_size[0] * token_size[1]} (native) or '
                f'{clip_token_size[0] * clip_token_size[1]} (clip).'
            )

        logits = logits.permute(0, 2, 1).reshape(tokens.shape[0], logits.shape[-1], native_size[0], native_size[1])
        layer_logits_native.append(logits)
        logits_up = F.interpolate(logits, size=out_hw, mode='bilinear')
        layer_logits_up.append(logits_up)

    return final_logits, layer_logits_native, layer_logits_up, token_size


@torch.no_grad()
def _forward_slide_with_layers(model, img: torch.Tensor, img_metas, stride=112, crop_size=224):
    """Mirror proxyclip_segmentor.forward_slide and collect per-layer logits.

    Returns:
        final_logits: [B,Q,H,W]
        layer_logits_ori: list [B,Q,H,W], each aligned to ori_shape
        layer_logits_native_map: list [B,Q,h_img,w_img], crop-stitched pre-ori-resize
        token_size_ref: token grid from first crop
    """
    if isinstance(img, list):
        img = img[0].unsqueeze(0)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = img.shape
    out_channels = model.num_queries
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

    layer_preds: Optional[List[torch.Tensor]] = None
    token_size_ref: Optional[Tuple[int, int]] = None

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]

            H, W = crop_img.shape[2:]
            pad = model.compute_padsize(H, W, 56)
            if any(pad):
                crop_img = F.pad(crop_img, pad)

            crop_seg_logit, _, crop_layer_logits_up, token_size = _forward_feature_with_layer_logits(model, crop_img)
            crop_seg_logit = crop_seg_logit.detach()
            crop_layer_logits_up = [x.detach() for x in crop_layer_logits_up]

            if token_size_ref is None:
                token_size_ref = token_size

            if any(pad):
                l, t = pad[0], pad[2]
                crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]
                crop_layer_logits_up = [x[:, :, t:t + H, l:l + W] for x in crop_layer_logits_up]

            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

            if layer_preds is None:
                layer_preds = [img.new_zeros((batch_size, out_channels, h_img, w_img)) for _ in crop_layer_logits_up]
            for i, lay in enumerate(crop_layer_logits_up):
                layer_preds[i] += F.pad(lay,
                                        (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0

    preds = preds / count_mat
    layer_preds = [x / count_mat for x in (layer_preds or [])]

    img_size = img_metas[0]['ori_shape'][:2]
    final_logits = F.interpolate(preds, size=img_size, mode='bilinear')
    layer_logits_ori = [F.interpolate(x, size=img_size, mode='bilinear') for x in layer_preds]

    if token_size_ref is None:
        token_size_ref = (0, 0)

    return final_logits, layer_logits_ori, layer_preds, token_size_ref


@torch.no_grad()
def _predict_with_layer_logits(model, inputs: torch.Tensor, data_samples):
    """Run eval-compatible inference and collect layer logits as side-channel."""
    if data_samples is not None:
        batch_img_metas = [ds.metainfo for ds in data_samples]
    else:
        batch_img_metas = [
            dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0],
            )
        ] * inputs.shape[0]

    if model.slide_crop > 0:
        final_logits, layer_logits, _, token_size = _forward_slide_with_layers(
            model,
            inputs,
            batch_img_metas,
            model.slide_stride,
            model.slide_crop,
        )
    else:
        final_logits, _, layer_logits, token_size = _forward_feature_with_layer_logits(
            model,
            inputs,
            logit_size=batch_img_metas[0]['ori_shape'],
        )

    final_pred = _seg_logits_to_pred(model, final_logits)
    layer_preds = [_seg_logits_to_pred(model, x) for x in layer_logits]
    return final_logits, final_pred, layer_logits, layer_preds, token_size


def _find_first_resize(pipeline: List[dict]) -> Optional[int]:
    for i, t in enumerate(pipeline):
        if isinstance(t, dict) and t.get('type') == 'Resize':
            return i
    return None


def _override_resize_scale(cfg: Config, scale: Tuple[int, int]) -> None:
    idx = _find_first_resize(cfg.test_dataloader.dataset.pipeline)
    if idx is None:
        return
    cfg.test_dataloader.dataset.pipeline[idx]['scale'] = scale
    cfg.test_dataloader.dataset.pipeline[idx]['keep_ratio'] = True


def _ensure_annotations_before_resize(cfg: Config) -> None:
    """Ensure GT is resized consistently with image.

    If the pipeline is ordered as LoadImage -> Resize -> LoadAnnotations,
    the annotation will not be resized, leading to misaligned GT and very poor metrics.
    We fix this for analysis by moving the first LoadAnnotations to be right before
    the first Resize.
    """
    pipe = cfg.test_dataloader.dataset.pipeline
    resize_i = _find_first_resize(pipe)
    if resize_i is None:
        return

    ann_i = None
    for i, t in enumerate(pipe):
        if isinstance(t, dict) and t.get('type') == 'LoadAnnotations':
            ann_i = i
            break
    if ann_i is None:
        return

    if ann_i > resize_i:
        ann = pipe.pop(ann_i)
        pipe.insert(resize_i, ann)


def _pool_gt_to_grid(
    gt: torch.Tensor,
    num_classes: int,
    grid_size: Tuple[int, int],
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample pixel GT to an arbitrary token grid using area (majority vote).

    This is robust when ProxyCLIP uses external VFM feats: the final token grid is then the VFM grid,
    not necessarily the CLIP patch grid.

    Args:
        gt: [B, H, W] int64
        grid_size: (I, J)
    Returns:
        gt_grid: [B, I, J] int64
        valid_grid: [B, I, J] bool (ignore fraction <= 0.5)
    """
    if gt.dtype != torch.long:
        gt = gt.long()

    ignore = gt == ignore_index
    gt_safe = gt.clone()
    gt_safe[ignore] = 0

    one_hot = F.one_hot(gt_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]
    pooled = F.interpolate(one_hot, size=grid_size, mode='area')
    gt_grid = pooled.argmax(dim=1)

    ignore_f = F.interpolate(ignore.float().unsqueeze(1), size=grid_size, mode='area').squeeze(1)
    valid_grid = ignore_f <= 0.5
    return gt_grid, valid_grid


def _confusion_update(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-class intersection and union for a single layer.

    pred, gt: [B,I,J]
    valid: [B,I,J]
    """
    pred_f = pred[valid].view(-1)
    gt_f = gt[valid].view(-1)
    if pred_f.numel() == 0:
        return (
            torch.zeros(num_classes, device=pred.device, dtype=torch.long),
            torch.zeros(num_classes, device=pred.device, dtype=torch.long),
        )

    # bincount on combined indices
    k = gt_f * num_classes + pred_f
    cm = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    inter = torch.diag(cm)
    gt_sum = cm.sum(dim=1)
    pred_sum = cm.sum(dim=0)
    union = gt_sum + pred_sum - inter
    return inter.to(torch.long), union.to(torch.long)


def _accumulate(
    agg: LayerAgg,
    pred_token_layers: List[torch.Tensor],
    pred_pixel_layers: List[torch.Tensor],
    gt_pixel: torch.Tensor,
    valid_pixel: torch.Tensor,
    gt_patch: torch.Tensor,
    valid_patch: torch.Tensor,
    ratio_threshold: float,
    ignore_first_stable: bool = True,
) -> LayerAgg:
    L = len(pred_token_layers)
    B, I, J = gt_patch.shape
    N = B * I * J

    gt_f = gt_patch.view(N)
    valid_f = valid_patch.view(N)

    correct_stack = []

    for l in range(L):
        # token-level correctness for convergence dynamics
        pred = pred_token_layers[l].view(N)
        correct = (pred == gt_f) & valid_f
        correct_stack.append(correct)

        # pixel-level metrics using standard segmentation-style predictions
        inter_px, union_px = _confusion_update(
            pred_pixel_layers[l],
            gt_pixel,
            valid_pixel,
            num_classes=agg.px_inter.shape[1],
        )
        agg.px_inter[l] += inter_px
        agg.px_union[l] += union_px

        px_correct = (pred_pixel_layers[l] == gt_pixel) & valid_pixel
        agg.px_correct[l] += px_correct.sum()
        agg.px_total[l] += valid_pixel.sum()

    # first / stable (token-level dynamics)
    correct_mat = torch.stack(correct_stack, dim=0)  # [L, N]
    valid_any = valid_f

    if ignore_first_stable:
        # invalid tokens -> always false, and excluded from hist
        pass

    first, converge = _first_and_converge_layers(
        correct_mat=correct_mat,
        valid=valid_any,
    )

    # ratio stability against final prediction and against GT
    final_pred = pred_token_layers[-1].view(N)
    final_correct_mat = torch.stack([(p.view(N) == final_pred) for p in pred_token_layers], dim=0)
    stable_ratio_final = _stable_layer_by_ratio(
        correct_mat=final_correct_mat,
        valid=valid_any,
        ratio_threshold=ratio_threshold,
    )

    stable_ratio_gt = _stable_layer_by_ratio(
        correct_mat=correct_mat,
        valid=valid_any,
        ratio_threshold=ratio_threshold,
    )

    for l in range(L):
        agg.first_hist[l] += (first == l).sum()
        agg.converge_hist[l] += (converge == l).sum()
        agg.stable_ratio_final_hist[l] += (stable_ratio_final == l).sum()
        agg.stable_ratio_gt_hist[l] += (stable_ratio_gt == l).sum()

    agg.valid_tokens += int(valid_any.sum().item())
    return agg


def _layer_miou(inter: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    # inter/union: [L, C]
    union_f = union.float()
    iou = inter.float() / torch.clamp(union_f, min=1.0)
    # IMPORTANT: ignore classes that do not appear in GT or prediction (union==0).
    # Otherwise, per-image mIoU is artificially suppressed when many classes are absent.
    valid_cls = union_f > 0
    miou = (iou * valid_cls.float()).sum(dim=1) / torch.clamp(valid_cls.float().sum(dim=1), min=1.0)
    return miou


def parse_args():
    p = argparse.ArgumentParser("Token-layer analysis for ProxyCLIP (segmentation)")
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--work-dir', type=str, default='./work_dirs/token_layer_analysis')
    p.add_argument('--max-samples', type=int, default=50)
    p.add_argument('--analysis-scale', type=int, nargs=2, default=None, metavar=('W', 'H'))
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--ignore-index', type=int, default=255)
    p.add_argument('--stable-ratio-threshold', type=float, default=0.9)
    p.add_argument('--strict-eval-compatible', type=int, choices=[0, 1], default=1)
    p.add_argument('--check-pred-match', type=int, choices=[0, 1], default=1)
    p.add_argument(
        '--save-convergence-maps',
        type=int,
        default=0,
        help='Save per-image convergence maps for the first N images (0 disables).',
    )
    return p.parse_args()


def main():
    args = parse_args()

    config_abs = os.path.abspath(args.config)
    work_dir_abs = os.path.abspath(args.work_dir)
    os.makedirs(work_dir_abs, exist_ok=True)

    # Many configs in this repo assume execution from the project root (e.g. './configs/..', './datasets/..').
    # Infer project root as the parent directory of a 'configs' folder.
    config_dir = os.path.dirname(config_abs)
    project_root = os.path.dirname(config_dir) if os.path.basename(config_dir) == 'configs' else os.path.dirname(config_abs)
    os.chdir(project_root)

    cfg = Config.fromfile(config_abs)
    cfg.work_dir = work_dir_abs

    # Optional experimental override: disabled by default to keep eval compatibility.
    strict_eval_compatible = bool(args.strict_eval_compatible)
    check_pred_match = bool(args.check_pred_match)
    stable_ratio_threshold = float(args.stable_ratio_threshold)
    if not (0.0 <= stable_ratio_threshold <= 1.0):
        raise ValueError('--stable-ratio-threshold must be in [0, 1].')

    if args.analysis_scale is not None and not strict_eval_compatible:
        _override_resize_scale(cfg, tuple(args.analysis_scale))

    if args.batch_size is not None:
        cfg.test_dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.test_dataloader.num_workers = args.num_workers

    runner = Runner.from_cfg(cfg)

    model = runner.model
    model.eval()

    # infer num_classes from model
    num_classes = int(model.num_classes)

    # infer number of layers from CLIP vision tower
    num_layers = len(model.clip.visual.transformer.resblocks)

    device = next(model.parameters()).device

    agg = LayerAgg(
        px_correct=torch.zeros(num_layers, device=device, dtype=torch.long),
        px_total=torch.zeros(num_layers, device=device, dtype=torch.long),
        px_inter=torch.zeros(num_layers, num_classes, device=device, dtype=torch.long),
        px_union=torch.zeros(num_layers, num_classes, device=device, dtype=torch.long),
        first_hist=torch.zeros(num_layers, device=device, dtype=torch.long),
        converge_hist=torch.zeros(num_layers, device=device, dtype=torch.long),
        stable_ratio_final_hist=torch.zeros(num_layers, device=device, dtype=torch.long),
        stable_ratio_gt_hist=torch.zeros(num_layers, device=device, dtype=torch.long),
        valid_tokens=0,
        pred_equal_pixels=0,
        pred_total_pixels=0,
    )

    dataloader = runner.test_dataloader

    seen = 0
    saved_maps = 0
    # record a representative token grid / prediction map size for debugging (first batch only)
    debug_token_grid: Optional[Tuple[int, int]] = None
    debug_pred_hw: Optional[Tuple[int, int]] = None
    for data_batch in dataloader:
        if args.max_samples is not None and seen >= args.max_samples:
            break

        # run preprocessor to obtain inputs / data_samples
        processed = model.data_preprocessor(data_batch, training=False)
        inputs = processed['inputs']
        data_samples = processed['data_samples']

        gt = torch.stack([ds.gt_sem_seg.data.squeeze(0) for ds in data_samples], dim=0).to(device)  # [B,H,W]

        # Reference prediction from the exact eval path.
        ref_samples = model.predict(inputs, data_samples)
        ref_pred = _stack_pred_from_samples(ref_samples)

        _, pred_eval, logits_layers, pred_pixel_layers, token_size = _predict_with_layer_logits(
            model,
            inputs,
            data_samples,
        )

        if check_pred_match:
            agg.pred_equal_pixels += int((pred_eval == ref_pred).sum().item())
            agg.pred_total_pixels += int(pred_eval.numel())

        if debug_token_grid is None:
            debug_token_grid = (int(token_size[0]), int(token_size[1]))
        if debug_pred_hw is None:
            debug_pred_hw = (int(gt.shape[-2]), int(gt.shape[-1]))

        gt_pixel = gt
        valid_pixel = gt_pixel != args.ignore_index

        pred_token_layers = [F.interpolate(x.unsqueeze(1).float(), size=token_size, mode='nearest').squeeze(1).long()
                             for x in pred_pixel_layers]

        gt_patch, valid_patch = _pool_gt_to_grid(
            gt,
            num_classes=num_classes,
            grid_size=token_size,
            ignore_index=args.ignore_index,
        )

        agg = _accumulate(
            agg,
            pred_token_layers,
            pred_pixel_layers,
            gt_pixel,
            valid_pixel,
            gt_patch,
            valid_patch,
            ratio_threshold=stable_ratio_threshold,
        )

        # optional: save per-image convergence maps
        if args.save_convergence_maps and saved_maps < int(args.save_convergence_maps):
            L = len(pred_token_layers)
            B, I, J = gt_patch.shape
            # [L,B,I,J]
            preds_stack = torch.stack(pred_token_layers, dim=0)
            correct_layers = (preds_stack == gt_patch.unsqueeze(0)) & valid_patch.unsqueeze(0)
            correct_flat = correct_layers.view(L, -1)
            valid_flat = valid_patch.view(-1)
            first, converge = _first_and_converge_layers(
                correct_mat=correct_flat,
                valid=valid_flat,
            )
            first_map = first.view(B, I, J)
            converge_map = converge.view(B, I, J)

            for b in range(B):
                if saved_maps >= int(args.save_convergence_maps):
                    break
                img_path = data_samples[b].metainfo.get('img_path', '') if b < len(data_samples) else ''
                base = os.path.splitext(os.path.basename(img_path))[0] if img_path else f'sample_{seen + b:06d}'
                out_path = os.path.join(work_dir_abs, f'{base}_convergence_maps.npz')
                np.savez_compressed(
                    out_path,
                    first_correct_layer_map=first_map[b].detach().cpu().numpy().astype(np.int16),
                    convergence_layer_map=converge_map[b].detach().cpu().numpy().astype(np.int16),
                    gt_patch=gt_patch[b].detach().cpu().numpy().astype(np.int16),
                    valid_patch=valid_patch[b].detach().cpu().numpy().astype(np.uint8),
                    final_pred=preds_stack[-1, b].detach().cpu().numpy().astype(np.int16),
                )
                saved_maps += 1

        seen += gt.shape[0]

    pixel_acc = (agg.px_correct.float() / torch.clamp(agg.px_total.float(), min=1.0)).cpu().numpy()
    pixel_miou = _layer_miou(agg.px_inter, agg.px_union).cpu().numpy()

    first = agg.first_hist.cpu().numpy()
    converge_new = agg.converge_hist.cpu().numpy()
    stable_ratio_final_new = agg.stable_ratio_final_hist.cpu().numpy()
    stable_ratio_gt_new = agg.stable_ratio_gt_hist.cpu().numpy()

    # cumulative proportions
    valid_tokens = max(int(agg.valid_tokens), 1)
    p_first = np.cumsum(first) / valid_tokens
    p_converge = np.cumsum(converge_new) / valid_tokens
    p_stable_ratio_final = np.cumsum(stable_ratio_final_new) / valid_tokens
    p_stable_ratio_gt = np.cumsum(stable_ratio_gt_new) / valid_tokens

    out = {
        'config': args.config,
        'num_layers': int(num_layers),
        'num_classes': int(num_classes),
        'max_samples': int(args.max_samples),
        'analysis_scale': args.analysis_scale,
        'stable_ratio_threshold': stable_ratio_threshold,
        'strict_eval_compatible': strict_eval_compatible,
        'check_pred_match': check_pred_match,
        'debug_token_grid': list(debug_token_grid) if debug_token_grid is not None else None,
        'debug_pred_hw': list(debug_pred_hw) if debug_pred_hw is not None else None,
        'pixel_acc': pixel_acc.tolist(),
        'pixel_miou': pixel_miou.tolist(),
        'first_hist': first.tolist(),
        'converge_new': converge_new.tolist(),
        'stable_ratio_final_new': stable_ratio_final_new.tolist(),
        'stable_ratio_gt_new': stable_ratio_gt_new.tolist(),
        'p_first': p_first.tolist(),
        'p_converge': p_converge.tolist(),
        'p_stable_ratio_final': p_stable_ratio_final.tolist(),
        'p_stable_ratio_gt': p_stable_ratio_gt.tolist(),
        'valid_tokens': int(valid_tokens),
        'pred_match': {
            'equal_pixels': int(agg.pred_equal_pixels),
            'total_pixels': int(agg.pred_total_pixels),
            'ratio': float(agg.pred_equal_pixels / max(agg.pred_total_pixels, 1)),
        },
    }

    with open(os.path.join(work_dir_abs, 'layer_token_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # quick plots
    try:
        import matplotlib.pyplot as plt

        xs = np.arange(1, num_layers + 1)

        plt.figure()
        plt.plot(xs, pixel_acc)
        plt.xlabel('Layer')
        plt.ylabel('Accuracy (pixel)')
        plt.title('Layer-wise Pixel Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_abs, 'layer_vs_pixel_accuracy.png'), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(xs, pixel_miou)
        plt.xlabel('Layer')
        plt.ylabel('mIoU (pixel)')
        plt.title('Layer-wise Pixel mIoU')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_abs, 'layer_vs_pixel_miou.png'), dpi=200)
        plt.close()

        # Cumulative convergence curve
        plt.figure()
        plt.plot(xs, p_converge, label='P_converged')
        plt.xlabel('Layer')
        plt.ylabel('Converged patch ratio (cumulative)')
        plt.title('Cumulative Convergence (strict)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_abs, 'convergence_cumulative.png'), dpi=200)
        plt.close()

        # New converged patches per layer
        plt.figure()
        plt.bar(xs, converge_new)
        plt.xlabel('Layer')
        plt.ylabel('New converged patches')
        plt.title('Newly Converged Patches per Layer')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_abs, 'convergence_new_per_layer.png'), dpi=200)
        plt.close()

        # Cumulative stability ratio curves
        plt.figure()
        plt.plot(xs, p_stable_ratio_final, label='P_stable_ratio_final')
        plt.plot(xs, p_stable_ratio_gt, label='P_stable_ratio_gt')
        plt.xlabel('Layer')
        plt.ylabel('Stable patch ratio (cumulative)')
        plt.title(f'Cumulative Stability (ratio >= {stable_ratio_threshold:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_abs, 'stability_ratio_cumulative.png'), dpi=200)
        plt.close()

        # Newly stable patches per layer (ratio criterion)
        plt.figure()
        plt.bar(xs - 0.2, stable_ratio_final_new, width=0.4, label='final')
        plt.bar(xs + 0.2, stable_ratio_gt_new, width=0.4, label='gt')
        plt.xlabel('Layer')
        plt.ylabel('Newly stable patches')
        plt.title(f'Newly Stable Patches (ratio >= {stable_ratio_threshold:.2f})')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir_abs, 'stability_ratio_new_per_layer.png'), dpi=200)
        plt.close()

    except Exception as e:  # pragma: no cover
        with open(os.path.join(work_dir_abs, 'plot_error.txt'), 'w') as f:
            f.write(repr(e))


if __name__ == '__main__':
    main()
