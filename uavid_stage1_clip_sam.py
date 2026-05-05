import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from proxyclip_segmentor import ProxyCLIPSegmentation
from segment_anything import SamPredictor, sam_model_registry


UAVID_CLASSES = [
    "background",
    "building",
    "road",
    "car",
    "tree",
    "vegetation",
    "human",
]

# Same palette as UAVidDataset.METAINFO in custom_datasets.py (RGB)
UAVID_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "background": (0, 0, 0),
    "building": (128, 0, 0),
    "road": (128, 64, 128),
    "car": (192, 0, 192),
    "tree": (0, 128, 0),
    "vegetation": (128, 128, 0),
    "human": (64, 64, 0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("UAVid stage1: CLIP similarity -> SAM prompts -> initial instance masks")
    p.add_argument(
        "--img_dir",
        type=str,
        default="/home/dongyuxiang/ProxyCLIP/datasets/UAVid/img_dir/1080_2160_1280_2560/seq21",
        help="Directory containing 1280x1080 UAVid tiles.",
    )
    p.add_argument("--out_dir", type=str, default="/home/dongyuxiang/ProxyCLIP/outputs/uavid_stage1")
    p.add_argument("--sam_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument(
        "--sam_ckpt",
        type=str,
        default="",
        help="Path to SAM checkpoint (e.g. sam_vit_b_01ec64.pth). If empty, only activation overlays are generated.",
    )

    p.add_argument("--clip_type", type=str, default="openai")
    p.add_argument("--model_type", type=str, default="ViT-L/14")
    p.add_argument("--vfm_model", type=str, default="dino", choices=["sam", "mae", "dino", "dinov2"])

    p.add_argument("--beta", type=float, default=1.2)
    p.add_argument("--gamma", type=float, default=3.0)

    p.add_argument(
        "--clip_long_side",
        type=int,
        default=672,
        help="Resize the tile for CLIP/VFM feature extraction by setting the long side to this value (0 = disable). Helps avoid OOM.",
    )
    p.add_argument(
        "--pad_h",
        type=int,
        default=0,
        help="Pad CLIP input height to this value (0 = auto pad to nearest multiple of patch size).",
    )
    p.add_argument(
        "--pad_w",
        type=int,
        default=0,
        help="Pad CLIP input width to this value (0 = auto pad to nearest multiple of patch size).",
    )
    p.add_argument("--max_instances_per_class", type=int, default=100)
    p.add_argument("--min_area", type=int, default=40)
    p.add_argument("--quantiles", type=str, default="0.99,0.97,0.95")

    p.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha")
    p.add_argument(
        "--no_draw_contour",
        action="store_true",
        help="Disable drawing 1px contours for instance masks",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 for CLIP/VFM forward (SAM kept in fp32 for stability).",
    )

    p.add_argument(
        "--save_masks",
        action="store_true",
        help="Save raw instance masks as instance-id map + class map (+ meta).",
    )

    p.add_argument("--limit", type=int, default=0, help="Only process first N images (0 = all)")
    return p.parse_args()


def list_images(img_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    return files


def read_rgb_uint8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def pad_to_height_rgb(image_rgb: np.ndarray, target_h: int) -> Tuple[np.ndarray, int]:
    h, w, _ = image_rgb.shape
    if h >= target_h:
        return image_rgb, 0
    pad = target_h - h
    padded = cv2.copyMakeBorder(image_rgb, 0, pad, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, pad


def resize_long_side_rgb(image_rgb: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    if long_side <= 0:
        return image_rgb, 1.0
    h, w = image_rgb.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return image_rgb, 1.0
    scale = long_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def pad_to_hw_rgb(image_rgb: np.ndarray, target_h: int, target_w: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image_rgb.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return image_rgb, (0, 0)
    padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (pad_h, pad_w)


def clip_preprocess_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    # ToTensor -> float32 [0,1] and CHW
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    x = t(image_rgb).unsqueeze(0).to(device)
    return x


@torch.no_grad()
def compute_similarity_maps(
    proxy_model: ProxyCLIPSegmentation,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """Returns per-class similarity maps (B, Q, H, W)."""
    maps = proxy_model.forward_feature(image_tensor)
    return maps


def normalize_heatmap(heat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    hmin = float(np.min(heat))
    hmax = float(np.max(heat))
    if hmax - hmin < eps:
        return np.zeros_like(heat, dtype=np.float32)
    out = (heat - hmin) / (hmax - hmin)
    return out.astype(np.float32)


def heatmap_to_overlay_rgb(
    image_rgb: np.ndarray, heat01: np.ndarray, color_rgb: Tuple[int, int, int], alpha: float
) -> np.ndarray:
    heat01 = np.clip(heat01, 0.0, 1.0)
    color = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    base = image_rgb.astype(np.float32)
    heat_rgb = heat01[..., None] * color
    # Standard alpha blend: out = base*(1-a) + heat*a
    out = base * (1.0 - alpha) + heat_rgb * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def connected_components(binary: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    # binary: HxW bool/0-1
    binary_u8 = (binary.astype(np.uint8) * 255)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    return num, labels, stats, centroids


def component_score(heat: np.ndarray, labels: np.ndarray, comp_id: int) -> float:
    vals = heat[labels == comp_id]
    if vals.size == 0:
        return -1e9
    return float(vals.max())


def sample_points_in_component(
    heat: np.ndarray,
    labels: np.ndarray,
    comp_id: int,
    num_pos: int = 3,
    num_neg: int = 3,
    bbox: Tuple[int, int, int, int] | None = None,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (point_coords Nx2 in xy, point_labels N)."""
    if rng is None:
        rng = np.random.default_rng(0)

    ys, xs = np.where(labels == comp_id)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    vals = heat[ys, xs]
    order = np.argsort(-vals)
    xs = xs[order]
    ys = ys[order]

    # Greedy farthest-like selection for positive points
    pos = []
    for x, y in zip(xs, ys):
        if len(pos) == 0:
            pos.append((x, y))
        else:
            d2 = [float((x - px) ** 2 + (y - py) ** 2) for px, py in pos]
            if min(d2) >= 64.0:  # >= 8px apart
                pos.append((x, y))
        if len(pos) >= num_pos:
            break
    if len(pos) == 0:
        pos.append((int(xs[0]), int(ys[0])))

    # Negative points: sample outside component but inside bbox if provided
    if bbox is None:
        x1, y1, x2, y2 = 0, 0, heat.shape[1] - 1, heat.shape[0] - 1
    else:
        x1, y1, x2, y2 = bbox

    neg = []
    tries = 0
    while len(neg) < num_neg and tries < 500:
        tries += 1
        x = int(rng.integers(x1, x2 + 1))
        y = int(rng.integers(y1, y2 + 1))
        if labels[y, x] == comp_id:
            continue
        # prefer low-heat negatives
        if heat[y, x] > 0.4:
            continue
        neg.append((x, y))

    coords = np.array(pos + neg, dtype=np.float32)
    lab = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return coords, lab


def bbox_from_stats(stats_row: np.ndarray, margin: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh, _area = [int(v) for v in stats_row]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w - 1, x + bw - 1 + margin)
    y2 = min(h - 1, y + bh - 1 + margin)
    return x1, y1, x2, y2


def build_sam_mask_input_from_heatmap(
    heat01: np.ndarray,
    predictor: SamPredictor,
    eps: float = 1e-4,
) -> np.ndarray:
    """Convert HxW heat in [0,1] to SAM mask_input (1,256,256) logits aligned with predictor."""
    h, w = heat01.shape
    # Resize to predictor input (ResizeLongestSide) and then pad to square as Sam.preprocess does.
    new_h, new_w = predictor.transform.get_preprocess_shape(h, w, predictor.model.image_encoder.img_size)

    resized = cv2.resize(heat01, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((predictor.model.image_encoder.img_size, predictor.model.image_encoder.img_size), dtype=np.float32)
    padded[:new_h, :new_w] = resized

    lowres = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_LINEAR)

    p = np.clip(lowres, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p)).astype(np.float32)
    return logit[None, :, :]


def mask_score_by_heat(mask: np.ndarray, heat01: np.ndarray) -> float:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    vals = heat01[mask]
    if vals.size == 0:
        return -1e9
    return float(vals.mean())


def overlay_instances_rgb(
    image_rgb: np.ndarray,
    instances: List[Dict],
    alpha: float,
    draw_contour: bool,
) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)

    for inst in instances:
        mask = inst["mask"].astype(bool)
        color = np.array(inst["color_rgb"], dtype=np.float32)
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha

        if draw_contour:
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color=(255, 255, 255), thickness=1)

    return np.clip(out, 0, 255).astype(np.uint8)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    images = list_images(img_dir)
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    # Build ProxyCLIP model (only for similarity maps)
    proxy = ProxyCLIPSegmentation(
        clip_type=args.clip_type,
        model_type=args.model_type,
        vfm_model=args.vfm_model,
        name_path=str(Path(__file__).parent / "configs" / "cls_uavid.txt"),
        checkpoint=None,
        device=device,
        beta=args.beta,
        gamma=args.gamma,
        slide_stride=0,
        slide_crop=0,
    )

    # Build SAM predictor (optional)
    predictor = None
    sam_ckpt = (args.sam_ckpt or "").strip()
    if sam_ckpt:
        sam = sam_model_registry[args.sam_type](checkpoint=sam_ckpt)
        sam.eval().to(device)
        predictor = SamPredictor(sam)

    quantiles = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]

    clip_patch = proxy.clip.visual.patch_size
    if isinstance(clip_patch, int):
        clip_patch_h = clip_patch_w = int(clip_patch)
    else:
        clip_patch_h, clip_patch_w = int(clip_patch[0]), int(clip_patch[1])

    for idx, img_path in enumerate(images):
        rgb = read_rgb_uint8(img_path)
        h0, w0 = rgb.shape[:2]
        if (h0, w0) != (1080, 1280):
            # We still support other shapes but assume UAVid tiles are 1080x1280
            pass

        # Compute similarity maps on a potentially smaller resolution to avoid OOM.
        rgb_clip, scale = resize_long_side_rgb(rgb, args.clip_long_side)
        hc, wc = rgb_clip.shape[:2]

        # Pad to user targets or to nearest multiple of CLIP patch size
        target_h = args.pad_h if args.pad_h > 0 else ceil_to_multiple(hc, clip_patch_h)
        target_w = args.pad_w if args.pad_w > 0 else ceil_to_multiple(wc, clip_patch_w)
        rgb_clip_pad, (pad_h, pad_w) = pad_to_hw_rgb(rgb_clip, target_h, target_w)

        x = clip_preprocess_tensor(rgb_clip_pad, device)
        if args.fp16:
            x = x.half()

        sim_maps = compute_similarity_maps(proxy, x)[0]  # (Q, Hc_pad, Wc_pad)

        # Remove CLIP padding, then upscale back to original tile size
        sim_maps = sim_maps[:, :hc, :wc].float().detach().cpu()
        sim_maps = F.interpolate(sim_maps.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False).squeeze(0)
        sim_maps = sim_maps.numpy()

        # Convert to per-class heat01 for visualization & prompting
        # Use per-pixel softmax across classes (keeps relative semantics)
        sim_t = torch.from_numpy(sim_maps)
        prob = F.softmax(sim_t, dim=0).numpy().astype(np.float32)

        # Output folders
        stem = img_path.stem
        sample_out = out_dir / img_dir.name / stem
        sample_out.mkdir(parents=True, exist_ok=True)

        # Save activation overlays
        for ci, cname in enumerate(UAVID_CLASSES):
            heat01 = normalize_heatmap(prob[ci])
            overlay = heatmap_to_overlay_rgb(rgb, heat01, UAVID_PALETTE[cname], alpha=args.alpha)
            save_rgb(sample_out / f"activation_overlay_{ci:02d}_{cname}.png", overlay)

        if predictor is None:
            print(f"[{idx+1}/{len(images)}] {img_path.name}: activation overlays only (no --sam_ckpt) -> {sample_out}")
            continue

        # SAM: set image once
        predictor.set_image(rgb)

        instances: List[Dict] = []

        # For each foreground class, propose instances and run SAM
        for ci, cname in enumerate(UAVID_CLASSES):
            if cname == "background":
                continue

            heat01 = normalize_heatmap(prob[ci])

            # Build candidates from multiple quantiles
            candidates = []
            for q in quantiles:
                thr = float(np.quantile(heat01, q))
                binary = heat01 >= thr
                num, labels, stats, _ = connected_components(binary)
                for comp_id in range(1, num):
                    area = int(stats[comp_id, cv2.CC_STAT_AREA])
                    if area < args.min_area:
                        continue
                    s = component_score(heat01, labels, comp_id)
                    candidates.append((s, labels.copy(), stats[comp_id].copy(), comp_id))

            # Sort and keep top-K
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[: args.max_instances_per_class]

            rng = np.random.default_rng(12345 + ci)

            for rank, (_s, labels, stats_row, comp_id) in enumerate(candidates):
                x1, y1, x2, y2 = bbox_from_stats(stats_row, margin=8, w=w0, h=h0)
                box = np.array([x1, y1, x2, y2], dtype=np.float32)

                pts, pls = sample_points_in_component(
                    heat01,
                    labels,
                    comp_id,
                    num_pos=3,
                    num_neg=3,
                    bbox=(x1, y1, x2, y2),
                    rng=rng,
                )

                # Mask prior: restrict heatmap to this component for focus
                prior = np.zeros_like(heat01, dtype=np.float32)
                prior[labels == comp_id] = heat01[labels == comp_id]
                mask_input = build_sam_mask_input_from_heatmap(prior, predictor)

                masks, ious, _low = predictor.predict(
                    point_coords=pts if len(pts) else None,
                    point_labels=pls if len(pls) else None,
                    box=box,
                    mask_input=mask_input,
                    multimask_output=True,
                    return_logits=False,
                )

                # Choose best mask by heat consistency
                best_k = int(np.argmax([mask_score_by_heat(m, heat01) for m in masks]))
                best_mask = masks[best_k].astype(bool)
                best_score = float(mask_score_by_heat(best_mask, heat01))
                best_area = int(best_mask.sum())

                instances.append(
                    {
                        "class_id": ci,
                        "class_name": cname,
                        "mask": best_mask,
                        "score": best_score,
                        "area": best_area,
                        "box": (x1, y1, x2, y2),
                        "color_rgb": UAVID_PALETTE[cname],
                    }
                )

        # Save instance overlay
        overlay_mask = overlay_instances_rgb(rgb, instances, alpha=args.alpha, draw_contour=not args.no_draw_contour)
        save_rgb(sample_out / "instances_overlay.png", overlay_mask)

        # Optionally dump raw masks in a compact form
        if args.save_masks:
            order = sorted(range(len(instances)), key=lambda i: float(instances[i]["score"]), reverse=True)
            id_map = np.zeros((h0, w0), dtype=np.int32)
            class_map = np.zeros((h0, w0), dtype=np.uint8)  # default background=0

            meta = {
                "image": img_path.name,
                "height": int(h0),
                "width": int(w0),
                "num_instances": int(len(instances)),
                "instances": [],
            }

            next_id = 1
            for i in order:
                inst = instances[i]
                m = inst["mask"]
                if not np.any(m):
                    continue
                write = (id_map == 0) & m
                if not np.any(write):
                    continue
                id_map[write] = next_id
                class_map[write] = np.uint8(inst["class_id"])
                meta["instances"].append(
                    {
                        "instance_id": int(next_id),
                        "class_id": int(inst["class_id"]),
                        "class_name": str(inst["class_name"]),
                        "score": float(inst["score"]),
                        "area": int(inst.get("area", int(m.sum()))),
                        "box": [int(v) for v in inst["box"]],
                    }
                )
                next_id += 1

            np.save(sample_out / "instances_id_map.npy", id_map)
            np.save(sample_out / "instances_class_map.npy", class_map)

            if next_id - 1 <= 65535:
                cv2.imwrite(str(sample_out / "instances_id_map.png"), id_map.astype(np.uint16))
            cv2.imwrite(str(sample_out / "instances_class_map.png"), class_map)

            with open(sample_out / "instances_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        # Also dump a lightweight summary
        with open(sample_out / "instances.txt", "w", encoding="utf-8") as f:
            for inst in instances:
                f.write(f"{inst['class_id']}\t{inst['class_name']}\t{inst['score']:.4f}\t{inst['box']}\n")

        print(f"[{idx+1}/{len(images)}] {img_path.name}: {len(instances)} instances -> {sample_out}")


if __name__ == "__main__":
    main()
