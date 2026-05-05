import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import custom_datasets  
import proxyclip_segmentor  
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='CLIP ViT token semantic analysis')
	parser.add_argument('--config', required=True, help='Path to config file')
	parser.add_argument('--work-dir', required=True, help='Directory to save analysis outputs')
	parser.add_argument(
		'--layer-indices',
		default='',
		help='Comma-separated layer indices, empty means all layers (e.g., 0,3,6,11)',
	)
	parser.add_argument(
		'--no-cosine-heatmap',
		action='store_true',
		help='Disable saving layer-to-layer cosine similarity heatmap',
	)
	parser.add_argument(
		'--raw-cosine',
		choices=['yes', 'no'],
		default='no',
		help='yes: also compute cosine matrix from raw tokens and ln-only tokens (ln_post without proj); no: projected cosine only',
	)
	parser.add_argument(
		'--max-samples',
		type=int,
		default=0,
		help='Process at most this many samples, 0 means all samples',
	)
	parser.add_argument('--device', default='cuda', help='Device to run model on')
	return parser.parse_args()


def load_cfg_and_prepare(args: argparse.Namespace) -> Tuple[Config, str]:
	cfg = Config.fromfile(args.config)
	cfg.work_dir = args.work_dir
	output_dir = os.path.join(args.work_dir, 'token_analysis')
	os.makedirs(output_dir, exist_ok=True)
	return cfg, output_dir


def build_runner_and_dataloader(cfg: Config, device: str):
	runner = Runner.from_cfg(cfg)
	model = runner.model
	model.eval()
	if device:
		model.to(device)
	dataloader = runner.test_dataloader
	return model, dataloader


def resolve_layer_indices(model, layer_indices_arg: str) -> List[int]:
	num_layers = len(model.clip.visual.transformer.resblocks)
	if not layer_indices_arg.strip():
		return list(range(num_layers))

	indices = []
	for part in layer_indices_arg.split(','):
		p = part.strip()
		if not p:
			continue
		idx = int(p)
		if idx < 0 or idx >= num_layers:
			raise ValueError(f'Invalid layer index {idx}, expected in [0, {num_layers - 1}]')
		indices.append(idx)

	if not indices:
		raise ValueError('No valid layer index found in --layer-indices')
	return sorted(set(indices))


def _extract_gt_from_data_samples(data_samples) -> Optional[torch.Tensor]:
	if data_samples is None:
		return None
	gts = []
	for ds in data_samples:
		if not hasattr(ds, 'gt_sem_seg'):
			return None
		gt = ds.gt_sem_seg.data
		if gt.ndim == 3 and gt.shape[0] == 1:
			gt = gt[0]
		gts.append(gt)
	if not gts:
		return None
	return torch.stack(gts, dim=0)


def _build_external_feats(model, img: torch.Tensor) -> Optional[torch.Tensor]:
	imgs_norm = [model.norm(model.unnorm(img[i])) for i in range(len(img))]
	imgs_norm = torch.stack(imgs_norm, dim=0).half()

	if model.vfm_model == 'sam':
		imgs_norm = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
		ex_feats = model.vfm.image_encoder(imgs_norm)
		return ex_feats

	if model.vfm_model == 'dino':
		feat = model.vfm.get_intermediate_layers(imgs_norm)[0]
		patch_size = model.vfm.patch_embed.patch_size
		if isinstance(patch_size, tuple):
			ph, pw = patch_size
		else:
			ph, pw = patch_size, patch_size
		h_tokens = imgs_norm.shape[-2] // ph
		w_tokens = imgs_norm.shape[-1] // pw
		ex_feats = feat[:, 1:, :].reshape(feat.shape[0], h_tokens, w_tokens, -1).permute(0, 3, 1, 2)
		return ex_feats

	if model.vfm_model == 'dinov2':
		ex_feats = model.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]
		return ex_feats

	if model.vfm_model == 'mae':
		patch_size = model.vfm.patch_embed.patch_size
		imgs_norm = F.interpolate(
			imgs_norm,
			size=(model.slide_crop, model.slide_crop),
			mode='bilinear',
			align_corners=False,
		)
		h_tokens = imgs_norm.shape[-2] // patch_size[0]
		w_tokens = imgs_norm.shape[-1] // patch_size[1]
		image_feat = model.vfm.forward_features(imgs_norm)
		ex_feats = image_feat.reshape(image_feat.shape[0], h_tokens, w_tokens, -1).permute(0, 3, 1, 2)
		return ex_feats

	return None


@torch.no_grad()
def extract_layer_tokens(
	model,
	images: torch.Tensor,
	layer_indices: Sequence[int],
	collect_raw_cosine: bool = False,
):
	ex_feats = _build_external_feats(model, images)
	encode_out = model.clip.encode_image(
		images.half(),
		external_feats=ex_feats,
		beta=model.beta,
		gamma=model.gamma,
		return_layer_tokens=True,
		layer_indices=layer_indices,
		layer_token_mode='all' if collect_raw_cosine else 'proj',
	)

	if not isinstance(encode_out, tuple):
		raise RuntimeError('Expected tuple output from encode_image when return_layer_tokens=True')

	if collect_raw_cosine:
		if len(encode_out) != 4:
			raise RuntimeError(
				'Expected 4 outputs (final, projected_tokens, raw_tokens, ln_tokens) when raw cosine is enabled'
			)
		_, layer_tokens, layer_tokens_raw, layer_tokens_ln = encode_out
	else:
		if len(encode_out) < 2:
			raise RuntimeError('Expected at least 2 outputs (final, layer_tokens) from encode_image')
		_, layer_tokens = encode_out[:2]
		layer_tokens_raw = None
		layer_tokens_ln = None

	patch_size = model.clip.visual.patch_size
	if isinstance(patch_size, tuple):
		ph, pw = patch_size
	else:
		ph, pw = patch_size, patch_size
	h_tokens = images.shape[-2] // ph
	w_tokens = images.shape[-1] // pw
	return layer_tokens, layer_tokens_raw, layer_tokens_ln, (h_tokens, w_tokens)


def _query_logits_to_class_logits(query_logits: torch.Tensor, query_idx: torch.Tensor, num_classes: int):
	num_queries = query_logits.shape[-1]
	if num_queries == num_classes:
		return query_logits

	class_logits = []
	for c in range(num_classes):
		q_mask = (query_idx == c)
		if torch.any(q_mask):
			class_logits.append(query_logits[..., q_mask].max(dim=-1).values)
		else:
			class_logits.append(torch.full_like(query_logits[..., 0], -1e6))
	return torch.stack(class_logits, dim=-1)


def _infer_hw_by_aspect(n_tokens: int, ref_h: int, ref_w: int) -> Tuple[int, int]:
	# Infer a factorized grid h*w=n_tokens whose aspect is closest to reference.
	target = ref_h / max(ref_w, 1)
	best_h, best_w = 1, n_tokens
	best_diff = abs((best_h / best_w) - target)
	limit = int(np.sqrt(n_tokens)) + 1
	for h in range(1, limit):
		if n_tokens % h != 0:
			continue
		w = n_tokens // h
		diff = abs((h / w) - target)
		if diff < best_diff:
			best_h, best_w, best_diff = h, w, diff
	return best_h, best_w


def _align_tokens_to_ref_grid(tokens: torch.Tensor, ref_grid: Tuple[int, int]) -> torch.Tensor:
	# Align [B, N, D] tokens from different layers to the same reference grid.
	ref_h, ref_w = ref_grid
	ref_n = ref_h * ref_w
	bsz, n_tokens, dim = tokens.shape
	if n_tokens == ref_n:
		return tokens

	h, w = _infer_hw_by_aspect(n_tokens, ref_h, ref_w)
	if h * w != n_tokens:
		raise RuntimeError(f'Cannot infer token grid for N={n_tokens}')

	feat = tokens.reshape(bsz, h, w, dim).permute(0, 3, 1, 2).contiguous()
	feat = F.interpolate(feat, size=(ref_h, ref_w), mode='bilinear', align_corners=False)
	return feat.permute(0, 2, 3, 1).reshape(bsz, ref_n, dim).contiguous()


@torch.no_grad()
def layer_token_to_pred(
	layer_tokens: Sequence[torch.Tensor],
	query_features: torch.Tensor,
	query_idx: torch.Tensor,
	num_classes: int,
	ref_token_grid: Tuple[int, int],
) -> torch.Tensor:
	preds = []
	qf = F.normalize(query_features, dim=-1)
	for tokens in layer_tokens:
		tokens = _align_tokens_to_ref_grid(tokens, ref_token_grid)
		tokens = F.normalize(tokens, dim=-1)
		logits_q = tokens @ qf.T
		logits_c = _query_logits_to_class_logits(logits_q, query_idx, num_classes)
		pred = torch.argmax(logits_c, dim=-1)
		preds.append(pred)
	return torch.stack(preds, dim=0)


def _majority_label_in_region(region: torch.Tensor, num_classes: int, ignore_index: int) -> int:
	region = region.reshape(-1)
	valid = (region >= 0) & (region < num_classes) & (region != ignore_index)
	if not torch.any(valid):
		return ignore_index
	counts = torch.bincount(region[valid].to(torch.long), minlength=num_classes)
	return int(torch.argmax(counts).item())


def build_token_gt(
	seg_label: torch.Tensor,
	token_grid: Tuple[int, int],
	num_classes: int,
	ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
	bsz, height, width = seg_label.shape
	h_tokens, w_tokens = token_grid

	token_gt = torch.full((bsz, h_tokens * w_tokens), ignore_index, dtype=torch.long, device=seg_label.device)
	valid_mask = torch.zeros((bsz, h_tokens * w_tokens), dtype=torch.bool, device=seg_label.device)

	y_edges = [int(np.floor(i * height / h_tokens)) for i in range(h_tokens + 1)]
	x_edges = [int(np.floor(i * width / w_tokens)) for i in range(w_tokens + 1)]

	for b in range(bsz):
		flat_idx = 0
		for i in range(h_tokens):
			y0, y1 = y_edges[i], y_edges[i + 1]
			y1 = max(y1, y0 + 1)
			for j in range(w_tokens):
				x0, x1 = x_edges[j], x_edges[j + 1]
				x1 = max(x1, x0 + 1)
				label = _majority_label_in_region(seg_label[b, y0:y1, x0:x1], num_classes, ignore_index)
				token_gt[b, flat_idx] = label
				valid_mask[b, flat_idx] = label != ignore_index
				flat_idx += 1

	return token_gt, valid_mask


def flatten_valid(
	pred_by_layer: torch.Tensor,
	token_gt: torch.Tensor,
	valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	valid = valid_mask.reshape(-1)
	if not torch.any(valid):
		return pred_by_layer.new_empty((pred_by_layer.shape[0], 0), dtype=torch.long), token_gt.new_empty((0,), dtype=torch.long)
	pred_valid = pred_by_layer.reshape(pred_by_layer.shape[0], -1)[:, valid]
	gt_valid = token_gt.reshape(-1)[valid]
	return pred_valid, gt_valid


def compute_layer_accuracy_counts(pred_valid: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
	correct = pred_valid.eq(gt_valid.unsqueeze(0))
	return correct.sum(dim=1).to(torch.long)


def compute_adjacent_consistency_counts(pred_valid: torch.Tensor) -> torch.Tensor:
	if pred_valid.shape[0] <= 1:
		return torch.zeros((0,), dtype=torch.long, device=pred_valid.device)
	same_adj = pred_valid[:-1].eq(pred_valid[1:])
	return same_adj.sum(dim=1).to(torch.long)


def accumulate_layer_cosine_sums(
	layer_tokens: Sequence[torch.Tensor],
	ref_token_grid: Tuple[int, int],
) -> Tuple[torch.Tensor, int, int, int]:
	aligned = []
	total_nan_count = 0
	total_zero_norm_count = 0
	for tokens in layer_tokens:
		tokens = _align_tokens_to_ref_grid(tokens, ref_token_grid)
		tokens = tokens.float()
		norms = tokens.norm(dim=-1, keepdim=True)
		total_zero_norm_count += int((norms <= 1e-12).sum().item())
		tokens = tokens / norms.clamp_min(1e-6)
		total_nan_count += int(torch.isnan(tokens).sum().item())
		tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
		aligned.append(tokens)

	stacked = torch.stack(aligned, dim=0)  # [L, B, N, D]
	num_layers, bsz, n_tokens, dim = stacked.shape
	flat = stacked.reshape(num_layers, bsz * n_tokens, dim).float()
	cos_sum = torch.einsum('lmd,kmd->lk', flat, flat)
	denominator = bsz * n_tokens
	return cos_sum, denominator, total_nan_count, total_zero_norm_count


def aggregate_over_dataset(
	aggregate: Optional[Dict[str, torch.Tensor]],
	batch_result: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
	if aggregate is None:
		return {k: v.clone().cpu() if torch.is_tensor(v) else v for k, v in batch_result.items()}

	for k in batch_result:
		if torch.is_tensor(batch_result[k]):
			aggregate[k] += batch_result[k].cpu()
	return aggregate


def _to_list(x):
	if torch.is_tensor(x):
		return x.cpu().tolist()
	return x


def _build_stability_block(
	agg_stats: Dict[str, torch.Tensor],
	sum_key: str,
	count_key: str,
	nan_key: str,
	zero_norm_key: str,
) -> Dict:
	cos_count = max(int(agg_stats[count_key].item()), 1)
	cosine_matrix = agg_stats[sum_key] / float(cos_count)
	return {
		'cosine_matrix': _to_list(cosine_matrix),
		'cosine_token_count': cos_count,
		'cosine_nan_count': int(agg_stats.get(nan_key, torch.tensor([0])).item()),
		'cosine_zero_norm_count': int(agg_stats.get(zero_norm_key, torch.tensor([0])).item()),
	}


def build_report_dict(
	agg_stats: Dict[str, torch.Tensor],
	meta: Dict,
) -> Dict:
	valid_total = int(agg_stats['valid_total'].item()) if 'valid_total' in agg_stats else meta['total_valid_tokens']
	layer_acc = agg_stats['accuracy_correct_total']
	adj_count = agg_stats['adjacent_consistency_total']

	acc_ratio = (layer_acc.float() / max(valid_total, 1)).cpu().tolist()
	adj_ratio = (adj_count.float() / max(valid_total, 1)).cpu().tolist()
	overall_adj_ratio = float(adj_count.sum().item()) / float(max(valid_total * max(len(meta['layers']) - 1, 1), 1))

	report = {
		'layers': meta['layers'],
		'num_classes': int(meta['num_classes']),
		'totals': {
			'accuracy_correct_count': _to_list(layer_acc),
			'accuracy_ratio': acc_ratio,
			'adjacent_consistency_count': _to_list(adj_count),
			'adjacent_consistency_ratio': adj_ratio,
			'adjacent_consistency_overall_ratio': overall_adj_ratio,
			'valid_total': valid_total,
		},
		'stability': _build_stability_block(
			agg_stats,
			sum_key='cosine_sum',
			count_key='cosine_token_count',
			nan_key='cosine_nan_total',
			zero_norm_key='cosine_zero_norm_total',
		),
		'meta': meta,
	}

	if 'cosine_sum_raw' in agg_stats:
		report['stability_raw'] = _build_stability_block(
			agg_stats,
			sum_key='cosine_sum_raw',
			count_key='cosine_token_count_raw',
			nan_key='cosine_nan_total_raw',
			zero_norm_key='cosine_zero_norm_total_raw',
		)

	if 'cosine_sum_ln' in agg_stats:
		cos_count_ln = max(int(agg_stats['cosine_token_count_ln'].item()), 1)
		cosine_matrix_ln = agg_stats['cosine_sum_ln'] / float(cos_count_ln)
		report['stability_ln'] = {
			'cosine_matrix': _to_list(cosine_matrix_ln),
		}

	return report


def _save_total_csv(path: str, layers: Sequence[int], values: Sequence[int], denominator: int = 0) -> None:
	with open(path, 'w', newline='') as f:
		writer = csv.writer(f)
		if denominator > 0:
			writer.writerow(['layer', 'value', 'ratio'])
			for layer, v in zip(layers, values):
				writer.writerow([layer, int(v), f"{(v/denominator):.6f}"])
		else:
			writer.writerow(['layer', 'value'])
			for layer, v in zip(layers, values):
				writer.writerow([layer, int(v)])


def _save_adjacent_csv(path: str, layers: Sequence[int], values: Sequence[int], denominator: int) -> None:
	with open(path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['layer_from', 'layer_to', 'value', 'ratio'])
		for idx, v in enumerate(values):
			writer.writerow([layers[idx], layers[idx + 1], int(v), f"{(v/max(denominator, 1)):.6f}"])


def _save_cosine_matrix_csv(path: str, layers: Sequence[int], matrix: Sequence[Sequence[float]]) -> None:
	with open(path, 'w', newline='') as f:
		writer = csv.writer(f)
		header = ['layer'] + [f'layer_{l}' for l in layers]
		writer.writerow(header)
		for i, row in enumerate(matrix):
			writer.writerow([layers[i]] + [f'{float(v):.6f}' for v in row])


def save_json_csv(report: Dict, output_dir: str) -> Dict[str, str]:
	paths = {}
	summary_path = os.path.join(output_dir, 'summary.json')
	with open(summary_path, 'w') as f:
		json.dump(report, f, indent=2)
	paths['summary'] = summary_path

	layers = report['layers']
	totals = report['totals']
	stability = report['stability']
	valid_total = totals.get('valid_total', 0)

	p = os.path.join(output_dir, 'accuracy_layer.csv')
	_save_total_csv(p, layers, totals['accuracy_correct_count'], valid_total)
	paths['accuracy_layer'] = p

	p = os.path.join(output_dir, 'stability_adjacent_consistency.csv')
	_save_adjacent_csv(p, layers, totals['adjacent_consistency_count'], valid_total)
	paths['stability_adjacent_consistency'] = p

	p = os.path.join(output_dir, 'stability_cosine_matrix.csv')
	_save_cosine_matrix_csv(p, layers, stability['cosine_matrix'])
	paths['stability_cosine_matrix'] = p

	stability_raw = report.get('stability_raw', None)
	stability_ln = report.get('stability_ln', None)
	if stability_raw is not None:
		p = os.path.join(output_dir, 'stability_cosine_matrix_raw.csv')
		_save_cosine_matrix_csv(p, layers, stability_raw['cosine_matrix'])
		paths['stability_cosine_matrix_raw'] = p

	if stability_ln is not None:
		p = os.path.join(output_dir, 'stability_cosine_matrix_ln.csv')
		_save_cosine_matrix_csv(p, layers, stability_ln['cosine_matrix'])
		paths['stability_cosine_matrix_ln'] = p

	return paths


def _plot_line(layers: Sequence[int], values: Sequence[int], title: str, ylabel: str, save_path: str) -> None:
	plt.figure(figsize=(10, 4))
	plt.plot(layers, values, marker='o', linewidth=2)
	plt.title(title)
	plt.xlabel('Layer')
	plt.ylabel(ylabel)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(save_path, dpi=150)
	plt.close()


def plot_curves_and_bars(report: Dict, output_dir: str) -> Dict[str, str]:
	paths = {}
	layers = report['layers']
	totals = report['totals']
	stability = report['stability']
	stability_raw = report.get('stability_raw', None)
	stability_ln = report.get('stability_ln', None)

	p = os.path.join(output_dir, 'accuracy_layer.png')
	_plot_line(
		layers,
		totals['accuracy_ratio'],
		'Accuracy per Layer (Semantic Match with GT)',
		'Accuracy Ratio',
		p,
	)
	paths['accuracy_layer_plot'] = p

	adj_layers = [f'{layers[i]}->{layers[i + 1]}' for i in range(max(len(layers) - 1, 0))]
	p = os.path.join(output_dir, 'stability_adjacent_consistency.png')
	plt.figure(figsize=(10, 4))
	if len(adj_layers) > 0:
		x = np.arange(len(adj_layers))
		plt.plot(x, totals['adjacent_consistency_ratio'], marker='o', linewidth=2)
		plt.xticks(x, adj_layers, rotation=45, ha='right')
	plt.title('Adjacent-Layer Semantic Consistency')
	plt.xlabel('Adjacent Layer Pair')
	plt.ylabel('Consistency Ratio')
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(p, dpi=150)
	plt.close()
	paths['stability_adjacent_consistency_plot'] = p

	p = os.path.join(output_dir, 'stability_cosine_heatmap.png')
	mat = np.array(stability['cosine_matrix'], dtype=np.float32)
	plt.figure(figsize=(8, 7))
	im = plt.imshow(mat, cmap='viridis', vmin=-1.0, vmax=1.0)
	plt.colorbar(im, fraction=0.046, pad=0.04, label='Cosine Similarity')
	plt.title('Layer-to-Layer Cosine Similarity Matrix')
	plt.xlabel('Layer')
	plt.ylabel('Layer')
	plt.xticks(np.arange(len(layers)), layers, rotation=45, ha='right')
	plt.yticks(np.arange(len(layers)), layers)
	plt.tight_layout()
	plt.savefig(p, dpi=150)
	plt.close()
	paths['stability_cosine_heatmap'] = p

	p = os.path.join(output_dir, 'stability_cosine_to_last.csv')
	with open(p, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['layer', 'cosine_to_last'])
		if mat.size > 0:
			last_col = mat[:, -1]
			for layer, val in zip(layers, last_col.tolist()):
				writer.writerow([layer, f'{float(val):.6f}'])
	paths['stability_cosine_to_last'] = p

	if stability_raw is not None:
		p = os.path.join(output_dir, 'stability_cosine_heatmap_raw.png')
		mat_raw = np.array(stability_raw['cosine_matrix'], dtype=np.float32)
		plt.figure(figsize=(8, 7))
		im = plt.imshow(mat_raw, cmap='viridis', vmin=-1.0, vmax=1.0)
		plt.colorbar(im, fraction=0.046, pad=0.04, label='Cosine Similarity')
		plt.title('Layer-to-Layer Cosine Similarity Matrix (Raw Tokens)')
		plt.xlabel('Layer')
		plt.ylabel('Layer')
		plt.xticks(np.arange(len(layers)), layers, rotation=45, ha='right')
		plt.yticks(np.arange(len(layers)), layers)
		plt.tight_layout()
		plt.savefig(p, dpi=150)
		plt.close()
		paths['stability_cosine_heatmap_raw'] = p

		p = os.path.join(output_dir, 'stability_cosine_to_last_raw.csv')
		with open(p, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['layer', 'cosine_to_last'])
			if mat_raw.size > 0:
				last_col_raw = mat_raw[:, -1]
				for layer, val in zip(layers, last_col_raw.tolist()):
					writer.writerow([layer, f'{float(val):.6f}'])
		paths['stability_cosine_to_last_raw'] = p

	if stability_ln is not None:
		p = os.path.join(output_dir, 'stability_cosine_heatmap_ln.png')
		mat_ln = np.array(stability_ln['cosine_matrix'], dtype=np.float32)
		plt.figure(figsize=(8, 7))
		im = plt.imshow(mat_ln, cmap='viridis', vmin=-1.0, vmax=1.0)
		plt.colorbar(im, fraction=0.046, pad=0.04, label='Cosine Similarity')
		plt.title('Layer-to-Layer Cosine Similarity Matrix (LN Only)')
		plt.xlabel('Layer')
		plt.ylabel('Layer')
		plt.xticks(np.arange(len(layers)), layers, rotation=45, ha='right')
		plt.yticks(np.arange(len(layers)), layers)
		plt.tight_layout()
		plt.savefig(p, dpi=150)
		plt.close()
		paths['stability_cosine_heatmap_ln'] = p

		p = os.path.join(output_dir, 'stability_cosine_to_last_ln.csv')
		with open(p, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['layer', 'cosine_to_last'])
			if mat_ln.size > 0:
				last_col_ln = mat_ln[:, -1]
				for layer, val in zip(layers, last_col_ln.tolist()):
					writer.writerow([layer, f'{float(val):.6f}'])
		paths['stability_cosine_to_last_ln'] = p

	return paths


def _prepare_batch(model, raw_batch: Dict, device: str):
	processed = model.data_preprocessor(raw_batch, training=False)
	inputs = processed['inputs']
	if isinstance(inputs, list):
		inputs = torch.stack(inputs, dim=0)
	inputs = inputs.to(device)
	data_samples = processed.get('data_samples', None)
	gt = _extract_gt_from_data_samples(data_samples)
	if gt is not None:
		gt = gt.to(device)
	return inputs, gt


def _truncate_batch(inputs: torch.Tensor, gt: Optional[torch.Tensor], remaining: int):
	if remaining <= 0:
		return inputs, gt, 0
	bsz = inputs.shape[0]
	take = min(bsz, remaining)
	inputs = inputs[:take]
	if gt is not None:
		gt = gt[:take]
	return inputs, gt, take


def main() -> None:
	args = parse_args()
	cfg, output_dir = load_cfg_and_prepare(args)
	model, dataloader = build_runner_and_dataloader(cfg, args.device)

	layer_indices = resolve_layer_indices(model, args.layer_indices)
	raw_cosine_enabled = args.raw_cosine == 'yes'
	num_classes = int(model.num_classes)
	query_features = model.query_features.to(args.device)
	query_idx = model.query_idx.to(args.device)

	total_samples = 0
	total_valid_tokens = 0
	aggregate = None

	max_samples = args.max_samples if args.max_samples > 0 else None
	for batch in dataloader:
		if max_samples is not None and total_samples >= max_samples:
			break

		inputs, gt = _prepare_batch(model, batch, args.device)
		if gt is None:
			continue

		if max_samples is not None:
			remaining = max_samples - total_samples
			inputs, gt, taken = _truncate_batch(inputs, gt, remaining)
		else:
			taken = inputs.shape[0]

		if taken == 0:
			break

		layer_tokens, layer_tokens_raw, layer_tokens_ln, token_grid = extract_layer_tokens(
			model,
			inputs,
			layer_indices,
			collect_raw_cosine=raw_cosine_enabled,
		)
		pred_by_layer = layer_token_to_pred(
			layer_tokens,
			query_features,
			query_idx,
			num_classes,
			token_grid,
		)
		token_gt, valid_mask = build_token_gt(gt, token_grid, num_classes=num_classes, ignore_index=255)
		pred_valid, gt_valid = flatten_valid(pred_by_layer, token_gt, valid_mask)

		if gt_valid.numel() == 0:
			total_samples += taken
			continue

		total_valid_tokens += int(gt_valid.numel())

		acc_total = compute_layer_accuracy_counts(pred_valid, gt_valid)
		adj_total = compute_adjacent_consistency_counts(pred_valid)
		cos_sum, cos_denom, cos_nan_count, cos_zero_norm_count = accumulate_layer_cosine_sums(layer_tokens, token_grid)
		if raw_cosine_enabled:
			if layer_tokens_raw is None:
				raise RuntimeError('Raw cosine is enabled but raw layer tokens were not returned by encoder.')
			cos_sum_raw, cos_denom_raw, cos_nan_count_raw, cos_zero_norm_count_raw = accumulate_layer_cosine_sums(
				layer_tokens_raw,
				token_grid,
			)
			if layer_tokens_ln is None:
				raise RuntimeError('Raw cosine is enabled but ln-only layer tokens were not returned by encoder.')
			cos_sum_ln, cos_denom_ln, cos_nan_count_ln, cos_zero_norm_count_ln = accumulate_layer_cosine_sums(
				layer_tokens_ln,
				token_grid,
			)

		batch_result = {
			'accuracy_correct_total': acc_total,
			'adjacent_consistency_total': adj_total,
			'cosine_sum': cos_sum,
			'cosine_token_count': torch.tensor([cos_denom], dtype=torch.long, device=cos_sum.device),
			'cosine_nan_total': torch.tensor([cos_nan_count], dtype=torch.long, device=cos_sum.device),
			'cosine_zero_norm_total': torch.tensor([cos_zero_norm_count], dtype=torch.long, device=cos_sum.device),
			'valid_total': torch.tensor([gt_valid.numel()], dtype=torch.long),
		}
		if raw_cosine_enabled:
			batch_result.update({
				'cosine_sum_raw': cos_sum_raw,
				'cosine_token_count_raw': torch.tensor([cos_denom_raw], dtype=torch.long, device=cos_sum_raw.device),
				'cosine_nan_total_raw': torch.tensor([cos_nan_count_raw], dtype=torch.long, device=cos_sum_raw.device),
				'cosine_zero_norm_total_raw': torch.tensor([cos_zero_norm_count_raw], dtype=torch.long, device=cos_sum_raw.device),
				'cosine_sum_ln': cos_sum_ln,
				'cosine_token_count_ln': torch.tensor([cos_denom_ln], dtype=torch.long, device=cos_sum_ln.device),
				'cosine_nan_total_ln': torch.tensor([cos_nan_count_ln], dtype=torch.long, device=cos_sum_ln.device),
				'cosine_zero_norm_total_ln': torch.tensor([cos_zero_norm_count_ln], dtype=torch.long, device=cos_sum_ln.device),
			})
		aggregate = aggregate_over_dataset(aggregate, batch_result)
		total_samples += taken

	if aggregate is None:
		raise RuntimeError('No valid samples were processed. Please check dataset and annotations.')

	meta = {
		'config': args.config,
		'work_dir': args.work_dir,
		'timestamp': datetime.now().isoformat(timespec='seconds'),
		'layers': layer_indices,
		'num_classes': num_classes,
		'alignment': 'interpolate_all_layers_to_reference_grid',
		'cosine_aggregation': 'mean_over_all_valid_patch_tokens',
		'raw_cosine_enabled': raw_cosine_enabled,
		'raw_token_def': 'pre_ln_post_pre_proj' if raw_cosine_enabled else None,
		'ln_token_def': 'post_ln_pre_proj' if raw_cosine_enabled else None,
		'num_samples': total_samples,
		'total_valid_tokens': total_valid_tokens,
		'save_cosine_heatmap': not bool(args.no_cosine_heatmap),
		'max_samples': args.max_samples,
	}
	report = build_report_dict(aggregate, meta)
	file_paths = save_json_csv(report, output_dir)
	plot_paths = plot_curves_and_bars(report, output_dir)
	if args.no_cosine_heatmap:
		heatmap_path = plot_paths.pop('stability_cosine_heatmap', None)
		if heatmap_path is not None and os.path.exists(heatmap_path):
			os.remove(heatmap_path)
		heatmap_path_raw = plot_paths.pop('stability_cosine_heatmap_raw', None)
		if heatmap_path_raw is not None and os.path.exists(heatmap_path_raw):
			os.remove(heatmap_path_raw)
		heatmap_path_ln = plot_paths.pop('stability_cosine_heatmap_ln', None)
		if heatmap_path_ln is not None and os.path.exists(heatmap_path_ln):
			os.remove(heatmap_path_ln)

	print('Token analysis finished.')
	print(f'Output directory: {output_dir}')
	print(f'Processed samples: {total_samples}')
	print(f'Total valid tokens: {total_valid_tokens}')
	print('Saved files:')
	for k, v in {**file_paths, **plot_paths}.items():
		print(f'  {k}: {v}')


if __name__ == '__main__':
	main()
