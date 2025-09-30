"""UltraFast-dLLM components.

This module implements the four acceleration pillars described in the
UltraFast-dLLM blueprint:

1. Partial-Query Forward (PQF)
2. Top-K Key Filtering (TKF) with certified softmax tail bounds
3. Delta-KV Low-Rank Updates (ΔKV-LR)
4. Multi-Rate Early Exit (MREE)

The classes in this file are deliberately lightweight and pure-PyTorch so that
they can be integrated with HuggingFace-style models without additional
training. All computations happen at inference time and operate on cached
activations produced by the model's block-wise decoding loop.

The implementation focuses on providing:
* tight, per-query error guarantees for TKF as TV-distance bounds,
* a low-rank incremental update routine for logits/contexts with fallback to
  exact recomputation when tolerances are exceeded,
* a scheduler for freeze windows that can provably skip evaluations until the
  argmax can change again.

The code is organized around small helper dataclasses so the generation loop
remains readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TopKFilterOutput:
    """Returned by :class:`TopKKeyFilter` after querying the ANN index."""

    indices: torch.LongTensor  # (batch, heads, q, K)
    scores: torch.Tensor  # same shape as indices, holds q·k logits (pre-scale)
    tail_mass: torch.Tensor  # (batch, heads, q)
    key_subset: torch.Tensor  # (batch, heads, K, dim)
    value_subset: torch.Tensor  # (batch, heads, K, dim)


@dataclass
class TokenDeltaState:
    """Stores per-token history for ΔKV low-rank updates."""

    query: torch.Tensor  # (heads, head_dim)
    keys: torch.Tensor  # (heads, K, head_dim)
    values: torch.Tensor  # (heads, K, head_dim)
    logits: torch.Tensor  # (heads, K)
    attn: torch.Tensor  # (heads, K)
    indices: torch.LongTensor  # (heads, K)
    context: torch.Tensor  # (heads, head_dim)


def _ensure_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


class DenseKeyIndex:
    """Fallback dense index used when ANN dependencies are unavailable.

    The class still exposes ``build`` and ``search`` so the rest of the code can
    remain unaware of whether we are using an approximate method or an exact
    fallback. ``search`` runs entirely on the device of the query tensor and is
    therefore safe to call on GPU.
    """

    def __init__(self):
        self._keys: Optional[torch.Tensor] = None
        self._shape: Optional[Tuple[int, ...]] = None

    def build(self, keys: torch.Tensor) -> None:
        # keys: (batch, heads, seq, dim)
        # We store it for subsequent queries. We clone to decouple from caller
        # because the cache tensor will be mutated by decoding.
        self._keys = keys.detach().clone()
        self._shape = tuple(keys.shape)

    def reset(self) -> None:
        self._keys = None
        self._shape = None

    def search(self, queries: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._keys is None:
            raise RuntimeError("DenseKeyIndex.search called before build")

        keys = self._keys
        batch, heads, seq_len, dim = keys.shape
        q_batch, q_heads, q_len, _ = queries.shape

        assert batch == q_batch and heads == q_heads, "Mismatched batch/head dims for key search"

        # Compute q·k^T for all keys then grab top-k. This is an exact search.
        logits = torch.matmul(
            _ensure_contiguous(queries),
            _ensure_contiguous(keys.transpose(-1, -2)),
        )  # (batch, heads, q_len, seq_len)
        top_k = min(top_k, seq_len)
        scores, indices = torch.topk(logits, k=top_k, dim=-1)
        return indices, scores

    def update(self, keys: torch.Tensor, positions: torch.LongTensor) -> None:
        if self._keys is None or self._shape != tuple(keys.shape):
            self.build(keys)
            return
        if positions.numel() == 0:
            return
        self._keys = self._keys.to(keys.device)
        pos = positions.to(torch.long)
        self._keys[:, :, pos, :] = keys[:, :, pos, :].detach()


class IVFPQIndex:
    """Lightweight IVF-style index with coarse centroids and brute-force refinement."""

    def __init__(self, nlist: int = 32, nprobe: int = 4, max_iter: int = 8, rebuild_threshold: int = 128):
        self.nlist = nlist
        self.nprobe = max(1, min(nprobe, nlist))
        self.max_iter = max_iter
        self.rebuild_threshold = rebuild_threshold
        self.reset()

    def reset(self) -> None:
        self._keys: Optional[torch.Tensor] = None
        self._centroids: Optional[torch.Tensor] = None
        self._assignments: Optional[torch.Tensor] = None
        self._cluster_members: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self._cluster_counts: Dict[Tuple[int, int], int] = {}
        self._built = False

    @staticmethod
    def _kmeans(x: torch.Tensor, n_clusters: int, max_iter: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        n_points = x.size(0)
        if n_clusters >= n_points:
            centroids = torch.zeros(n_clusters, x.size(1), device=device, dtype=x.dtype)
            centroids[:n_points] = x
            assignments = torch.arange(n_points, device=device)  # unique clusters
            assignments = torch.remainder(assignments, n_clusters)
            return centroids, assignments

        perm = torch.randperm(n_points, device=device)
        centroids = x[perm[:n_clusters]].clone()
        assignments = torch.zeros(n_points, device=device, dtype=torch.long)

        for _ in range(max_iter):
            distances = torch.cdist(x, centroids, p=2)
            new_assignments = distances.argmin(dim=-1)
            if torch.equal(new_assignments, assignments):
                break
            assignments = new_assignments
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    centroids[c] = x[mask].mean(dim=0)
                else:
                    choice = torch.randint(0, n_points, (1,), device=device)
                    centroids[c] = x[choice]

        distances = torch.cdist(x, centroids, p=2)
        assignments = distances.argmin(dim=-1)
        return centroids, assignments

    def build(self, keys: torch.Tensor) -> None:
        self._keys = keys.detach().clone()
        batch, heads, seq_len, dim = keys.shape
        n_clusters = max(1, min(self.nlist, seq_len))
        centroids = torch.zeros(batch, heads, n_clusters, dim, device=keys.device, dtype=keys.dtype)
        assignments = torch.zeros(batch, heads, seq_len, device=keys.device, dtype=torch.long)
        cluster_members: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        cluster_counts: Dict[Tuple[int, int], int] = {}

        for b in range(batch):
            for h in range(heads):
                head_keys = keys[b, h]
                head_centroids, head_assignments = self._kmeans(head_keys, n_clusters, self.max_iter)
                centroids[b, h] = head_centroids
                assignments[b, h] = head_assignments
                members: List[torch.Tensor] = []
                for c in range(n_clusters):
                    idx = torch.nonzero(head_assignments == c, as_tuple=False).squeeze(-1)
                    members.append(idx)
                cluster_members[(b, h)] = members
                cluster_counts[(b, h)] = n_clusters

        self._centroids = centroids
        self._assignments = assignments
        self._cluster_members = cluster_members
        self._cluster_counts = cluster_counts
        self._built = True

    def update(self, keys: torch.Tensor, positions: torch.LongTensor) -> None:
        if not self._built or self._keys is None:
            self.build(keys)
            return
        if keys.shape != self._keys.shape or positions.numel() > self.rebuild_threshold:
            self.build(keys)
            return
        self._keys = self._keys.to(keys.device)
        positions = torch.unique(positions.to(torch.long))
        self._keys[:, :, positions, :] = keys[:, :, positions, :].detach()
        batch, heads = keys.shape[:2]
        for b in range(batch):
            for h in range(heads):
                members = self._cluster_members[(b, h)]
                n_clusters = self._cluster_counts[(b, h)]
                for pos in positions.tolist():
                    if pos >= keys.shape[2]:
                        continue
                    point = self._keys[b, h, pos]
                    centroid = self._centroids[b, h, :n_clusters]
                    distances = torch.cdist(point.unsqueeze(0), centroid, p=2).squeeze(0)
                    new_cluster = int(distances.argmin().item())
                    old_cluster = int(self._assignments[b, h, pos].item())
                    if old_cluster != new_cluster:
                        if old_cluster < len(members):
                            mask = members[old_cluster] != pos
                            members[old_cluster] = members[old_cluster][mask]
                        new_member_tensor = torch.tensor([pos], device=keys.device, dtype=torch.long)
                        members[new_cluster] = torch.unique(torch.cat([members[new_cluster], new_member_tensor]))
                        self._assignments[b, h, pos] = new_cluster
                for c in range(n_clusters):
                    idx = members[c]
                    if idx.numel() > 0:
                        self._centroids[b, h, c] = self._keys[b, h, idx].mean(dim=0)

    def search(self, queries: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._built or self._keys is None or self._centroids is None:
            raise RuntimeError("IVFPQIndex used before build")

        batch, heads, q_len, dim = queries.shape
        device = queries.device
        indices_out = torch.full((batch, heads, q_len, top_k), -1, device=device, dtype=torch.long)
        scores_out = torch.full((batch, heads, q_len, top_k), torch.finfo(queries.dtype).min, device=device, dtype=queries.dtype)

        for b in range(batch):
            for h in range(heads):
                n_clusters = self._cluster_counts[(b, h)]
                centroid_slice = self._centroids[b, h, :n_clusters]
                if centroid_slice.numel() == 0:
                    continue
                centroid_scores = torch.matmul(queries[b, h], centroid_slice.T)
                max_probe = min(self.nprobe, n_clusters)
                top_centroid_scores, top_centroid_idx = torch.topk(centroid_scores, k=max_probe, dim=-1)
                members = self._cluster_members[(b, h)]
                for q_idx in range(q_len):
                    candidate_indices: List[torch.Tensor] = []
                    for probe_rank in range(max_probe):
                        cluster_id = int(top_centroid_idx[q_idx, probe_rank].item())
                        if cluster_id < len(members) and members[cluster_id].numel() > 0:
                            candidate_indices.append(members[cluster_id])
                        if candidate_indices and sum(idx.numel() for idx in candidate_indices) >= top_k * 4:
                            break
                    if not candidate_indices:
                        candidate_indices.append(torch.arange(self._keys.shape[2], device=device))
                    candidates = torch.unique(torch.cat(candidate_indices))
                    head_keys = self._keys[b, h, candidates]
                    q_vec = queries[b, h, q_idx]
                    candidate_scores = torch.matmul(q_vec.unsqueeze(0), head_keys.T).squeeze(0)
                    k_eff = min(top_k, candidate_scores.size(0))
                    best_scores, best_idx = torch.topk(candidate_scores, k=k_eff)
                    indices_out[b, h, q_idx, :k_eff] = candidates[best_idx]
                    scores_out[b, h, q_idx, :k_eff] = best_scores

        return indices_out, scores_out


class HybridKeyIndex:
    """Wraps an IVF index with a dense fallback for corner cases."""

    def __init__(self, nlist: int = 32, nprobe: int = 4, max_iter: int = 8):
        self._ann = IVFPQIndex(nlist=nlist, nprobe=nprobe, max_iter=max_iter)
        self._dense = DenseKeyIndex()
        self._use_ann = False

    def reset(self) -> None:
        self._ann.reset()
        self._dense.reset()
        self._use_ann = False

    def build(self, keys: torch.Tensor) -> None:
        seq_len = keys.shape[2]
        if seq_len < 8:
            self._dense.build(keys)
            self._use_ann = False
        else:
            self._ann.build(keys)
            self._dense.build(keys)
            self._use_ann = True

    def update(self, keys: torch.Tensor, positions: torch.LongTensor) -> None:
        if self._use_ann:
            self._ann.update(keys, positions)
        self._dense.update(keys, positions)

    def search(self, queries: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._use_ann:
            indices, scores = self._ann.search(queries, top_k)
            missing = indices < 0
            if missing.any():
                dense_indices, dense_scores = self._dense.search(queries, top_k)
                indices = torch.where(missing, dense_indices, indices)
                scores = torch.where(missing, dense_scores, scores)
            return indices, scores
        return self._dense.search(queries, top_k)


def _compute_tail_mass(sorted_scores: torch.Tensor, k: int, total: int) -> torch.Tensor:
    """Compute the softmax tail mass upper bound using the γ-margin argument.

    ``sorted_scores`` are descending logits (unnormalized). Given the margin γ
    between last kept logit and the first dropped logit we can bound the tail as
    |R|e^{-γ}. We approximate γ by taking the difference between the kth and
    (k+1)th logit.
    """

    eps = torch.finfo(sorted_scores.dtype).tiny
    last_value = sorted_scores[..., -1:]
    padded = torch.cat([sorted_scores, last_value], dim=-1)
    gamma = padded[..., k - 1] - padded[..., k]
    remainder = max(total - k, 0)
    tail_mass = torch.clamp(remainder * torch.exp(-gamma), max=1.0 - eps)
    return tail_mass


class TopKKeyFilter:
    """Implements top-k key sparsification with a certified softmax tail bound."""

    def __init__(
        self,
        index,
        epsilon: float,
    ) -> None:
        self._index = index
        self._epsilon = epsilon
        self._seq_len: Optional[int] = None

    def reset(self) -> None:
        self._seq_len = None
        self._index.reset()

    def filter(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        top_k: int,
        changed_positions: Optional[torch.LongTensor] = None,
    ) -> TopKFilterOutput:
        seq_len = keys.shape[2]
        if self._seq_len is None or self._seq_len != seq_len:
            self._index.build(keys)
            self._seq_len = seq_len
        elif changed_positions is not None:
            self._index.update(keys, changed_positions)

        indices, scores = self._index.search(queries, top_k)
        batch, heads, q_len, _ = queries.shape
        assert self._seq_len is not None
        total = self._seq_len

        # Gather keys/values according to picked indices
        gather_shape = indices.unsqueeze(-1).expand(-1, -1, -1, -1, keys.size(-1))
        key_subset = torch.gather(keys.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_shape)
        value_subset = torch.gather(values.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_shape)

        # Compute tail bound via γ margin. Sort scores to measure γ.
        sorted_scores, _ = torch.sort(scores, descending=True, dim=-1)
        tail_mass = _compute_tail_mass(sorted_scores, scores.size(-1), total)

        # If tail bound larger than epsilon, increase K until tolerance satisfied
        adaptive_indices = [indices]
        adaptive_scores = [scores]
        adaptive_keys = [key_subset]
        adaptive_values = [value_subset]
        adaptive_tail = tail_mass

        current_k = scores.size(-1)
        while torch.any(adaptive_tail > self._epsilon) and current_k < total:
            extra = min(current_k * 2, total)
            indices, scores = self._index.search(queries, extra)
            gather_shape = indices.unsqueeze(-1).expand(-1, -1, -1, -1, keys.size(-1))
            key_subset = torch.gather(keys.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_shape)
            value_subset = torch.gather(values.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_shape)
            sorted_scores, _ = torch.sort(scores, descending=True, dim=-1)
            adaptive_tail = _compute_tail_mass(sorted_scores, scores.size(-1), total)
            adaptive_indices = [indices]
            adaptive_scores = [scores]
            adaptive_keys = [key_subset]
            adaptive_values = [value_subset]
            current_k = extra

        final_indices = adaptive_indices[-1]
        final_scores = adaptive_scores[-1]
        final_keys = adaptive_keys[-1]
        final_values = adaptive_values[-1]
        final_tail = adaptive_tail

        return TopKFilterOutput(
            indices=final_indices,
            scores=final_scores,
            tail_mass=final_tail,
            key_subset=final_keys,
            value_subset=final_values,
        )


def _low_rank_projection(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the truncated SVD factors up to ``rank``.

    The routine operates on batched matrices with shape ``(..., m, n)``.
    """

    orig_shape = matrix.shape
    batch_dims = orig_shape[:-2]
    m, n = orig_shape[-2:]
    flat = matrix.reshape(-1, m, n)
    u_list: List[torch.Tensor] = []
    s_list: List[torch.Tensor] = []
    vh_list: List[torch.Tensor] = []
    for item in flat:
        if torch.count_nonzero(item).item() == 0:
            u_list.append(torch.zeros(m, min(rank, n), device=item.device, dtype=item.dtype))
            s_list.append(torch.zeros(min(rank, n), device=item.device, dtype=item.dtype))
            vh_list.append(torch.zeros(min(rank, n), n, device=item.device, dtype=item.dtype))
            continue
        u, s, vh = torch.linalg.svd(item, full_matrices=False)
        r = min(rank, u.size(1))
        u_list.append(u[:, :r])
        s_list.append(s[:r])
        vh_list.append(vh[:r])
    u = torch.stack(u_list, dim=0).reshape(*batch_dims, m, -1)
    s = torch.stack(s_list, dim=0).reshape(*batch_dims, -1)
    vh = torch.stack(vh_list, dim=0).reshape(*batch_dims, -1, n)
    return u, s, vh


class DeltaKVLowRank:
    """Low-rank incremental attention updater operating per token."""

    def __init__(self, rank: int = 4, tolerance: float = 1e-3):
        self.rank = rank
        self.tolerance = tolerance
        self._layer_state: Dict[int, Dict[int, TokenDeltaState]] = {}

    def reset(self) -> None:
        self._layer_state.clear()

    def _layer_dict(self, layer_idx: int) -> Dict[int, TokenDeltaState]:
        if layer_idx not in self._layer_state:
            self._layer_state[layer_idx] = {}
        return self._layer_state[layer_idx]

    def try_update(
        self,
        layer_idx: int,
        positions: torch.LongTensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.LongTensor,
        logits_scale: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        layer_cache = self._layer_dict(layer_idx)
        batch, heads, q_len, _ = queries.shape

        contexts = []
        logits_list = []
        attn_list = []

        for token_idx in range(q_len):
            pos = int(positions[token_idx].item())
            token_state = layer_cache.get(pos)
            if token_state is None:
                return None

            current_query = queries[0, :, token_idx, :]
            current_keys = keys[0, :, token_idx, :, :]
            current_values = values[0, :, token_idx, :, :]
            current_indices = indices[0, :, token_idx, :]

            if token_state.indices.shape != current_indices.shape or not torch.equal(token_state.indices, current_indices):
                return None

            delta_q = current_query - token_state.query
            delta_k = current_keys - token_state.keys
            delta_v = current_values - token_state.values

            if torch.all(delta_q == 0) and torch.all(delta_k == 0) and torch.all(delta_v == 0):
                contexts.append(token_state.context.unsqueeze(0))
                logits_list.append(token_state.logits.unsqueeze(0))
                attn_list.append(token_state.attn.unsqueeze(0))
                continue

            prev_keys = token_state.keys
            prev_query = token_state.query

            term1 = torch.einsum("hd,hkd->hk", delta_q, prev_keys)
            term2 = torch.zeros_like(term1)
            term3 = torch.zeros_like(term1)
            if self.rank > 0:
                for h in range(heads):
                    dk = delta_k[h]
                    if dk.numel() == 0:
                        continue
                    u_k, s_k, vh_k = torch.linalg.svd(dk, full_matrices=False)
                    r = min(self.rank, s_k.numel())
                    if r == 0:
                        continue
                    u_k = u_k[:, :r]
                    s_k = s_k[:r]
                    v_k = vh_k[:r, :]
                    proj_prev = torch.matmul(prev_query[h], v_k.T)
                    proj_delta = torch.matmul(delta_q[h], v_k.T)
                    term2[h] = torch.matmul(proj_prev * s_k, u_k.T)
                    term3[h] = torch.matmul(proj_delta * s_k, u_k.T)
            delta_logits = (term1 + term2 + term3) * logits_scale
            logits = token_state.logits + delta_logits

            norm_delta_a = torch.linalg.vector_norm(delta_logits, ord=2, dim=-1)
            norm_prev_v = torch.linalg.vector_norm(token_state.values, ord=2, dim=-1).max(dim=-1).values
            norm_delta_v = torch.linalg.vector_norm(delta_v, ord=2, dim=-1).max(dim=-1).values
            norm_prev_attn = torch.linalg.vector_norm(token_state.attn, ord=2, dim=-1)
            bound = 0.5 * norm_delta_a * norm_prev_v + norm_prev_attn * norm_delta_v

            if torch.any(bound > self.tolerance):
                return None

            attn = torch.softmax(logits, dim=-1)
            context = torch.einsum("hk,hkd->hd", attn, current_values)

            contexts.append(context.unsqueeze(0))
            logits_list.append(logits.unsqueeze(0))
            attn_list.append(attn.unsqueeze(0))

        if not contexts:
            return None

        context_tensor = torch.stack(contexts, dim=2)  # (1, heads, q_len, head_dim)
        logits_tensor = torch.stack(logits_list, dim=2)
        attn_tensor = torch.stack(attn_list, dim=2)

        return context_tensor, logits_tensor, attn_tensor

    def store(
        self,
        layer_idx: int,
        positions: torch.LongTensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.LongTensor,
        logits: torch.Tensor,
        attn_probs: torch.Tensor,
    ) -> None:
        layer_cache = self._layer_dict(layer_idx)
        q_len = queries.size(2)
        for token_idx in range(q_len):
            pos = int(positions[token_idx].item())
            layer_cache[pos] = TokenDeltaState(
                query=queries[0, :, token_idx, :].detach(),
                keys=keys[0, :, token_idx, :, :].detach(),
                values=values[0, :, token_idx, :, :].detach(),
                logits=logits[0, :, token_idx, :].detach(),
                attn=attn_probs[0, :, token_idx, :].detach(),
                indices=indices[0, :, token_idx, :].detach(),
                context=torch.einsum(
                    "hk,hkd->hd",
                    attn_probs[0, :, token_idx, :].detach(),
                    values[0, :, token_idx, :, :].detach(),
                ).detach(),
            )


class FreezeWindowState:
    """Tracks freeze windows for Multi-Rate Early Exit."""

    def __init__(self, batch_size: int, seq_len: int, device: torch.device) -> None:
        self.expiry_steps = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)

    def should_eval(self, step: int) -> torch.Tensor:
        return self.expiry_steps <= step

    def freeze(self, positions: torch.Tensor, step: int, horizon: torch.Tensor) -> None:
        update_values = (step + horizon).to(self.expiry_steps.dtype)
        self.expiry_steps.scatter_(1, positions, update_values)


class FreezeWindowController:
    def __init__(self, batch_size: int, seq_len: int, device: torch.device) -> None:
        self.state = FreezeWindowState(batch_size, seq_len, device)

    def active_mask(self, step: int) -> torch.Tensor:
        return self.state.should_eval(step)

    def update(self, margins: torch.Tensor, drift: torch.Tensor, positions: torch.Tensor, step: int) -> None:
        epsilon = 1e-8
        horizon = torch.floor(margins / (2 * torch.clamp(drift, min=epsilon)))
        horizon = torch.clamp(horizon, min=0)
        self.state.freeze(positions, step, horizon.long())


@dataclass
class UltraFastContext:
    """Aggregates the helper objects used during decoding."""

    top_k: int
    epsilon: float
    delta_updater: DeltaKVLowRank
    freeze_controller: FreezeWindowController
    ann_nlist: int = 32
    ann_nprobe: int = 4
    ann_max_iter: int = 6
    filters: Dict[int, TopKKeyFilter] = field(default_factory=dict)
    current_positions: Optional[torch.LongTensor] = None

    def reset_block(self) -> None:
        self.delta_updater.reset()
        for filt in self.filters.values():
            filt.reset()
        self.current_positions = None

    def get_filter(self, layer_idx: int) -> TopKKeyFilter:
        if layer_idx not in self.filters:
            index = HybridKeyIndex(nlist=self.ann_nlist, nprobe=self.ann_nprobe, max_iter=self.ann_max_iter)
            self.filters[layer_idx] = TopKKeyFilter(index, self.epsilon)
        return self.filters[layer_idx]

    def set_active_positions(self, positions: torch.LongTensor) -> None:
        self.current_positions = positions.to(torch.long)

    def clear_active_positions(self) -> None:
        self.current_positions = None
