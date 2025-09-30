import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TokenMetric:
    """Stores per-token metrics accumulated within a segment."""

    margin_integral: float = 0.0
    last_margin: float = 0.0
    last_slope: float = 0.0
    top1_prob: float = 0.0
    top2_prob: float = 0.0
    committed_token: Optional[int] = None
    commit_step: Optional[int] = None
    sor_score: float = 0.0
    recovered_step: Optional[int] = None


@dataclass
class SegmentRecord:
    """Summary of a segment run, including commit metadata and diagnostics."""

    index: int
    start_step: int
    end_step: int
    certificate: float
    committed_tokens: Set[int] = field(default_factory=set)
    token_metrics: Dict[int, TokenMetric] = field(default_factory=dict)
    branch_topk: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    sor_results: Dict[int, float] = field(default_factory=dict)
    x_before: Optional[torch.Tensor] = None
    x_after: Optional[torch.Tensor] = None
    conflict_tokens: Set[int] = field(default_factory=set)


@dataclass
class SearchNode:
    """Represents a DFS node in the IDD controller."""

    x: torch.Tensor
    segment_idx: int
    certificate: float
    history: List[SegmentRecord] = field(default_factory=list)
    branches_used: int = 0


class IDDController:
    """Depth-First Denoise-and-Backtrack controller implementing Instant Diligent Diffusion."""

    def __init__(
        self,
        model,
        x_init: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        tok_idx: Optional[torch.Tensor],
        timesteps: torch.Tensor,
        num_steps: int,
        segment_boundaries: Sequence[Tuple[int, int]],
        mask_token_id: int,
        threshold: float,
        branch_top_k: int,
        branch_budget: int,
        max_nodes: int,
        sor_steps: int,
        sor_threshold: float,
        sor_max_tokens: int,
        lambda_sor: float,
        mmi_threshold: float,
        slope_epsilon: float,
        tokens_hook: Callable[[Optional[int], torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        logits_hook: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
        max_conflict_tokens: int = 4,
        forward_override: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        reset_context: Optional[Callable[[], None]] = None,
    ) -> None:
        self.model = model
        self.device = x_init.device
        self.x_init = x_init
        self.attention_mask = attention_mask
        self.tok_idx = tok_idx
        self.timesteps = timesteps
        self.num_steps = num_steps
        self.segment_boundaries = list(segment_boundaries)
        self.mask_token_id = mask_token_id
        self.tau_c = threshold
        self.branch_top_k = max(2, branch_top_k)
        self.branch_budget = branch_budget
        self.max_nodes = max_nodes
        self.sor_steps = sor_steps
        self.sor_threshold = sor_threshold
        self.sor_max_tokens = sor_max_tokens
        self.lambda_sor = lambda_sor
        self.mmi_threshold = mmi_threshold
        self.slope_epsilon = slope_epsilon
        self.tokens_hook = tokens_hook
        self.logits_hook = logits_hook
        self.max_conflict_tokens = max_conflict_tokens
        self.forward_override = forward_override
        self.reset_context = reset_context

        self.vocab_size = None  # lazy init

    def run(self) -> Tuple[torch.Tensor, float, List[SegmentRecord]]:
        """Executes the DFS-based denoising search and returns the best sequence."""
        stack: List[SearchNode] = [
            SearchNode(
                x=self.tokens_hook(None, self.x_init.clone(), None),
                segment_idx=0,
                certificate=0.0,
                history=[],
                branches_used=0,
            )
        ]
        visited = 0
        best_sequence: Optional[torch.Tensor] = None
        best_score = -math.inf
        best_history: List[SegmentRecord] = []

        while stack and visited < self.max_nodes:
            node = stack.pop()
            visited += 1

            if node.segment_idx >= len(self.segment_boundaries):
                # Completed all segments. Check if there are remaining masks; if so, treat as incomplete path.
                if torch.any(node.x == self.mask_token_id):
                    continue
                if node.certificate > best_score:
                    best_score = node.certificate
                    best_sequence = node.x.clone()
                    best_history = node.history
                continue

            record = self._run_segment(node.x.clone(), node.segment_idx)
            if record is None:
                # No progress possible; skip this branch.
                continue

            new_history = node.history + [record]
            new_certificate = node.certificate + record.certificate

            if not record.conflict_tokens:
                stack.append(
                    SearchNode(
                        x=record.x_after.clone(),
                        segment_idx=node.segment_idx + 1,
                        certificate=new_certificate,
                        history=new_history,
                        branches_used=node.branches_used,
                    )
                )
                continue

            # Conflict encountered: determine earliest segment that needs backtracking
            backtrack_idx = self._find_backtrack_segment(new_history, record.conflict_tokens)
            if backtrack_idx is None:
                continue

            branches_remaining = self.branch_budget - node.branches_used
            if branches_remaining <= 0:
                # Branch budget exhausted; skip this path.
                continue

            candidates = self._generate_branch_candidates(
                new_history,
                backtrack_idx,
                record.conflict_tokens,
                branches_remaining,
                prefix_certificate=sum(r.certificate for r in new_history[:backtrack_idx]),
                branches_used=node.branches_used,
            )
            for candidate in candidates:
                stack.append(candidate)

        if best_sequence is None:
            # Fallback: return the best partial sequence encountered (if any)
            if stack:
                node = max(stack, key=lambda n: n.certificate)
                best_sequence = node.x.clone()
                best_history = node.history
                best_score = node.certificate
            else:
                best_sequence = self.x_init.clone()
                best_history = []
                best_score = 0.0

        return best_sequence, best_score, best_history

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _run_segment(self, x: torch.Tensor, segment_idx: int) -> Optional[SegmentRecord]:
        start_step, end_step = self.segment_boundaries[segment_idx]
        record = SegmentRecord(
            index=segment_idx,
            start_step=start_step,
            end_step=end_step,
            certificate=0.0,
            committed_tokens=set(),
            token_metrics={},
            branch_topk={},
            sor_results={},
            x_before=x.clone(),
            conflict_tokens=set(),
        )

        self._reset_context_if_needed()

        for step_idx in range(start_step, end_step):
            mask_index = (x == self.mask_token_id)
            if mask_index.sum() == 0:
                break

            logits = self._forward(x)
            logits = self.logits_hook(step_idx, x, logits)

            mask_logits = logits[mask_index]
            if mask_logits.numel() == 0:
                break

            if self.vocab_size is None:
                self.vocab_size = mask_logits.size(-1)
            k = min(self.branch_top_k, self.vocab_size)

            probs = torch.softmax(mask_logits, dim=-1)
            top_probs, top_ids = probs.topk(k, dim=-1)
            if probs.size(-1) > 1:
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                p1 = sorted_probs[:, 0]
                p2 = sorted_probs[:, 1]
            else:
                p1 = probs.squeeze(-1)
                p2 = torch.zeros_like(p1)
            margins = p1 - p2
            delta_t = float((self.timesteps[step_idx] - self.timesteps[step_idx + 1]).abs().item())
            delta_t = max(delta_t, 1e-6)

            positions = mask_index.nonzero(as_tuple=False)
            slopes = []
            for idx, pos in enumerate(positions):
                token_pos = pos[1].item()
                metrics = record.token_metrics.setdefault(token_pos, TokenMetric())
                metrics.margin_integral += float(margins[idx].item() * delta_t)
                metrics.last_slope = (
                    float(margins[idx].item()) - metrics.last_margin
                ) / delta_t
                metrics.last_margin = float(margins[idx].item())
                metrics.top1_prob = float(p1[idx].item())
                metrics.top2_prob = float(p2[idx].item())
                metrics.commit_step = step_idx
                if step_idx == start_step:
                    record.branch_topk[token_pos] = (top_ids[idx].detach().clone(), top_probs[idx].detach().clone())
                slopes.append(metrics.last_slope)

            slopes_tensor = torch.tensor(slopes, device=mask_logits.device)
            eligible = (p1 >= self.tau_c) & (slopes_tensor >= -self.slope_epsilon)

            if eligible.any():
                commit_positions = positions[eligible].tolist()
                commit_values = top_ids[eligible, 0]
                for row_col, value in zip(commit_positions, commit_values):
                    row, col = row_col
                    x[row, col] = value
                    metrics = record.token_metrics[col]
                    metrics.committed_token = int(value.item())
                    record.committed_tokens.add(col)

            x = self.tokens_hook(step_idx, x, logits)

        # If no tokens were committed in this segment, skip SoR and conflict checks
        if not record.committed_tokens:
            record.x_after = x
            return record

        sor_scores = self._run_sor_checks(x.clone(), record)
        record.sor_results = sor_scores

        # Conflict detection
        certificate = 0.0
        conflicts: Set[int] = set()
        for token_pos in record.committed_tokens:
            metrics = record.token_metrics[token_pos]
            sor_score = sor_scores.get(token_pos, 1.0)
            metrics.sor_score = sor_score

            if metrics.margin_integral < self.mmi_threshold:
                conflicts.add(token_pos)
            if metrics.last_slope < -self.slope_epsilon:
                conflicts.add(token_pos)
            if sor_score < self.sor_threshold:
                conflicts.add(token_pos)

            certificate += metrics.margin_integral + self.lambda_sor * sor_score

        if conflicts:
            if len(conflicts) > self.max_conflict_tokens:
                ordered = sorted(
                    list(conflicts),
                    key=lambda pos: record.token_metrics[pos].margin_integral,
                )
                conflicts = set(ordered[: self.max_conflict_tokens])
            # Reset conflicted tokens to mask to reflect backtracking requirement
            for token_pos in conflicts:
                x[0, token_pos] = self.mask_token_id
            record.conflict_tokens = conflicts
            # Remove conflicts from committed set as they are reverted
            record.committed_tokens = {pos for pos in record.committed_tokens if pos not in conflicts}
            certificate = sum(
                record.token_metrics[pos].margin_integral + self.lambda_sor * record.token_metrics[pos].sor_score
                for pos in record.committed_tokens
            )

        record.certificate = certificate
        record.x_after = x
        return record

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.forward_override is not None:
            logits = self.forward_override(x)
        else:
            attention_mask = None if isinstance(self.attention_mask, str) else self.attention_mask
            logits = self.model(x, attention_mask, self.tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits

    def _reset_context_if_needed(self) -> None:
        if self.reset_context is not None:
            self.reset_context()

    def _run_sor_checks(self, x: torch.Tensor, record: SegmentRecord) -> Dict[int, float]:
        if not record.committed_tokens:
            return {}

        tokens_sorted = sorted(
            record.committed_tokens,
            key=lambda pos: record.token_metrics[pos].margin_integral,
        )
        tokens_to_probe = tokens_sorted[: self.sor_max_tokens]
        if not tokens_to_probe:
            return {}

        original_tokens = {pos: int(x[0, pos].item()) for pos in tokens_to_probe}
        probe = x.clone()
        for pos in tokens_to_probe:
            probe[0, pos] = self.mask_token_id

        recovered_steps: Dict[int, Optional[int]] = {pos: None for pos in tokens_to_probe}

        start_step = record.start_step
        end_step = min(record.end_step, start_step + self.sor_steps)
        for local_idx, step_idx in enumerate(range(start_step, end_step)):
            mask_index = (probe == self.mask_token_id)
            if mask_index.sum() == 0:
                break

            logits = self._forward(probe)
            logits = self.logits_hook(step_idx, probe, logits)
            mask_logits = logits[mask_index]
            if mask_logits.numel() == 0:
                break
            probs = torch.softmax(mask_logits, dim=-1)
            top_values, top_ids = probs.max(dim=-1)

            positions = mask_index.nonzero(as_tuple=False)
            for idx, pos in enumerate(positions):
                token_pos = pos[1].item()
                if token_pos not in tokens_to_probe:
                    continue
                predicted = int(top_ids[idx].item())
                probe[pos[0], token_pos] = predicted
                if predicted == original_tokens[token_pos] and recovered_steps[token_pos] is None:
                    recovered_steps[token_pos] = local_idx + 1

            probe = self.tokens_hook(step_idx, probe, logits)

        sor_scores: Dict[int, float] = {}
        for pos in tokens_to_probe:
            step = recovered_steps[pos]
            if step is None:
                sor_scores[pos] = 0.0
            else:
                sor_scores[pos] = 1.0 - (step - 1) / max(1, self.sor_steps)
        return sor_scores

    def _find_backtrack_segment(
        self, history: Sequence[SegmentRecord], conflict_tokens: Set[int]
    ) -> Optional[int]:
        candidate_indices: List[int] = []
        for record in history:
            if conflict_tokens & record.committed_tokens:
                candidate_indices.append(record.index)
        if not candidate_indices:
            return None
        return min(candidate_indices)

    def _generate_branch_candidates(
        self,
        history: List[SegmentRecord],
        backtrack_idx: int,
        conflict_tokens: Set[int],
        branches_remaining: int,
        prefix_certificate: float,
        branches_used: int,
    ) -> List[SearchNode]:
        record = history[backtrack_idx]
        tokens_in_segment = list(conflict_tokens & record.branch_topk.keys())
        if not tokens_in_segment:
            return []

        # Focus on the most problematic token (lowest certificate contribution)
        def token_score(pos: int) -> float:
            metrics = record.token_metrics.get(pos)
            if metrics is None:
                return -math.inf
            return metrics.margin_integral + self.lambda_sor * metrics.sor_score

        tokens_in_segment.sort(key=token_score)
        pivot_token = tokens_in_segment[0]
        top_ids, top_probs = record.branch_topk[pivot_token]
        current_token = record.token_metrics[pivot_token].committed_token
        if current_token is None:
            return []

        alternatives: List[int] = []
        for token_id in top_ids.tolist():
            if token_id != current_token:
                alternatives.append(int(token_id))
            if len(alternatives) >= branches_remaining:
                break

        if not alternatives:
            return []

        # Build prefix history up to the segment before the backtrack target
        prefix_history = [rec for rec in history if rec.index < backtrack_idx]

        candidates: List[SearchNode] = []
        for alt_token in alternatives:
            x_seed = record.x_before.clone()
            x_seed[0, pivot_token] = alt_token
            rerun_record = self._run_segment(x_seed, backtrack_idx)
            if rerun_record is None:
                continue

            new_history = prefix_history + [rerun_record]
            new_certificate = prefix_certificate + rerun_record.certificate
            candidates.append(
                SearchNode(
                    x=rerun_record.x_after.clone(),
                    segment_idx=backtrack_idx + 1,
                    certificate=new_certificate,
                    history=new_history,
                    branches_used=branches_used + 1,
                )
            )

        # Sort candidates by certificate to prioritize DFS
        candidates.sort(key=lambda node: node.certificate, reverse=True)
        return candidates
