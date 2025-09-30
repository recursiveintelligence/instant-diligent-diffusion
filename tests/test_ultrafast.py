import math
import unittest

import torch

from dream.model.ultrafast import (
    DenseKeyIndex,
    DeltaKVLowRank,
    FreezeWindowController,
    TopKKeyFilter,
)


class UltraFastComponentsTest(unittest.TestCase):
    def test_topk_tail_bound_respects_epsilon(self):
        torch.manual_seed(0)
        batch, heads, seq_len, dim = 1, 2, 32, 16
        queries = torch.randn(batch, heads, 4, dim)
        keys = torch.randn(batch, heads, seq_len, dim)
        values = torch.randn(batch, heads, seq_len, dim)

        epsilon = 1e-2
        filter_obj = TopKKeyFilter(DenseKeyIndex(), epsilon=epsilon)
        result = filter_obj.filter(queries, keys, values, top_k=8)

        self.assertTrue(torch.all(result.tail_mass <= epsilon + 1e-6))
        self.assertEqual(result.indices.shape[-1], result.key_subset.shape[-2])

    def test_delta_kv_low_rank_handles_small_updates(self):
        torch.manual_seed(1)
        batch, heads, q_len, k_len, dim = 1, 2, 3, 4, 8
        queries = torch.randn(batch, heads, q_len, dim)
        base_keys = torch.randn(batch, heads, k_len, dim)
        base_values = torch.randn(batch, heads, k_len, dim)
        indices = torch.arange(k_len).view(1, 1, 1, k_len).expand(batch, heads, q_len, k_len)

        keys = base_keys.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
        values = base_values.unsqueeze(2).expand(-1, -1, q_len, -1, -1)

        logits_scale = 1.0 / math.sqrt(dim)
        logits = torch.einsum('bhqd,bhqkd->bhqk', queries, keys) * logits_scale
        attn_probs = torch.softmax(logits, dim=-1)

        positions = torch.arange(q_len)

        updater = DeltaKVLowRank(rank=2, tolerance=1e-2)
        updater.store(0, positions, queries, keys, values, indices, logits, attn_probs)

        delta = 1e-4
        queries_perturbed = queries + delta * torch.randn_like(queries)
        keys_perturbed = keys + delta * torch.randn_like(keys)
        values_perturbed = values + delta * torch.randn_like(values)

        result = updater.try_update(0, positions, queries_perturbed, keys_perturbed, values_perturbed, indices, logits_scale)
        self.assertIsNotNone(result)

        large_delta = 1.0
        queries_far = queries + large_delta * torch.randn_like(queries)
        res_far = updater.try_update(0, positions, queries_far, keys, values, indices, logits_scale)
        self.assertIsNone(res_far)

    def test_freeze_window_controller_schedule(self):
        controller = FreezeWindowController(batch_size=1, seq_len=4, device=torch.device("cpu"))
        margins = torch.tensor([[2.0, 4.0]])
        drift = torch.tensor([[0.1, 0.1]])
        positions = torch.tensor([[0, 2]])

        controller.update(margins, drift, positions, step=0)
        mask_step0 = controller.active_mask(step=0)
        self.assertFalse(mask_step0[0, 0])
        self.assertFalse(mask_step0[0, 2])

        mask_step25 = controller.active_mask(step=25)
        self.assertTrue(mask_step25[0, 0])
        self.assertTrue(mask_step25[0, 2])


if __name__ == "__main__":
    unittest.main()
