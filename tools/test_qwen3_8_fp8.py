"""Focused tests for FP8 checkpoint conversion and independent reference math."""

import unittest
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from convert_qwen3_8_fp8_to_gguf import transform_language
from qwen3_8_fp8_infer import fp32_attention, fp8_linear, prefill_ranges, register_fp32_attention


class Fp8ConversionTest(unittest.TestCase):
    def test_registered_attention_preserves_cached_prefix_in_causal_mask(self):
        register_fp32_attention()
        config = Qwen3_5TextConfig(
            hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, layer_types=["full_attention"],
        )
        config._attn_implementation = "fp32_sdpa"
        cache = DynamicCache(config=config)
        history = torch.zeros(1, 1, 3, 8)
        cache.update(history, history, 0)
        mask = create_causal_mask(
            config=config, inputs_embeds=torch.zeros(1, 2, 16),
            attention_mask=None, past_key_values=cache,
        )
        self.assertIsNotNone(mask)
        expected = torch.tensor([[[[True, True, True, True, False],
                                   [True, True, True, True, True]]]])
        torch.testing.assert_close(mask, expected, rtol=0, atol=0)

    def test_preserves_recurrent_log_decay_without_cpu_exponentiation(self):
        config = {
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        }
        source = torch.tensor([-4, -2, -1, 0, 1, 2], dtype=torch.bfloat16)
        actual = transform_language("model.language_model.layers.0.linear_attn.A_log",
                                    source, config, False)
        torch.testing.assert_close(actual, source.reshape(2, 3).T.flatten(), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "FP32 efficient SDPA requires CUDA")
    def test_attention_uses_float_products_and_accepts_bfloat_masks(self):
        query = ((torch.arange(24, device="cuda").reshape(1, 1, 3, 8) % 13 - 6) / 8).bfloat16()
        key = ((torch.arange(32, device="cuda").reshape(1, 1, 4, 8) % 11 - 5) / 8).bfloat16()
        value = ((torch.arange(32, device="cuda").reshape(1, 1, 4, 8) % 7 - 3) / 8).bfloat16()
        mask = torch.tensor([[0, 0, -3, -10]] * 3, device="cuda").bfloat16()
        scores = query.float() @ key.float().transpose(-2, -1) * 0.25 + mask.float()
        expected = (scores.softmax(-1) @ value.float()).transpose(1, 2).bfloat16()
        actual, _ = fp32_attention(
            SimpleNamespace(is_causal=False), query, key, value, mask, scaling=0.25
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cached_causal_attention_rejects_missing_history_mask(self):
        query = torch.zeros(1, 2, 2, 8)
        key = torch.zeros(1, 1, 5, 8)
        with self.assertRaisesRegex(ValueError, "historical KV"):
            fp32_attention(SimpleNamespace(is_causal=True), query, key, key, None)

    @unittest.skipUnless(torch.cuda.is_available(), "FP32 efficient SDPA requires CUDA")
    def test_grouped_decode_works_without_math_backend_fallback(self):
        query = ((torch.arange(128, device="cuda").reshape(1, 4, 1, 32) % 17 - 8) / 16).bfloat16()
        key = ((torch.arange(608, device="cuda").reshape(1, 1, 19, 32) % 13 - 6) / 16).bfloat16()
        value = ((torch.arange(608, device="cuda").reshape(1, 1, 19, 32) % 11 - 5) / 16).bfloat16()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            expected = torch.nn.functional.scaled_dot_product_attention(
                query.float(), key.float().repeat_interleave(4, dim=1),
                value.float().repeat_interleave(4, dim=1),
            ).transpose(1, 2).bfloat16()
            actual, _ = fp32_attention(
                SimpleNamespace(is_causal=True, num_key_value_groups=4),
                query, key, value, None,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefill_stops_at_image_boundaries_and_limits_each_batch(self):
        self.assertEqual(prefill_ranges(520, -1, 0), [(0, 512), (512, 520)])
        self.assertEqual(
            prefill_ranges(2766, 4, 2752),
            [(0, 4), (4, 516), (516, 1028), (1028, 1540),
             (1540, 2052), (2052, 2564), (2564, 2756), (2756, 2766)],
        )
        with self.assertRaises(ValueError):
            prefill_ranges(20, 10, 11)

    def test_reorders_codes_and_both_scale_axes_without_changing_weights(self):
        config = {
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        }
        for projection, shape in (
            ("in_proj_qkv", (1280, 256)),
            ("in_proj_z", (768, 256)),
            ("out_proj", (256, 768)),
        ):
            with self.subTest(projection=projection):
                codes = (torch.arange(shape[0] * shape[1]) % 127).to(torch.uint8).reshape(shape)
                scales = torch.arange(shape[0] // 128 * (shape[1] // 128)).reshape(
                    shape[0] // 128, shape[1] // 128
                ).float() + 1
                name = f"model.language_model.layers.0.linear_attn.{projection}.weight"
                reordered_codes = transform_language(name, codes, config, False)
                reordered_scales = transform_language(name + "_scale_inv", scales, config, True)
                original = codes.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
                expected = transform_language(name, original, config, False)
                actual = reordered_codes.float() * reordered_scales.repeat_interleave(
                    128, 0
                ).repeat_interleave(128, 1)
                self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP8 reference matmul")
    def test_fp8_reference_preserves_small_products_across_cancellation(self):
        weight = torch.zeros(128, 128, device="cuda")
        weight[:, 0] = 448
        weight[:, 32] = 1 / 512
        weight[:, 64] = -448
        projection = SimpleNamespace(
            weight=weight.to(torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(1, 1, device="cuda"),
            bias=None,
        )
        for rows in (1, 4, 17, 65):
            with self.subTest(rows=rows):
                inputs = weight[:rows].abs()
                actual = fp8_linear(projection, inputs)
                expected = torch.full((rows, 128), 1 / 262144, device="cuda")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fp8_linear(projection, torch.full((1, 128), float("nan"), device="cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP8 reference matmul")
    def test_fp8_reference_applies_independent_scales_and_rounds_halfway_to_even(self):
        class Projection:
            pass

        projection = Projection()
        projection.weight = torch.ones(256, 256, device="cuda", dtype=torch.float8_e4m3fn)
        projection.weight_scale_inv = torch.tensor([[2, 3], [5, 7]], device="cuda")
        projection.bias = None
        inputs = torch.zeros(1, 256, device="cuda")
        inputs[0, 0] = 448
        inputs[0, 1] = 1.0625
        inputs[0, 2] = -1.1875
        inputs[0, 128] = 224
        result = fp8_linear(projection, inputs)
        # First block scale=1, codes 448,1,-1.25; second scale=0.5, code448.
        expected = torch.tensor([447.75 * 2 + 224 * 3] * 128 +
                                [447.75 * 5 + 224 * 7] * 128, device="cuda")
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
        inputs.zero_()
        inputs[0, 0] = 2.40625
        inputs[0, 1] = -0.2470703125
        result = fp8_linear(projection, inputs)
        # A rounded scalar reciprocal produces code -44, not the exact-division tie -48.
        scale = torch.tensor(0.005371094215661287, device="cuda")
        expected = 404 * scale * torch.tensor([2] * 128 + [5] * 128, device="cuda")
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
