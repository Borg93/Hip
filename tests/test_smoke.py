"""End-to-end forward/backward smoke test on CPU with a tiny random LLM.

Validates the wiring the pure-Python tests can't cover: dummy vision -> connector
(pixel-unshuffle on a [B,C,H,W] map) -> masked scatter into <image> positions ->
LLM loss -> backward. Requires torch + transformers; skipped otherwise. No weights
are downloaded (the LLM is built from a tiny config).
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from hiptr.config import HipTRConfig  # noqa: E402
from hiptr.model.modeling_hiptr import HipTRForHTR  # noqa: E402


def _tiny():
    cfg = HipTRConfig()
    cfg.vision.embed_dim = 64          # DummyVisionEncoder uses this
    cfg.vision_input.mode = "single"
    cfg.vision_input.resolution = 56   # 56/28 = 2 -> 2x2 = 4 visual tokens
    llm_cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=512,
        max_position_embeddings=512,
    )
    model = HipTRForHTR(cfg, llm=LlamaForCausalLM(llm_cfg), image_token_id=511, dummy_vision=True)
    return cfg, model


def test_forward_backward_cpu():
    cfg, model = _tiny()
    n = cfg.tokens_per_tile                         # 4
    img = model.image_token_id
    ids = [1, 2] + [img] * n + [3, 4]
    labels = [-100, -100] + [-100] * n + [3, 4]     # loss only on the target
    out = model(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        pixel_values=[torch.rand(3, 56, 56)],
        labels=torch.tensor([labels]),
    )
    assert out.loss.requires_grad and torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None for p in model.connector.parameters())


def test_placeholder_count_mismatch_raises():
    cfg, model = _tiny()
    img = model.image_token_id
    ids = [1, 2] + [img] * 3 + [3]                  # 3 placeholders but unit yields 4 tokens
    with pytest.raises(ValueError):
        model(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            pixel_values=[torch.rand(3, 56, 56)],
            labels=torch.tensor([[-100] * len(ids)]),
        )
