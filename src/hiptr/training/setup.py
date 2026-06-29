"""Model/tokenizer construction and LoRA wiring for training."""
from __future__ import annotations

from ..config import HipTRConfig
from ..data.tokens import add_htr_tokens
from ..model.modeling_hiptr import HipTRForHTR

_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model_and_tokenizer(cfg: HipTRConfig, dummy_vision: bool = False):
    """Load tokenizer + model, add HipTR tokens, set the stage's trainable params."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HipTRForHTR(cfg, tokenizer=tokenizer, dummy_vision=dummy_vision)
    add_htr_tokens(tokenizer, model.llm, cfg.tokens)
    model.set_trainable(cfg.train.stage)
    if cfg.train.stage == "sft" and cfg.llm.use_lora:
        apply_lora(model, cfg)
    return model, tokenizer


def apply_lora(model: HipTRForHTR, cfg: HipTRConfig) -> None:
    """Wrap the LLM in LoRA adapters; keep the connector fully trainable."""
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=cfg.llm.lora_r,
        lora_alpha=cfg.llm.lora_alpha,
        lora_dropout=cfg.llm.lora_dropout,
        target_modules=_LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora)
    for p in model.connector.parameters():
        p.requires_grad = True
