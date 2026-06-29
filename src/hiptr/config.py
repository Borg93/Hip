"""Configuration dataclasses for HipTR.

Everything the placeholder hardcoded (1024 dims, 1024px, the model ids) lives here
instead, so the same code runs with Qwen3.5-0.8B and any TIPSv2 variant.
Dimensions that depend on a loaded model (e.g. the LLM hidden size) are read from
that model's config at build time, never hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VisionConfig:
    # TIPSv2 variant: one of {"B","L","SO","g"} (there is no "S" in v2).
    # L/14 (1024-dim) is the recommended default.
    variant: str = "L"
    patch_size: int = 14
    embed_dim: int = 1024  # B=768, L=1024, SO=1152, g=1536
    # TIPSv2 loads directly from HuggingFace as a DPT AutoModel with
    # trust_remote_code=True: google/tipsv2-{b14,l14,so400m14,g14}-dpt. The bare
    # vision encoder is dpt._backbone.vision_encoder. Auth for the (gated) repo via
    # env HF_TIPSv2 or HF_TOKEN. No local .npz/checkpoint wiring needed.
    # "intermediate" (get_intermediate_layers, the official dense path), "standard"
    # (plain forward), or "value" (value-attention dense feats).
    feature_mode: str = "intermediate"
    freeze: bool = True
    # TIPSv2 preprocessing is ToTensor-only -> pixels in [0,1], NO ImageNet
    # mean/std normalization. (See vision/preprocess.py.)


@dataclass
class VisionInputConfig:
    # "native" (default): aspect-preserving rectangular resize, each side snapped to
    #   patch_size*pixel_shuffle (the official TIPS dense recipe; no padding waste).
    # "single": one square pass at `resolution`.
    # "anyres": split the page into square tiles (very large / high-DPI pages).
    mode: str = "native"
    # native mode:
    native_target: int = 1372      # the longer side is resized toward this
    native_max_side: int = 1792    # clamp each side (keeps token counts sane)
    native_min_side: int = 56      # = 2 * patch_size * pixel_shuffle (>=1 shuffled token)
    # single mode:
    resolution: int = 896          # must be a multiple of patch_size*pixel_shuffle
    aspect: str = "pad"            # "pad" (preserve glyph aspect, letterbox) or "squish"
    # anyres mode:
    tile_size: int = 448           # multiple of patch_size * pixel_shuffle
    max_tiles: int = 12
    use_thumbnail: bool = True
    # TIPSv2's documented square resolution ladder (all multiples of patch size 14).
    supported_resolutions: Tuple[int, ...] = (224, 336, 448, 672, 896, 1120, 1372, 1792)


@dataclass
class ConnectorConfig:
    pixel_shuffle: int = 2  # 2 -> visual tokens reduced 4x, feature dim x4 into the MLP
    mlp_depth: int = 2
    activation: str = "gelu"


@dataclass
class LLMConfig:
    # Qwen3.5-0.8B: the smallest Qwen3.5 (there is no 0.6B in 3.5). Natively
    # multimodal, 262K context, hidden_size 1024 (matches TIPS L/14). hidden_size
    # is read from the loaded model, never set here.
    model_id: str = "Qwen/Qwen3.5-0.8B"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05


@dataclass
class TokenConfig:
    num_loc_bins: int = 1000  # coordinates quantized to <loc_0>..<loc_999>
    image_token: str = "<image>"
    line_open: str = "<line>"
    line_close: str = "</line>"
    poly_open: str = "<poly>"
    poly_close: str = "</poly>"


@dataclass
class DataConfig:
    img_dir: str = "./data/images"
    xml_dir: str = "./data/alto_xml"
    image_glob: str = "*.jpg"
    xml_glob: str = "*.xml"
    granularity: str = "polygon"  # "polygon" (recommended) | "line" (bbox) | "word"
    poly_max_points: int = 0      # >0 subsamples each polygon to at most this many points
    max_target_len: int = 2048


@dataclass
class TrainConfig:
    stage: str = "align"  # "align" | "sft" | "encoder"
    output_dir: str = "./checkpoints"
    epochs: int = 5
    batch_size: int = 2
    grad_accum: int = 8
    lr_projector: float = 1e-3
    lr_llm: float = 1e-5
    lr_vision: float = 5e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_workers: int = 4
    seed: int = 0


@dataclass
class HipTRConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    vision_input: VisionInputConfig = field(default_factory=VisionInputConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def divisor(self) -> int:
        """Pixels per shuffled visual token along one axis = patch_size * pixel_shuffle."""
        return self.vision.patch_size * self.connector.pixel_shuffle

    def grid_tokens(self, h_pix: int, w_pix: int) -> int:
        """Visual tokens for an image unit of ``h_pix x w_pix`` after pixel-shuffle."""
        d = self.divisor
        if h_pix % d or w_pix % d:
            raise ValueError(
                f"image unit {h_pix}x{w_pix} must have both sides divisible by "
                f"patch_size*pixel_shuffle ({d})."
            )
        return (h_pix // d) * (w_pix // d)

    @property
    def tokens_per_tile(self) -> int:
        """Fixed per-unit token count (single/anyres only; native varies per image).

        single@896, patch 14, shuffle 2 -> (896/28)^2 = 1024; anyres tile 448 ->
        (448/28)^2 = 256. In ``native`` mode token counts vary, so use
        ``grid_tokens(h, w)`` per image instead.
        """
        vi = self.vision_input
        if vi.mode == "single":
            side = vi.resolution
        elif vi.mode == "anyres":
            side = vi.tile_size
        else:
            raise ValueError(
                "tokens_per_tile is fixed only for single/anyres; native mode varies "
                "per image — use grid_tokens(h_pix, w_pix)."
            )
        return self.grid_tokens(side, side)
