"""Configuration dataclasses for HipTR.

Everything that the placeholder hardcoded (1024 dims, 1024px, the model ids) lives
here instead, so the same code runs with Qwen3.5-0.8B or Qwen3-0.6B and any TIPS
variant. Dimensions that depend on a loaded model (e.g. the LLM hidden size) are
read from that model's config at build time, never hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VisionConfig:
    # TIPS variant: one of {"S","B","L","SO","g"}. L/14 is the recommended default.
    variant: str = "L"
    patch_size: int = 14
    embed_dim: int = 1024  # L/14 -> 1024 (B->768, SO->1152, g->1536, S->384)
    # Path to the vendored TIPS pytorch package and the .npz vision checkpoint.
    # See scripts/fetch_tips.sh (the repo cannot be cloned from this sandbox).
    tips_pkg_path: str = "third_party/tips/pytorch"
    checkpoint_path: Optional[str] = None  # e.g. ".../tips_v2_oss_l14_vision.npz"
    # TIPS preprocessing normalization. CONFIRM these against the cloned repo;
    # the (0.5, 0.5, 0.5) default is a placeholder, not gospel.
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    freeze: bool = True  # frozen by default (unfreeze only in the optional stage C)


@dataclass
class TilingConfig:
    tile_size: int = 448  # must be a multiple of patch_size * pixel_shuffle (14*2=28)
    max_tiles: int = 12  # AnyRes upper bound; the main quality/cost dial for HTR
    use_thumbnail: bool = True  # add a downsized global view for layout


@dataclass
class ConnectorConfig:
    pixel_shuffle: int = 2  # 2 -> visual tokens reduced 4x, feature dim x4 into the MLP
    mlp_depth: int = 2
    activation: str = "gelu"


@dataclass
class LLMConfig:
    # The smallest Qwen3.5 is 0.8B (there is NO Qwen3.5-0.6B). Use Qwen3-0.6B for a
    # literal 0.6B decoder. hidden_size is taken from the loaded model, not set here.
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


@dataclass
class DataConfig:
    img_dir: str = "./data/images"
    xml_dir: str = "./data/alto_xml"
    image_glob: str = "*.jpg"
    xml_glob: str = "*.xml"
    granularity: str = "line"  # "line" (recommended) or "word"
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
    tiling: TilingConfig = field(default_factory=TilingConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def tokens_per_tile(self) -> int:
        """Deterministic visual-token count per tile after pixel-shuffle.

        tile/patch gives the patch grid; pixel_shuffle squeezes it by that factor
        on each axis. With tile=448, patch=14, shuffle=2 -> (448/14/2)^2 = 256.
        """
        grid = self.tiling.tile_size // self.vision.patch_size
        assert grid % self.connector.pixel_shuffle == 0, (
            f"tile/patch grid ({grid}) must be divisible by pixel_shuffle "
            f"({self.connector.pixel_shuffle}); pick a tile_size that is a multiple "
            f"of patch_size*pixel_shuffle."
        )
        side = grid // self.connector.pixel_shuffle
        return side * side
