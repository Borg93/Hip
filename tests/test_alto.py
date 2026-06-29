"""Pure-Python smoke tests (no torch/transformers required).

Run directly:  python tests/test_alto.py
Or with pytest: pytest tests/test_alto.py
"""
import os
import sys

# make src/ importable without installing the package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hiptr.config import HipTRConfig, TokenConfig  # noqa: E402
from hiptr.data.alto import parse_alto, quantize  # noqa: E402
from hiptr.data.tokens import all_special_tokens, location_tokens  # noqa: E402

SAMPLE = os.path.join(ROOT, "tests", "sample.alto.xml")


def test_quantize_bounds():
    assert quantize(0, 1000, 1000) == 0
    assert quantize(1000, 1000, 1000) == 999
    assert quantize(500, 1000, 1000) == 500  # midpoint -> 499/500 region
    assert quantize(-50, 1000, 1000) == 0  # clamped
    assert quantize(50, 0, 1000) == 0  # zero-size guard


def test_alto_line_format():
    out = parse_alto(SAMPLE, num_bins=1000, granularity="line")
    # two lines, each wrapped and prefixed by four loc tokens
    assert out.count("<line>") == 2
    assert out.count("</line>") == 2
    assert "der Briefträger kam" in out
    assert "am Morgen" in out
    # first line bbox x0 = round(100/1000*999) = 100 ; y0 = round(200/1400*999) = 143
    assert "<line><loc_100><loc_143>" in out
    print("line:", out)


def test_alto_word_format():
    out = parse_alto(SAMPLE, num_bins=1000, granularity="word")
    assert out.startswith("<loc_100><loc_143>der")
    assert "Morgen" in out
    print("word:", out)


def test_alto_polygon_format():
    out = parse_alto(SAMPLE, num_bins=1000, granularity="polygon")
    assert out.count("<poly>") == 2 and out.count("</poly>") == 2
    assert out.count("<line>") == 2
    # l1 polygon first vertex (100,200) -> <loc_100><loc_143>; 4 vertices -> 8 loc tokens
    assert out.startswith("<line><poly><loc_100><loc_143><loc_699><loc_143>")
    assert "</poly>der Briefträger kam</line>" in out
    assert "am Morgen</line>" in out
    print("polygon:", out)


def test_alto_polygon_default_and_subsample():
    # polygon is the default granularity
    assert parse_alto(SAMPLE) == parse_alto(SAMPLE, granularity="polygon")
    # subsampling caps vertices: 4-pt polygon -> 2 pts -> 4 loc tokens per line
    capped = parse_alto(SAMPLE, granularity="polygon", poly_max_points=2)
    assert capped.count("<loc_") == 2 * 2 * 2  # 2 lines * 2 pts * (x,y)


def test_special_tokens():
    locs = location_tokens(1000)
    assert locs[0] == "<loc_0>" and locs[-1] == "<loc_999>" and len(locs) == 1000
    toks = all_special_tokens(TokenConfig())
    for t in ("<image>", "<line>", "</line>", "<poly>", "</poly>"):
        assert t in toks
    assert len(toks) == 1005  # 5 structural + 1000 loc


def test_tokens_per_tile():
    cfg = HipTRConfig()
    assert cfg.vision_input.mode == "native"   # new default
    assert cfg.divisor == 28                    # patch 14 * pixel_shuffle 2
    # single @896 -> (896/28)^2 = 32^2 = 1024
    cfg.vision_input.mode = "single"
    assert cfg.tokens_per_tile == 1024
    # anyres tile 448 -> (448/28)^2 = 16^2 = 256
    cfg.vision_input.mode = "anyres"
    assert cfg.tokens_per_tile == 256


def test_grid_tokens_rectangular():
    cfg = HipTRConfig()
    # a 1372x896 native unit -> (1372/28)*(896/28) = 49*32 = 1568
    assert cfg.grid_tokens(1372, 896) == 49 * 32
    # non-divisible sizes are rejected
    raised = False
    try:
        cfg.grid_tokens(1000, 896)  # 1000 not divisible by 28
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
