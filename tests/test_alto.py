"""ALTO/PAGE-XML -> end-to-end page target tests (pure Python; no torch)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hiptr.config import HipTRConfig, TokenConfig  # noqa: E402
from hiptr.data.alto import parse_page, quantize  # noqa: E402
from hiptr.data.tokens import all_special_tokens, location_tokens  # noqa: E402

PAGE = os.path.join(ROOT, "tests", "sample.page.xml")
ALTO = os.path.join(ROOT, "tests", "sample.alto.xml")


def test_quantize_bounds():
    assert quantize(0, 1000, 1000) == 0
    assert quantize(1000, 1000, 1000) == 999
    assert quantize(-50, 1000, 1000) == 0
    assert quantize(50, 0, 1000) == 0


def test_page_output_pagexml():
    out = parse_page(PAGE, output="page")
    # reading order honours <ReadingOrder>: r2 (paragraph) before r1 (heading)
    assert out.startswith("<page><region><type>paragraph</type><poly><loc_100><loc_143>")
    assert out.index("paragraph") < out.index("heading")
    assert out.count("<region>") == 2 and out.count("</region>") == 2
    assert "der Briefträger kam\nam Morgen" in out
    assert "Kapitel Ett" in out and out.endswith("</page>")
    print("page:", out)


def test_text_output_reading_order():
    # plain transcription only, in reading order (paragraph region first)
    assert parse_page(PAGE, output="text") == "der Briefträger kam\nam Morgen\nKapitel Ett"


def test_lines_output_has_polygons():
    out = parse_page(PAGE, output="lines")
    assert out.count("<line>") == 3 and "<line><poly>" in out
    assert "der Briefträger kam</line>" in out


def test_page_output_alto_fallback():
    # ALTO TextBlock -> region type "text"; geometry from line Shape/Polygon
    out = parse_page(ALTO, output="page")
    assert "<type>text</type>" in out
    assert "der Briefträger kam\nam Morgen" in out
    assert out.count("<region>") == 1


def test_special_tokens():
    locs = location_tokens(1000)
    assert locs[0] == "<loc_0>" and locs[-1] == "<loc_999>" and len(locs) == 1000
    toks = all_special_tokens(TokenConfig())
    for t in ("<image>", "<page>", "<region>", "<type>", "<line>", "<poly>"):
        assert t in toks
    assert len(toks) == 1011  # 11 structural + 1000 loc


def test_tokens_per_tile():
    cfg = HipTRConfig()
    assert cfg.vision_input.mode == "native"
    assert cfg.divisor == 28
    cfg.vision_input.mode = "single"
    assert cfg.tokens_per_tile == 1024            # (896/28)^2
    cfg.vision_input.mode = "anyres"
    assert cfg.tokens_per_tile == 256             # (448/28)^2


def test_grid_tokens_rectangular():
    cfg = HipTRConfig()
    assert cfg.grid_tokens(1372, 896) == 49 * 32  # (1372/28)*(896/28)
    raised = False
    try:
        cfg.grid_tokens(1000, 896)                # 1000 not divisible by 28
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
