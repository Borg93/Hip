"""Eval-metric and output-parsing tests (pure Python; no torch)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hiptr.data.alto import parse_alto  # noqa: E402
from hiptr.eval import (  # noqa: E402
    cer,
    corpus_cer,
    edit_distance,
    page_text,
    parse_output,
    to_page_xml,
    wer,
)

SAMPLE = os.path.join(ROOT, "tests", "sample.alto.xml")


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


def test_cer_wer():
    assert cer("abcd", "abxd") == 0.25            # 1 substitution / 4 chars
    assert abs(wer("a b c", "a x c") - 1 / 3) < 1e-9
    assert cer("", "") == 0.0


def test_corpus_cer_aggregates_by_chars():
    refs = ["abcd", "ef"]
    hyps = ["abxd", "ef"]
    assert corpus_cer(refs, hyps) == 1 / 6        # 1 error over 6 ref chars


def test_parse_output_polygon_roundtrip():
    lines = parse_output(parse_alto(SAMPLE, granularity="polygon"))
    assert len(lines) == 2
    assert lines[0]["text"] == "der Briefträger kam"
    assert lines[0]["points"][0] == (100, 143)    # first polygon vertex (quantized bins)
    assert len(lines[0]["points"]) == 4
    assert lines[1]["text"] == "am Morgen"


def test_parse_output_line_bbox():
    lines = parse_output(parse_alto(SAMPLE, granularity="line"))
    assert lines[0]["text"] == "der Briefträger kam"
    assert len(lines[0]["points"]) == 2           # bbox = 2 corner points


def test_page_text_reading_order():
    txt = page_text(parse_output(parse_alto(SAMPLE, granularity="polygon")))
    assert txt == "der Briefträger kam\nam Morgen"


def test_to_page_xml_dequantizes():
    xml = to_page_xml(parse_output(parse_alto(SAMPLE, granularity="polygon")), 1000, 1400)
    assert "<Coords points=" in xml and "der Briefträger kam" in xml
    assert "100,200" in xml                        # bin (100,143) -> px (100,200)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
