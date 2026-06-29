"""Eval-metric and page-output parsing tests (pure Python; no torch)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hiptr.data.alto import parse_page  # noqa: E402
from hiptr.eval import (  # noqa: E402
    cer,
    corpus_cer,
    edit_distance,
    parse_regions,
    to_page_xml,
    transcription,
    wer,
)

PAGE = os.path.join(ROOT, "tests", "sample.page.xml")


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0


def test_cer_wer():
    assert cer("abcd", "abxd") == 0.25
    assert abs(wer("a b c", "a x c") - 1 / 3) < 1e-9
    assert cer("", "") == 0.0


def test_corpus_cer_aggregates_by_chars():
    assert corpus_cer(["abcd", "ef"], ["abxd", "ef"]) == 1 / 6


def test_transcription_strips_layout_markup():
    out = parse_page(PAGE, output="page")
    # region types, polygons and loc tokens removed; reading-order text remains
    assert transcription(out) == "der Briefträger kam\nam Morgen\nKapitel Ett"


def test_parse_regions():
    regions = parse_regions(parse_page(PAGE, output="page"))
    assert len(regions) == 2
    assert regions[0]["type"] == "paragraph"
    assert regions[0]["text"] == "der Briefträger kam\nam Morgen"
    assert regions[0]["points"][0] == (100, 143)   # first region-polygon vertex (bins)


def test_to_page_xml_dequantizes():
    regions = parse_regions(parse_page(PAGE, output="page"))
    xml = to_page_xml(regions, 1000, 1400)
    assert 'type="paragraph"' in xml
    assert "100,200" in xml                         # bin (100,143) -> px (100,200)
    assert "der Briefträger kam" in xml


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok: {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
