"""Evaluation + output parsing for HipTR (pure stdlib, no torch).

Provides:
  * ``cer`` / ``wer`` + corpus aggregates (Levenshtein edit distance),
  * ``transcription`` — strip layout markup to plain reading-order text (for CER),
  * ``parse_regions`` — turn the model's ``<page><region>…`` output into structured
    regions (type, polygon, text),
  * ``to_page_xml`` — export those regions to PAGE-XML (de-quantizing <loc_*> bins).
"""
from __future__ import annotations

import re
from typing import Dict, List

_LOC_RE = re.compile(r"<loc_(\d+)>")
_TYPE_BLOCK = re.compile(r"<type>.*?</type>", re.S)
_STRUCT_TAG = re.compile(r"</?(?:page|region|line|poly)>")
_REGION_RE = re.compile(r"<region>(.*?)</region>", re.S)
_TYPE_RE = re.compile(r"<type>(.*?)</type>", re.S)
_POLY_RE = re.compile(r"<poly>(.*?)</poly>", re.S)


# --- edit distance & rates ------------------------------------------------
def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return la or lb
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def corpus_cer(refs: List[str], hyps: List[str]) -> float:
    dist = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return dist / total if total else 0.0


def corpus_wer(refs: List[str], hyps: List[str]) -> float:
    dist = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return dist / total if total else 0.0


# --- output parsing -------------------------------------------------------
def transcription(s: str) -> str:
    """Strip layout markup (<page>/<region>/<type>/<poly>/<line>, <loc_*>) to the
    plain reading-order transcription — the right text for CER/WER."""
    s = _TYPE_BLOCK.sub(" ", s)
    s = _LOC_RE.sub("", s)
    s = _STRUCT_TAG.sub("\n", s)
    return "\n".join(ln.strip() for ln in s.splitlines() if ln.strip())


def parse_regions(s: str) -> List[Dict]:
    """Parse ``<page><region>…`` output into ``[{type, points, text}]`` in reading order.

    ``points`` are quantized ``<loc_*>`` bin pairs (use ``to_page_xml`` to de-quantize).
    """
    out: List[Dict] = []
    for body in _REGION_RE.findall(s):
        tm = _TYPE_RE.search(body)
        rtype = tm.group(1).strip() if tm else ""
        pm = _POLY_RE.search(body)
        if pm:
            bins = [int(x) for x in _LOC_RE.findall(pm.group(1))]
            rest = body[pm.end():]
        else:
            bins, rest = [], _TYPE_RE.sub("", body)
        points = [(bins[i], bins[i + 1]) for i in range(0, len(bins) - 1, 2)]
        out.append({"type": rtype, "points": points, "text": transcription(rest)})
    return out


# --- export ---------------------------------------------------------------
def _dequant(b: int, size: float, num_bins: int) -> int:
    return int(round(b / (num_bins - 1) * size))


def _xml_escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_page_xml(regions: List[Dict], image_w: int, image_h: int, num_bins: int = 1000) -> str:
    """Export parsed regions to minimal PAGE-XML (coords de-quantized to pixels)."""
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<PcGts xmlns="{ns}">',
        f'  <Page imageWidth="{image_w}" imageHeight="{image_h}">',
    ]
    for i, region in enumerate(regions):
        pts = " ".join(
            f"{_dequant(x, image_w, num_bins)},{_dequant(y, image_h, num_bins)}"
            for x, y in region["points"]
        )
        rows.append(f'    <TextRegion id="r{i}" type="{region.get("type", "")}">')
        if pts:
            rows.append(f'      <Coords points="{pts}"/>')
        rows.append(f'      <TextEquiv><Unicode>{_xml_escape(region["text"])}</Unicode></TextEquiv>')
        rows.append('    </TextRegion>')
    rows += ['  </Page>', '</PcGts>']
    return "\n".join(rows)
