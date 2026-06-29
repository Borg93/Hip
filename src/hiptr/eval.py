"""Evaluation + output parsing for HipTR (pure stdlib, no torch).

Provides:
  * ``cer`` / ``wer`` and corpus-level aggregates (Levenshtein edit distance),
  * ``parse_output`` — reverse of ``data/alto.py``: turn the model's
    ``<line><poly>…</poly>text</line>`` string back into structured lines,
  * ``page_text`` — concatenated transcription for page-level CER,
  * ``to_page_xml`` — export predictions to PAGE-XML (de-quantizing <loc_*> bins).

Kept dependency-free so metrics/parsing can be unit-tested without torch.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_LINE_RE = re.compile(r"<line>(.*?)</line>", re.S)
_POLY_RE = re.compile(r"<poly>(.*?)</poly>", re.S)
_LOC_RE = re.compile(r"<loc_(\d+)>")


# --- edit distance & rates ------------------------------------------------
def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def cer(ref: str, hyp: str) -> float:
    """Character error rate = edit_distance(chars) / len(ref chars)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Word error rate = edit_distance(words) / num ref words."""
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def corpus_cer(refs: List[str], hyps: List[str]) -> float:
    """Aggregate CER = total edit distance / total reference characters."""
    dist = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return dist / total if total else 0.0


def corpus_wer(refs: List[str], hyps: List[str]) -> float:
    dist = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total = sum(len(r.split()) for r in refs)
    return dist / total if total else 0.0


# --- output parsing -------------------------------------------------------
def parse_output(s: str) -> List[Dict]:
    """Parse ``<line>[<poly>locs</poly>]text</line>…`` into structured lines.

    Returns a list (in reading order) of ``{"points": [(xbin, ybin), …], "text": str}``.
    Handles polygon, line-bbox, and bare formats. Coordinates are the quantized
    ``<loc_*>`` bin indices (use ``to_page_xml`` to de-quantize to pixels).
    """
    out: List[Dict] = []
    blocks = _LINE_RE.findall(s)
    if not blocks:
        return out
    for body in blocks:
        pm = _POLY_RE.search(body)
        if pm:
            bins = [int(x) for x in _LOC_RE.findall(pm.group(1))]
            text = body[pm.end():]
        else:
            bins = [int(x) for x in _LOC_RE.findall(body)]
            text = _LOC_RE.sub("", body)
        pts = [(bins[i], bins[i + 1]) for i in range(0, len(bins) - 1, 2)]
        out.append({"points": pts, "text": text.strip()})
    return out


def page_text(parsed: List[Dict], sep: str = "\n") -> str:
    """Concatenated transcription (reading order) for page-level CER."""
    return sep.join(line["text"] for line in parsed)


# --- export ---------------------------------------------------------------
def _dequant(b: int, size: float, num_bins: int) -> int:
    return int(round(b / (num_bins - 1) * size))


def to_page_xml(parsed: List[Dict], image_w: int, image_h: int, num_bins: int = 1000) -> str:
    """Export parsed lines to minimal PAGE-XML (coords de-quantized to pixels)."""
    ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    rows = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<PcGts xmlns="{ns}">',
        f'  <Page imageWidth="{image_w}" imageHeight="{image_h}">',
        '    <TextRegion id="r1">',
    ]
    for i, line in enumerate(parsed):
        pts = " ".join(
            f"{_dequant(x, image_w, num_bins)},{_dequant(y, image_h, num_bins)}"
            for x, y in line["points"]
        )
        text = (line["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        rows += [
            f'      <TextLine id="l{i}">',
            f'        <Coords points="{pts}"/>',
            f'        <TextEquiv><Unicode>{text}</Unicode></TextEquiv>',
            '      </TextLine>',
        ]
    rows += ['    </TextRegion>', '  </Page>', '</PcGts>']
    return "\n".join(rows)
