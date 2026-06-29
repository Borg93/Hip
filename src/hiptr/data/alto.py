"""ALTO-XML -> training-target serialization (pure stdlib, no torch).

Produces the §4 output format from DESIGN.md:

  line granularity (recommended):
    <line><loc_x0><loc_y0><loc_x1><loc_y1>transcription</line> ...

  word granularity (the placeholder's format):
    <loc_x><loc_y>word ...

Coordinates are normalized to the page size and quantized to ``num_bins`` bins so
they can be emitted as the atomic ``<loc_*>`` tokens from ``tokens.py``. Reading
order is the document order of <TextLine>/<String> elements.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional


def _localname(tag: str) -> str:
    """Strip the XML namespace: '{...}TextLine' -> 'TextLine'."""
    return tag.rsplit("}", 1)[-1]


def _findall_local(root: ET.Element, name: str) -> List[ET.Element]:
    return [el for el in root.iter() if _localname(el.tag) == name]


def quantize(value: float, size: float, num_bins: int) -> int:
    """Normalize ``value`` by ``size`` and quantize into ``[0, num_bins-1]``."""
    if size <= 0:
        return 0
    b = int(round((value / size) * (num_bins - 1)))
    return max(0, min(num_bins - 1, b))


def _page_size(root: ET.Element):
    for page in _findall_local(root, "Page"):
        w, h = page.get("WIDTH"), page.get("HEIGHT")
        if w and h:
            return float(w), float(h)
    # fall back to PrintSpace if Page lacks dimensions
    for ps in _findall_local(root, "PrintSpace"):
        w, h = ps.get("WIDTH"), ps.get("HEIGHT")
        if w and h:
            return float(w), float(h)
    raise ValueError("ALTO file has no Page/PrintSpace WIDTH/HEIGHT")


def _bbox(el: ET.Element):
    """(x0, y0, x1, y1) from HPOS/VPOS/WIDTH/HEIGHT, or None if missing."""
    try:
        x = float(el.get("HPOS"))
        y = float(el.get("VPOS"))
        w = float(el.get("WIDTH"))
        h = float(el.get("HEIGHT"))
    except (TypeError, ValueError):
        return None
    return x, y, x + w, y + h


def parse_alto(xml_path: str, num_bins: int = 1000, granularity: str = "line") -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return serialize_alto_root(root, num_bins=num_bins, granularity=granularity)


def serialize_alto_root(root: ET.Element, num_bins: int = 1000, granularity: str = "line") -> str:
    pw, ph = _page_size(root)
    out: List[str] = []

    if granularity == "word":
        for line in _findall_local(root, "TextLine"):
            for word in _findall_local(line, "String"):
                content = word.get("CONTENT")
                bb = _bbox(word)
                if content is None or bb is None:
                    continue
                x = quantize(bb[0], pw, num_bins)
                y = quantize(bb[1], ph, num_bins)
                out.append(f"<loc_{x}><loc_{y}>{content}")
        return " ".join(out)

    # line granularity (default)
    for line in _findall_local(root, "TextLine"):
        words = [w.get("CONTENT") for w in _findall_local(line, "String") if w.get("CONTENT")]
        text = " ".join(words).strip()
        if not text:
            continue
        bb = _bbox(line)
        if bb is None:
            # derive a line bbox from its words if the line lacks one
            wbbs = [b for b in (_bbox(w) for w in _findall_local(line, "String")) if b]
            if not wbbs:
                continue
            bb = (min(b[0] for b in wbbs), min(b[1] for b in wbbs),
                  max(b[2] for b in wbbs), max(b[3] for b in wbbs))
        x0 = quantize(bb[0], pw, num_bins)
        y0 = quantize(bb[1], ph, num_bins)
        x1 = quantize(bb[2], pw, num_bins)
        y1 = quantize(bb[3], ph, num_bins)
        out.append(f"<line><loc_{x0}><loc_{y0}><loc_{x1}><loc_{y1}>{text}</line>")
    return "".join(out)
