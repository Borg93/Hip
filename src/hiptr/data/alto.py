"""ALTO-XML -> training-target serialization (pure stdlib, no torch).

Produces the §4 output format from DESIGN.md. Like Surya-OCR-2, each text line is
emitted with its **geometry + transcription**, in **reading order** (the document
order of the lines):

  polygon granularity (recommended; Surya-style):
    <line><poly><loc_x0><loc_y0>...<loc_xn><loc_yn></poly>transcription</line> ...

  line granularity (axis-aligned bbox):
    <line><loc_x0><loc_y0><loc_x1><loc_y1>transcription</line> ...

  word granularity (the placeholder's format):
    <loc_x><loc_y>word ...

Polygons come from ALTO ``<Shape><Polygon POINTS=...>`` or PAGE-XML
``<Coords points=...>``; if absent we fall back to the bbox rectangle. Coordinates
are normalized to the page size and quantized to ``num_bins`` bins so they can be
emitted as the atomic ``<loc_*>`` tokens from ``tokens.py``.
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


def _parse_points(s: str):
    """'x1,y1 x2,y2 ...' or 'x1 y1 x2 y2 ...' -> [(x, y), ...]."""
    nums = []
    for tok in s.replace(",", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _polygon_points(line_el: ET.Element):
    """Polygon vertices from ALTO <Shape><Polygon POINTS> or PAGE <Coords points>."""
    for el in line_el.iter():
        if _localname(el.tag) in ("Polygon", "Coords"):
            pts = el.get("POINTS") or el.get("points")
            if pts:
                parsed = _parse_points(pts)
                if parsed:
                    return parsed
    return None


def _subsample(points, max_points: int):
    """Uniformly subsample a polygon to at most ``max_points`` vertices (0 = keep all)."""
    if max_points and len(points) > max_points:
        stride = len(points) / max_points
        idx = sorted({min(len(points) - 1, int(round(i * stride))) for i in range(max_points)})
        points = [points[i] for i in idx]
    return points


def _loc(x: float, y: float, pw: float, ph: float, num_bins: int) -> str:
    return f"<loc_{quantize(x, pw, num_bins)}><loc_{quantize(y, ph, num_bins)}>"


def _line_text(line: ET.Element) -> str:
    words = [w.get("CONTENT") for w in _findall_local(line, "String") if w.get("CONTENT")]
    return " ".join(words).strip()


def _line_bbox(line: ET.Element):
    """Line bbox; derived from its words if the line element lacks HPOS/VPOS."""
    bb = _bbox(line)
    if bb is None:
        wbbs = [b for b in (_bbox(w) for w in _findall_local(line, "String")) if b]
        if wbbs:
            bb = (min(b[0] for b in wbbs), min(b[1] for b in wbbs),
                  max(b[2] for b in wbbs), max(b[3] for b in wbbs))
    return bb


def parse_alto(xml_path: str, num_bins: int = 1000, granularity: str = "polygon",
               poly_max_points: int = 0) -> str:
    root = ET.parse(xml_path).getroot()
    return serialize_alto_root(root, num_bins, granularity, poly_max_points)


def serialize_alto_root(root: ET.Element, num_bins: int = 1000, granularity: str = "polygon",
                        poly_max_points: int = 0) -> str:
    pw, ph = _page_size(root)
    out: List[str] = []

    if granularity == "word":
        for line in _findall_local(root, "TextLine"):
            for word in _findall_local(line, "String"):
                content = word.get("CONTENT")
                bb = _bbox(word)
                if content is None or bb is None:
                    continue
                out.append(f"{_loc(bb[0], bb[1], pw, ph, num_bins)}{content}")
        return " ".join(out)

    # line-level geometry + transcription, in reading order (document order)
    for line in _findall_local(root, "TextLine"):
        text = _line_text(line)
        if not text:
            continue

        if granularity == "polygon":
            pts = _polygon_points(line)
            if pts is None:
                bb = _line_bbox(line)
                if bb is None:
                    continue
                x0, y0, x1, y1 = bb
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # bbox as a 4-pt polygon
            pts = _subsample(pts, poly_max_points)
            geom = "".join(_loc(x, y, pw, ph, num_bins) for x, y in pts)
            out.append(f"<line><poly>{geom}</poly>{text}</line>")

        elif granularity == "line":
            bb = _line_bbox(line)
            if bb is None:
                continue
            geom = _loc(bb[0], bb[1], pw, ph, num_bins) + _loc(bb[2], bb[3], pw, ph, num_bins)
            out.append(f"<line>{geom}{text}</line>")

        else:
            raise ValueError(f"unknown granularity {granularity!r}")

    return "".join(out)
