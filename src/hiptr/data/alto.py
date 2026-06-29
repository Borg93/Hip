"""ALTO / PAGE-XML -> end-to-end full-page training target (pure stdlib, no torch).

The model reads a whole page and emits its **layout + reading order + transcription**
in one sequence. A page is a list of regions, in reading order; each region carries
a type and geometry, then its text:

  output="page" (default, end-to-end):
    <page>
      <region><type>paragraph</type><poly>locs…</poly>text … text</region>
      <region><type>marginalia</type><poly>locs…</poly>text</region>
    </page>

  output="lines"  -> flat regions' lines with geometry (no page/region wrappers)
  output="text"   -> plain transcription only (reading order), no geometry/layout

Geometry comes from PAGE ``<Coords points>`` or ALTO ``<Shape><Polygon POINTS>`` /
``HPOS/VPOS/WIDTH/HEIGHT``. Reading order uses PAGE ``<ReadingOrder>`` when present,
otherwise document order. Region type uses PAGE ``@type`` / ``@custom`` (else "text").
Coordinates are normalized to the page and quantized to ``num_bins`` ``<loc_*>`` tokens.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

_REGION_TAGS = ("TextRegion", "TextBlock")


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _findall_local(root: ET.Element, name: str) -> List[ET.Element]:
    return [el for el in root.iter() if _localname(el.tag) == name]


def quantize(value: float, size: float, num_bins: int) -> int:
    if size <= 0:
        return 0
    b = int(round((value / size) * (num_bins - 1)))
    return max(0, min(num_bins - 1, b))


def _loc(x: float, y: float, pw: float, ph: float, num_bins: int) -> str:
    return f"<loc_{quantize(x, pw, num_bins)}><loc_{quantize(y, ph, num_bins)}>"


def _parse_points(s: str):
    nums = []
    for tok in s.replace(",", " ").split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _subsample(points, max_points: int):
    if max_points and len(points) > max_points:
        stride = len(points) / max_points
        idx = sorted({min(len(points) - 1, int(round(i * stride))) for i in range(max_points)})
        points = [points[i] for i in idx]
    return points


# --- page geometry & structure -------------------------------------------
def _page_size(root: ET.Element):
    for page in _findall_local(root, "Page"):
        w = page.get("WIDTH") or page.get("imageWidth")
        h = page.get("HEIGHT") or page.get("imageHeight")
        if w and h:
            return float(w), float(h)
    for ps in _findall_local(root, "PrintSpace"):
        if ps.get("WIDTH") and ps.get("HEIGHT"):
            return float(ps.get("WIDTH")), float(ps.get("HEIGHT"))
    raise ValueError("no Page/PrintSpace dimensions found")


def _bbox(el: ET.Element):
    try:
        x, y = float(el.get("HPOS")), float(el.get("VPOS"))
        w, h = float(el.get("WIDTH")), float(el.get("HEIGHT"))
    except (TypeError, ValueError):
        return None
    return x, y, x + w, y + h


def _own_geom_points(el: ET.Element):
    """Polygon vertices from this element's OWN Coords/Shape (not its children's)."""
    for child in el:
        ln = _localname(child.tag)
        if ln == "Coords":
            pts = child.get("points") or child.get("POINTS")
            if pts:
                return _parse_points(pts)
        if ln == "Shape":
            for poly in child.iter():
                if _localname(poly.tag) == "Polygon":
                    pts = poly.get("POINTS") or poly.get("points")
                    if pts:
                        return _parse_points(pts)
    return None


def _geom_points(el: ET.Element):
    pts = _own_geom_points(el)
    if pts:
        return pts
    bb = _bbox(el)
    if bb:
        x0, y0, x1, y1 = bb
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return None


def _region_type(region: ET.Element) -> str:
    t = region.get("type")
    if t:
        return t
    custom = region.get("custom")
    if custom:
        m = re.search(r"type\s*:\s*([A-Za-z0-9_-]+)", custom)
        if m:
            return m.group(1)
    return "text"


def _line_text(line: ET.Element) -> str:
    for te in line:  # PAGE: line-level <TextEquiv><Unicode>
        if _localname(te.tag) == "TextEquiv":
            for u in te.iter():
                if _localname(u.tag) == "Unicode" and u.text:
                    return u.text.strip()
    words = [w.get("CONTENT") for w in _findall_local(line, "String") if w.get("CONTENT")]
    return " ".join(words).strip()


def _reading_order_map(root: ET.Element):
    order = {}
    for r in root.iter():
        if _localname(r.tag) == "RegionRefIndexed":
            ref, idx = r.get("regionRef"), r.get("index")
            if ref is not None and idx is not None:
                try:
                    order[ref] = int(idx)
                except ValueError:
                    pass
    return order


def _ordered_regions(root: ET.Element) -> List[ET.Element]:
    regions = [el for el in root.iter() if _localname(el.tag) in _REGION_TAGS]
    order = _reading_order_map(root)
    if order:
        regions.sort(key=lambda r: order.get(r.get("id"), len(order) + 1))
    return regions


# --- serialization --------------------------------------------------------
def _emit_geom(el, mode, pw, ph, num_bins, poly_max_points) -> str:
    if mode == "none":
        return ""
    pts = _geom_points(el)
    if not pts:
        return ""
    if mode == "bbox":
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return _loc(min(xs), min(ys), pw, ph, num_bins) + _loc(max(xs), max(ys), pw, ph, num_bins)
    pts = _subsample(pts, poly_max_points)
    return "<poly>" + "".join(_loc(x, y, pw, ph, num_bins) for x, y in pts) + "</poly>"


def _emit_line(line, mode, pw, ph, num_bins, poly_max_points) -> str:
    text = _line_text(line)
    if not text:
        return ""
    return f"<line>{_emit_geom(line, mode, pw, ph, num_bins, poly_max_points)}{text}</line>"


def parse_page(
    xml_path: str,
    num_bins: int = 1000,
    output: str = "page",
    region_geometry: str = "poly",
    line_geometry: str = "none",
    include_region_type: bool = True,
    poly_max_points: int = 0,
) -> str:
    root = ET.parse(xml_path).getroot()
    return serialize_page(
        root, num_bins, output, region_geometry, line_geometry, include_region_type, poly_max_points
    )


def serialize_page(
    root: ET.Element,
    num_bins: int = 1000,
    output: str = "page",
    region_geometry: str = "poly",
    line_geometry: str = "none",
    include_region_type: bool = True,
    poly_max_points: int = 0,
) -> str:
    pw, ph = _page_size(root)
    regions = _ordered_regions(root)

    if output == "text":
        texts = [_line_text(ln) for r in regions for ln in _findall_local(r, "TextLine")]
        return "\n".join(t for t in texts if t)

    if output == "lines":
        mode = line_geometry if line_geometry != "none" else "poly"
        parts = [
            _emit_line(ln, mode, pw, ph, num_bins, poly_max_points)
            for r in regions
            for ln in _findall_local(r, "TextLine")
        ]
        return "".join(p for p in parts if p)

    if output != "page":
        raise ValueError(f"unknown output mode {output!r}")

    parts = ["<page>"]
    for region in regions:
        lines = _findall_local(region, "TextLine")
        if not any(_line_text(ln) for ln in lines):
            continue
        parts.append("<region>")
        if include_region_type:
            parts.append(f"<type>{_region_type(region)}</type>")
        parts.append(_emit_geom(region, region_geometry, pw, ph, num_bins, poly_max_points))
        if line_geometry == "none":
            parts.append("\n".join(t for t in (_line_text(ln) for ln in lines) if t))
        else:
            parts += [_emit_line(ln, line_geometry, pw, ph, num_bins, poly_max_points) for ln in lines]
        parts.append("</region>")
    parts.append("</page>")
    return "".join(parts)
