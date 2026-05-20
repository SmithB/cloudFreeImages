"""CRS / coordinate-transformation helpers."""

from __future__ import annotations

from typing import Union

from pyproj import CRS, Transformer

# Type alias accepted wherever a CRS can be specified.
CRSInput = Union[int, str]   # e.g. 4326, "EPSG:3031", "WGS84"

# WGS-84 geographic (lon/lat) — what the STAC API expects.
_WGS84 = "EPSG:4326"


def _get_transformer(src_crs: CRSInput, dst_crs: CRSInput):
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    return Transformer.from_crs(src, dst, always_xy=True)


def transform_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: CRSInput,
    dst_crs: CRSInput = _WGS84,
) -> tuple[float, float, float, float]:
    """Transform *bbox* from *src_crs* to *dst_crs*.

    The returned envelope is the axis-aligned bounding box of the four
    input corners projected into *dst_crs*.  For highly curved projections
    (e.g. near the pole) this envelope is conservative but correct.

    Parameters
    ----------
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *src_crs* units.
    src_crs:
        An integer EPSG code, EPSG string, or any input accepted by
        ``pyproj.CRS.from_user_input``.
    dst_crs:
        Target CRS (default: ``EPSG:4326`` / WGS-84 lon/lat).

    Returns
    -------
    ``(min_x, min_y, max_x, max_y)`` in *dst_crs* units.
    """
    if CRS.from_user_input(src_crs) == CRS.from_user_input(dst_crs):
        return bbox

    transformer = _get_transformer(src_crs, dst_crs)

    min_x, min_y, max_x, max_y = bbox

    # Sample the four corners plus mid-edge points for better coverage near
    # the poles where meridian curvature is large.
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    xs = [min_x, min_x, max_x, max_x, mid_x, mid_x, min_x, max_x]
    ys = [min_y, max_y, min_y, max_y, min_y, max_y, mid_y, mid_y]

    new_xs, new_ys = transformer.transform(xs, ys)

    return (min(new_xs), min(new_ys), max(new_xs), max(new_ys))


def bbox_to_wgs84(
    bbox: tuple[float, float, float, float],
    src_crs: CRSInput,
) -> tuple[float, float, float, float]:
    """Convenience wrapper: transform *bbox* from *src_crs* to WGS-84."""
    return transform_bbox(bbox, src_crs, _WGS84)
