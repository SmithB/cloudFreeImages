"""ItsLive STAC API client.

Uses ``pystac_client`` to page through the ItsLive STAC catalog and yield
raw item dictionaries matching a bounding box and date range.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator, Optional

import pystac_client

from .crs_utils import CRSInput, bbox_to_wgs84  # pyproj only needed for non-WGS84 bbox
from .scene_parser import parse_scene_id

logger = logging.getLogger(__name__)

STAC_URL = "https://stac.itslive.cloud"
COLLECTION = "itslive-granules"

# Optical platform codes present in the ItsLive catalog.
_OPTICAL = {"LC04", "LT04", "LC05", "LT05", "LE07", "LC08", "LC09", "S2A", "S2B"}
_SAR = {"S1A", "S1B", "S1C"}


def _to_iso(dt: datetime | str) -> str:
    """Normalise a datetime or ISO string to a UTC ISO-8601 string."""
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_datetime(dt: datetime | str) -> datetime:
    """Normalise a datetime or ISO string to a UTC-aware datetime."""
    if isinstance(dt, datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    s = dt.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {dt!r}")


def _scene_in_window(
    scene_id: Optional[str],
    platform_hint: str,
    start_dt: datetime,
    end_dt: datetime,
) -> bool:
    """Return True if *scene_id*'s acquisition date falls within [start_dt, end_dt].

    Returns True when the scene ID is absent or unparseable so we never
    silently discard a pair we can't validate.
    """
    if not scene_id:
        return True
    parsed = parse_scene_id(scene_id, platform_hint)
    if parsed is None:
        return True
    return start_dt <= parsed.acquisition_date <= end_dt


def search_granules(
    bbox: tuple[float, float, float, float],
    start_date: datetime | str,
    end_date: datetime | str,
    bbox_crs: CRSInput = "EPSG:4326",
    optical_only: bool = True,
    max_dt_days: Optional[int] = None,
    min_valid_pixels: Optional[float] = 30.0,
    max_items: Optional[int] = None,
) -> Iterator[dict]:
    """Yield ItsLive STAC item dicts that fall within *bbox* and the date range.

    Parameters
    ----------
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *bbox_crs* units.
    start_date, end_date:
        Temporal bounds.  Either ``datetime`` objects or ISO-8601 strings.
    bbox_crs:
        CRS of *bbox*.  Accepts integer EPSG codes (e.g. ``3031``) or
        strings (e.g. ``"EPSG:3031"``).  Defaults to ``"EPSG:4326"``
        (WGS-84 lon/lat — no transformation performed).
    optical_only:
        When ``True`` (default) only Landsat and Sentinel-2 granules are
        returned.  A CQL2-JSON platform filter is sent to the server so SAR
        granules do not consume the ``max_items`` budget.
    max_dt_days:
        Discard pairs with more than this many days between acquisitions.
        Applied client-side via the ``date_dt`` property.
    min_valid_pixels:
        Discard pairs whose ``percent_valid_pixels`` is below this threshold.
        Defaults to 30.  Applied client-side (``percent_valid_pixels`` is not
        a CQL2-queryable on the ItsLive STAC server).  Pass ``None`` to
        disable.
    max_items:
        Stop after yielding this many items (useful for quick tests).
    """
    # The ItsLive STAC API always expects WGS-84 lon/lat.
    search_bbox = bbox_to_wgs84(bbox, bbox_crs) if str(bbox_crs) not in ("4326", "EPSG:4326") else bbox

    datetime_str = f"{_to_iso(start_date)}/{_to_iso(end_date)}"
    start_dt = _to_datetime(start_date)
    end_dt = _to_datetime(end_date)

    client = pystac_client.Client.open(STAC_URL)

    search_kwargs: dict = {
        "collections": [COLLECTION],
        "bbox": list(search_bbox),
        "datetime": datetime_str,
        "max_items": max_items,
    }

    # Server-side platform filter via CQL2-JSON.
    # Keeps SAR granules from consuming the max_items budget when optical_only=True.
    # Confirmed supported queryable on the ItsLive STAC server.
    if optical_only:
        search_kwargs["filter"] = {
            "op": "in",
            "args": [{"property": "platform"}, sorted(_OPTICAL)],
        }
        search_kwargs["filter_lang"] = "cql2-json"

    search = client.search(**search_kwargs)

    yielded = 0
    for item in search.items():
        props = item.properties

        # --- optical filter (not a queryable — must be done client-side) ---
        platform = props.get("platform", "")
        if optical_only and platform in _SAR:
            continue

        # --- date-window check ---
        # The STAC API returns pairs that straddle the window boundary.
        # Require at least one scene's acquisition date to fall inside [start, end].
        scene_1 = props.get("scene_1_id") or props.get("scene_1_frame")
        scene_2 = props.get("scene_2_id") or props.get("scene_2_frame")
        if not (
            _scene_in_window(scene_1, platform, start_dt, end_dt)
            or _scene_in_window(scene_2, platform, start_dt, end_dt)
        ):
            logger.debug("Skipping straddling pair: %s / %s", scene_1, scene_2)
            continue

        # --- client-side fallback for quality filters ---
        # These are redundant when the server honoured the CQL2 / max_interval
        # parameters, but protect against servers that silently ignore them.
        dt_days = props.get("date_dt")
        if max_dt_days is not None and dt_days is not None and dt_days > max_dt_days:
            continue

        valid_pct = props.get("percent_valid_pixels")
        if min_valid_pixels is not None and valid_pct is not None and valid_pct < min_valid_pixels:
            continue

        yield item.to_dict()

        yielded += 1
        if max_items is not None and yielded >= max_items:
            break

    logger.debug("search_granules: yielded %d items", yielded)
