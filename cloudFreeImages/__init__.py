"""ItsLive Cloud-Free Scene Finder.

Use ItsLive feature-tracking success as a proxy for cloud-free conditions
in Landsat and Sentinel-2 optical imagery.

Quickstart
----------
    from itslive_cloudfree import search

    results = search(
        bbox=(-102.0, -76.0, -99.0, -74.0),   # Pine Island Glacier
        start_date="2019-10-01",
        end_date="2020-03-31",
    )

    for r in results[:10]:
        print(r.scene_id, r.cloud_free_score)

Optional enrichment with official eo:cloud_cover metadata::

    from itslive_cloudfree import search
    from itslive_cloudfree.enrich import enrich_cloud_cover

    results = enrich_cloud_cover(search(...))
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .crs_utils import CRSInput, transform_bbox, bbox_to_wgs84
from .results import SceneResult, to_dataframe, to_geojson, to_csv
from .stac_client import search_granules
from .scorer import score_scenes
from .coverage import score_date_path_groups, DatePathGroup

__all__ = [
    "search",
    "search_granules",
    "score_scenes",
    "score_date_path_groups",
    "DatePathGroup",
    "SceneResult",
    "to_dataframe",
    "to_geojson",
    "to_csv",
    "transform_bbox",
    "bbox_to_wgs84",
]


def search(
    bbox: tuple[float, float, float, float],
    start_date: datetime | str,
    end_date: datetime | str,
    *,
    bbox_crs: CRSInput = "EPSG:4326",
    optical_only: bool = True,
    max_dt_days: Optional[int] = None,
    min_valid_pixels: Optional[float] = 30.0,
    max_granules: Optional[int] = None,
) -> list[SceneResult]:
    """Find cloud-free optical scenes over a geographic area and time range.

    Searches the ItsLive STAC catalog for feature-tracking granule pairs,
    then ranks every unique scene by how often it appears in successful
    pairs and how high its valid-pixel fraction is.

    Parameters
    ----------
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *bbox_crs* units.
    start_date, end_date:
        Search window.  ``datetime`` objects or ISO-8601 strings
        (``'YYYY-MM-DD'`` is accepted).
    bbox_crs:
        CRS of *bbox*.  Accepts an integer EPSG code (e.g. ``3031``) or a
        string (e.g. ``"EPSG:3031"``).  Defaults to ``"EPSG:4326"``
        (WGS-84 lon/lat).  Requires ``pyproj`` when a non-WGS84 CRS is
        given: ``pip install 'itslive-cloudfree[download]'``.
    optical_only:
        Exclude Sentinel-1 SAR granules (default ``True``).  SAR is
        unaffected by cloud so its pair frequency tells us nothing about
        cloud cover.
    max_dt_days:
        Discard pairs separated by more than this many days.
    min_valid_pixels:
        Discard pairs with fewer valid pixels than this percentage (default 30).
    max_granules:
        Cap the number of granules fetched (useful for quick exploratory
        queries; ``None`` means fetch everything).

    Returns
    -------
    list[SceneResult]
        Scenes sorted by ``cloud_free_score`` descending (most cloud-free
        first).  Returns an empty list if no optical granules were found.
    """
    granules = list(
        search_granules(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            bbox_crs=bbox_crs,
            optical_only=optical_only,
            max_dt_days=max_dt_days,
            min_valid_pixels=min_valid_pixels,
            max_items=max_granules,
        )
    )
    return score_scenes(granules)
