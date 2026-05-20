"""Optional enrichment: attach official eo:cloud_cover from source catalogs.

This module queries public STAC catalogs for the source Landsat and
Sentinel-2 imagery to retrieve the official ``eo:cloud_cover`` metadata
and attach it to :class:`~itslive_cloudfree.results.SceneResult` objects.

Supported catalogs
------------------
* Landsat Collection 2 on AWS Element84 Earth Search
  ``https://earth-search.aws.element84.com/v1``
  Collections: ``landsat-c2-l1``, ``landsat-c2-l2``

* Sentinel-2 L2A on AWS Element84 Earth Search
  ``https://earth-search.aws.element84.com/v1``
  Collection: ``sentinel-2-l2a``

Usage
-----
    from itslive_cloudfree.enrich import enrich_cloud_cover
    results = enrich_cloud_cover(results)
"""

from __future__ import annotations

import logging
from typing import Optional

from .results import SceneResult

logger = logging.getLogger(__name__)

_EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

# Maps ItsLive platform codes to Earth Search collection names.
_PLATFORM_TO_COLLECTIONS: dict[str, list[str]] = {
    "LC08": ["landsat-c2-l1", "landsat-c2-l2"],
    "LC09": ["landsat-c2-l1", "landsat-c2-l2"],
    "LE07": ["landsat-c2-l1", "landsat-c2-l2"],
    "LC05": ["landsat-c2-l1"],
    "LT05": ["landsat-c2-l1"],
    "LC04": ["landsat-c2-l1"],
    "LT04": ["landsat-c2-l1"],
    "S2A": ["sentinel-2-l2a", "sentinel-2-l1c"],
    "S2B": ["sentinel-2-l2a", "sentinel-2-l1c"],
}


def _lookup_cloud_cover(scene_id: str, platform: str) -> Optional[float]:
    """Query Earth Search for the official cloud cover of a single scene.

    Returns the ``eo:cloud_cover`` value (0-100) or ``None`` if not found.
    """
    try:
        import pystac_client
    except ImportError as exc:
        raise ImportError(
            "pystac-client is required for enrichment: pip install pystac-client"
        ) from exc

    collections = _PLATFORM_TO_COLLECTIONS.get(platform, [])
    if not collections:
        return None

    client = pystac_client.Client.open(_EARTH_SEARCH_URL)

    for collection in collections:
        try:
            # Earth Search stores Landsat scenes with their full scene ID as
            # the item ID (case-insensitive).  Try a direct item fetch first.
            item = client.get_collection(collection).get_item(scene_id)
            if item is not None:
                cc = item.properties.get("eo:cloud_cover")
                if cc is not None:
                    return float(cc)
        except Exception:
            pass

        # Fallback: text search by scene ID in properties.
        try:
            results = client.search(
                collections=[collection],
                filter={"op": "=", "args": [{"property": "id"}, scene_id]},
                filter_lang="cql2-json",
                max_items=1,
            )
            for item in results.items():
                cc = item.properties.get("eo:cloud_cover")
                if cc is not None:
                    return float(cc)
        except Exception as exc:
            logger.debug("Earth Search lookup failed for %s: %s", scene_id, exc)

    return None


def enrich_cloud_cover(
    results: list[SceneResult],
    *,
    skip_missing: bool = True,
) -> list[SceneResult]:
    """Attach official ``eo_cloud_cover`` values to *results* in-place.

    Parameters
    ----------
    results:
        List returned by :func:`~itslive_cloudfree.search`.
    skip_missing:
        When ``True`` (default), scenes whose cloud cover cannot be found
        are left with ``eo_cloud_cover=None`` and no error is raised.

    Returns
    -------
    The same list with ``eo_cloud_cover`` populated where available.
    """
    for scene in results:
        try:
            cc = _lookup_cloud_cover(scene.scene_id, scene.platform)
            scene.eo_cloud_cover = cc
        except Exception as exc:
            if not skip_missing:
                raise
            logger.warning("Could not enrich %s: %s", scene.scene_id, exc)

    return results
