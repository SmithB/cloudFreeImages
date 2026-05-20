"""Cloud-free scoring derived from ItsLive feature-tracking success.

Scoring rationale
-----------------
ItsLive only produces a granule when feature tracking *succeeded* between
two images.  For optical sensors (Landsat, Sentinel-2) that means both
images must have been cloud-free over the tracked surface.

Given a set of granules we can therefore infer:

* **pair_count** — how many successful pairs a scene participates in.
  A scene that appears in many pairs is reliably cloud-free across the
  whole search window.

* **mean_valid_pixels** — the average ``percent_valid_pixels`` of those
  pairs, which reflects how much of the *glacier surface* was visible
  (clouds, snow-on-sea-ice confusion, etc. reduce this number).

The composite ``cloud_free_score`` (0-100) is:

    score = mean_valid_pixels × log2(1 + pair_count)

Then linearly scaled so the highest-scoring scene in the result set
equals 100.  This rewards both high quality *and* repeatability.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timezone
from typing import Optional
import numpy as np

from .results import SceneResult
from .scene_parser import is_optical, parse_scene_id


def score_scenes(granules: list[dict]) -> list[SceneResult]:
    """Derive per-scene cloud-free scores from a list of ItsLive STAC items.

    Parameters
    ----------
    granules:
        Raw STAC item dicts as returned by :func:`~itslive_cloudfree.stac_client.search_granules`.

    Returns
    -------
    list[SceneResult]
        Scenes ranked by ``cloud_free_score`` descending, with optical-only
        scenes included (SAR scenes are silently dropped).
    """
    # scene_id -> list of percent_valid_pixels values (one per pair)
    valid_pixels_per_scene: dict[str, list[float]] = defaultdict(list)

    for item in granules:
        props = item.get("properties", {})
        platform = props.get("platform", "")
        valid_pct = props.get("percent_valid_pixels")

        # Skip SAR — clouds don't affect SAR so frequency means nothing.
        if not is_optical(platform):
            continue

        scene_1 = props.get("scene_1_id") or props.get("scene_1_frame")
        scene_2 = props.get("scene_2_id") or props.get("scene_2_frame")

        for sid in (scene_1, scene_2):
            if sid:
                pct = float(valid_pct) if valid_pct is not None else 0.0
                valid_pixels_per_scene[sid].append(pct)

    if not valid_pixels_per_scene:
        return []

    # Build SceneResult objects, resolving platform / date from the scene ID.
    # We need to re-scan granules to get the platform hint for each scene.
    platform_hint: dict[str, str] = {}
    for item in granules:
        props = item.get("properties", {})
        plat = props.get("platform", "")
        for key in ("scene_1_id", "scene_2_id"):
            sid = props.get(key)
            if sid and sid not in platform_hint:
                platform_hint[sid] = plat

    raw_scores: list[tuple[float, SceneResult]] = []

    for scene_id, pct_list in valid_pixels_per_scene.items():
        pair_count = len(pct_list)
        max_valid = np.max(pct_list)
        mean_valid = sum(pct_list) / pair_count
        #raw = mean_valid * math.log2(1 + pair_count)
        raw = max_valid
        parsed = parse_scene_id(scene_id, platform_hint.get(scene_id, ""))

        if parsed is not None:
            acq_date = parsed.acquisition_date
            platform = parsed.platform
            path_row = parsed.path_row
            tile = parsed.tile
        else:
            # Fallback: store what we know; date will be epoch.
            from datetime import datetime
            acq_date = datetime(1970, 1, 1, tzinfo=timezone.utc)
            platform = platform_hint.get(scene_id, "UNKNOWN")
            path_row = None
            tile = None

        raw_scores.append((
            raw,
            SceneResult(
                scene_id=scene_id,
                platform=platform,
                acquisition_date=acq_date,
                path_row=path_row,
                tile=tile,
                pair_count=pair_count,
                mean_valid_pixels=round(mean_valid, 1),
                cloud_free_score=0.0,  # filled in below
            ),
        ))

    if not raw_scores:
        return []

    # Normalise to 0-100
    max_raw = max(r for r, _ in raw_scores)
    results = []
    for raw, scene in raw_scores:
        scene.cloud_free_score = round((raw / max_raw) * 100.0, 2) if max_raw > 0 else 0.0
        results.append(scene)

    results.sort(key=lambda s: s.cloud_free_score, reverse=True)
    return results
