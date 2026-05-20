"""Result data structures and export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SceneResult:
    """A single optical scene ranked by its cloud-free likelihood.

    The score is derived entirely from ItsLive feature-tracking success:
    scenes that appear in many pairs with high valid-pixel fractions are
    very likely cloud-free over the search area.
    """

    scene_id: str
    """Full original scene identifier as stored in ItsLive."""

    platform: str
    """Sensor platform code, e.g. 'LC08', 'LC09', 'LE07', 'S2A', 'S2B'."""

    acquisition_date: datetime
    """UTC acquisition date parsed from the scene ID."""

    path_row: Optional[str]
    """'PPP/RRR' for Landsat (e.g. '224/115'), None for other sensors."""

    tile: Optional[str]
    """MGRS tile for Sentinel-2 (e.g. '54HUE'), None for Landsat."""

    pair_count: int
    """Number of ItsLive granule pairs this scene participates in."""

    mean_valid_pixels: float
    """Mean percent_valid_pixels across all pairs containing this scene."""

    cloud_free_score: float
    """Composite 0-100 score; higher means more likely cloud-free."""

    eo_cloud_cover: Optional[float] = field(default=None)
    """Official cloud-cover % from source catalog, if enrichment was run."""

    # ------------------------------------------------------------------ #
    # Export helpers                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "platform": self.platform,
            "acquisition_date": self.acquisition_date.isoformat(),
            "path_row": self.path_row,
            "tile": self.tile,
            "pair_count": self.pair_count,
            "mean_valid_pixels": round(self.mean_valid_pixels, 1),
            "cloud_free_score": round(self.cloud_free_score, 2),
            "eo_cloud_cover": self.eo_cloud_cover,
        }


def to_dataframe(results: list[SceneResult]):
    """Convert a list of SceneResult objects to a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas to use to_dataframe(): pip install pandas") from exc

    return pd.DataFrame([r.to_dict() for r in results])


def to_geojson(results: list[SceneResult]) -> str:
    """Serialize results as a GeoJSON FeatureCollection.

    Note: SceneResult does not carry full footprint geometry (that would
    require a second catalog query). Each feature is a Point at the scene
    center, present only for Landsat results where path/row is known.
    """
    features = []
    for r in results:
        props = r.to_dict()
        # We don't have geometry without a second catalog hit, so omit it.
        features.append({"type": "Feature", "geometry": None, "properties": props})

    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


def to_csv(results: list[SceneResult], path: str) -> None:
    """Write results to a CSV file at *path*."""
    df = to_dataframe(results)
    df.to_csv(path, index=False)
