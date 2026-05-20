"""ItsLive match-coverage scoring over a bounding box.

Workflow
--------
1. Search ItsLive STAC for optical pairs overlapping the bbox.
2. For each pair, download its NetCDF and read ``chip_size_height`` over the
   bbox window.  ``chip_size_height > 0`` marks cells where feature-tracking
   produced a real (non-interpolated) match.
3. For every scene that appears in a pair, accumulate the per-cell **maximum**
   coverage across all pairs it participates in.
4. Group scenes by ``(acquisition_date, wrs_path)``.  These are the scenes
   that will be mosaicked together, so combine their coverage grids via a
   union (logical OR).
5. Return groups ranked by the fraction of bbox cells that are covered.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as Date, datetime, timezone
from typing import Optional

import re

import fsspec
import h5py
import numpy as np

from .crs_utils import CRSInput, transform_bbox, bbox_to_wgs84
from .results import SceneResult
from .scene_parser import is_optical, parse_scene_id

logger = logging.getLogger(__name__)

# Variable used to detect real (non-interpolated) matches.
_MATCH_VAR = "chip_size_height"

# ItsLive Landsat/S2 granules use 120 m pixels; SAR uses 240 m.
# We snap the bbox to this grid so slices from different granules
# covering the same area have the same shape.
_GRID_SPACING = 120.0


# ------------------------------------------------------------------ #
# Data classes                                                         #
# ------------------------------------------------------------------ #

@dataclass
class DatePathGroup:
    """Same-date, same-orbit scenes evaluated jointly for bbox coverage.

    For Landsat scenes ``wrs_path`` is set and ``tile`` is ``None``.
    For Sentinel-2 scenes ``tile`` is set and ``wrs_path`` is ``None``.
    """

    acquisition_date: Date
    scenes: list[SceneResult]
    coverage_fraction: float
    """Fraction of bbox grid cells with at least one valid ItsLive match."""
    coverage_grid: np.ndarray
    """2D bool array (rows × cols) on the snapped bbox grid."""
    wrs_path: Optional[str] = None
    """WRS path number (Landsat only)."""
    tile: Optional[str] = None
    """MGRS tile ID (Sentinel-2 only)."""


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _snap_bbox(bbox: tuple[float, float, float, float], spacing: float = _GRID_SPACING):
    """Expand bbox outward to align with the ItsLive grid."""
    import math
    x0, y0, x1, y1 = bbox
    x0 = math.floor(x0 / spacing) * spacing + spacing / 2
    y0 = math.floor(y0 / spacing) * spacing + spacing / 2
    x1 = math.ceil(x1 / spacing) * spacing - spacing / 2
    y1 = math.ceil(y1 / spacing) * spacing - spacing / 2
    return x0, y0, x1, y1


def _get_data_url(item: dict) -> Optional[str]:
    assets = item.get("assets") or {}
    data = assets.get("data") or {}
    return data.get("href")


def _fetch_pair_coverage(
    url: str,
    bbox_native: tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """Read the bbox window of ``chip_size_height`` from an ItsLive NetCDF.

    Uses fsspec HTTP range requests so only the HDF5 chunks that cover the
    requested slice are transferred — not the full file.

    Parameters
    ----------
    url:
        HTTPS URL of the ItsLive granule NetCDF file.
    bbox_native:
        ``(x0, y0, x1, y1)`` in the granule's native CRS (EPSG:3031 for
        Antarctic data), already snapped to the ItsLive grid.

    Returns
    -------
    2D ``bool`` array where ``True`` = valid match, or ``None`` if the
    granule does not overlap the bbox or the read fails.
    """
    try:
        of = fsspec.open(url, "rb")
        with of as f, h5py.File(f, "r") as ds:
            x = ds["x"][:]
            y = ds["y"][:]
            x0, y0, x1, y1 = bbox_native

            xi0 = int(np.searchsorted(x, x0, side="left"))
            xi1 = int(np.searchsorted(x, x1, side="right"))

            # y may be stored descending (top → bottom).
            if len(y) > 1 and float(y[0]) > float(y[-1]):
                yi0 = int(np.searchsorted(-y, -y1, side="left"))
                yi1 = int(np.searchsorted(-y, -y0, side="right"))
            else:
                yi0 = int(np.searchsorted(y, y0, side="left"))
                yi1 = int(np.searchsorted(y, y1, side="right"))

            if xi0 >= xi1 or yi0 >= yi1:
                return None  # no overlap

            var = ds[_MATCH_VAR]
            if var.ndim == 3:
                chip = var[0, yi0:yi1, xi0:xi1]
            else:
                chip = var[yi0:yi1, xi0:xi1]
            chip = np.asarray(chip)

            # A cell is a real (non-interpolated) match when chip_size_height
            # is a small positive value (the height of the search chip in
            # metres, typically 120–3840 m for Landsat OLI).
            #
            # ItsLive v02 encodes "outside valid scene area" with the MSB set
            # (e.g. 32769 = 0x8001) so that the value is positive but NOT a
            # real chip size.  We exclude any value >= 2^15 (bit 15 set) in
            # addition to 0 (the NetCDF fill value).
            valid = (chip > 0) & (chip < np.uint16(0x8000))
            return valid

    except Exception as exc:
        logger.warning("Error reading coverage from %s: %s", url, exc)
        return None


def _tile_zone_overlaps_bbox(tile: str, bbox_wgs84: tuple[float, float, float, float]) -> bool:
    """Return True if the MGRS tile's UTM zone longitude range overlaps *bbox_wgs84*.

    Sentinel-2 MGRS tiles are prefixed with a 1-or-2-digit UTM zone number
    (e.g. "13CES" → zone 13, covering -108° to -102°).  ItsLive sometimes
    pairs scenes from adjacent UTM zones whose imagery does not overlap the
    query bbox; this check discards such cross-zone scenes before they
    accumulate spurious coverage credit.

    Landsat cross-path pairs are handled separately by ``_wrs_path_number``
    and the cross-path filter in the main loop.

    Returns True when the tile ID is unparseable so we never silently discard
    a scene we cannot validate.
    """
    m = re.match(r"^(\d{1,2})", tile)
    if not m:
        return True
    zone = int(m.group(1))
    zone_lon_min = (zone - 1) * 6.0 - 180.0
    zone_lon_max = zone * 6.0 - 180.0
    lon_min, _, lon_max, _ = bbox_wgs84
    # Interior overlap: the ranges must share more than a single boundary point.
    return zone_lon_max > lon_min and zone_lon_min < lon_max


def _wrs_path_number(scene_id: str) -> Optional[str]:
    """Extract the 3-digit WRS path string from a Landsat scene ID.

    For example, ``'LC08_L1GT_231114_...'`` → ``'231'``.
    Returns ``None`` when the scene ID does not follow the expected format.
    """
    parts = scene_id.split("_")
    if len(parts) >= 3:
        ppr = parts[2]          # e.g. "231114" (path 231, row 114)
        if len(ppr) >= 3 and ppr[:3].isdigit():
            return ppr[:3]
    return None


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def score_date_path_groups(
    granules: list[dict],
    bbox: tuple[float, float, float, float],
    bbox_crs: CRSInput = 3031,
    granule_crs: CRSInput = 3031,
    show_progress: bool = False,
) -> list[DatePathGroup]:
    """Score (date, WRS-path) scene groups by ItsLive match coverage in bbox.

    Parameters
    ----------
    granules:
        Raw STAC item dicts from :func:`~itslive_cloudfree.stac_client.search_granules`.
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *bbox_crs* units.
    bbox_crs:
        CRS of *bbox*.  Defaults to ``3031`` (EPSG:3031, metres).
    granule_crs:
        Native CRS of the ItsLive granules.  Defaults to ``3031``
        (Antarctic data).  Change to the appropriate UTM zone for other
        regions.
    show_progress:
        When ``True``, display a ``tqdm`` progress bar instead of per-granule
        log messages.

    Returns
    -------
    List of :class:`DatePathGroup` objects sorted by ``coverage_fraction``
    descending.
    """
    bbox_native = transform_bbox(bbox, bbox_crs, granule_crs)
    bbox_snapped = _snap_bbox(bbox_native)
    bbox_wgs84 = bbox_to_wgs84(bbox, bbox_crs)
    x0, y0, x1, y1 = bbox_snapped
    n_cols = round((x1 - x0) / _GRID_SPACING) + 1
    n_rows = round((y1 - y0) / _GRID_SPACING) + 1
    logger.debug("bbox grid: %d rows × %d cols at %.0fm", n_rows, n_cols, _GRID_SPACING)

    # scene_id → accumulated max coverage grid (logical OR across pairs)
    scene_coverage: dict[str, np.ndarray] = {}
    # scene_id → ParsedScene (for date/path metadata)
    scene_meta: dict[str, object] = {}

    if show_progress:
        try:
            from tqdm.auto import tqdm
            granule_iter = tqdm(granules, desc="Fetching coverage", unit="pair")
        except ImportError:
            logger.warning("tqdm not installed; falling back to log output")
            granule_iter = granules
    else:
        granule_iter = granules

    for item in granule_iter:
        props = item.get("properties", {})
        platform = props.get("platform", "")
        if not is_optical(platform):
            continue

        url = _get_data_url(item)
        if not url:
            logger.debug("No data URL in item %s", item.get("id"))
            continue

        logger.debug("Fetching coverage from %s", url.split("/")[-1])
        pair_cov = _fetch_pair_coverage(url, bbox_snapped)
        if pair_cov is None:
            continue

        # Resize to bbox grid if needed (handles minor edge differences).
        if pair_cov.shape != (n_rows, n_cols):
            pr, pc = pair_cov.shape
            out = np.zeros((n_rows, n_cols), dtype=bool)
            out[: min(pr, n_rows), : min(pc, n_cols)] = pair_cov[
                : min(pr, n_rows), : min(pc, n_cols)
            ]
            pair_cov = out

        # For Landsat: detect cross-path pairs.  Features tracked between
        # different WRS paths at high latitudes don't reliably indicate that
        # either scene fully images the query bbox.  Only within-path pairs
        # are credited so that a path (e.g. 002) cannot accumulate high
        # coverage by appearing in many pairs with distant paths (e.g. 231)
        # that do cover the bbox.
        scene_1_raw = props.get("scene_1_id") or props.get("scene_1_frame")
        scene_2_raw = props.get("scene_2_id") or props.get("scene_2_frame")
        wrs_path_1 = _wrs_path_number(scene_1_raw) if scene_1_raw else None
        wrs_path_2 = _wrs_path_number(scene_2_raw) if scene_2_raw else None
        is_cross_path = bool(wrs_path_1 and wrs_path_2 and wrs_path_1 != wrs_path_2)
        if is_cross_path:
            logger.debug(
                "Skipping cross-path Landsat pair: path %s ↔ path %s",
                wrs_path_1, wrs_path_2,
            )

        for key in ("scene_1_id", "scene_1_frame", "scene_2_id", "scene_2_frame"):
            sid = props.get(key)
            if not sid:
                continue
            if "frame" in key and props.get(key.replace("frame", "id")):
                continue  # prefer _id over _frame

            # Parse metadata first so we can check geographic overlap.
            if sid not in scene_meta:
                parsed = parse_scene_id(sid, platform)
                if parsed is not None:
                    scene_meta[sid] = parsed

            # Skip Landsat cross-path pairs (see comment above).
            if is_cross_path:
                continue

            # Skip scenes whose tile does not geographically cover the query bbox.
            # This prevents cross-tile ItsLive pairs from crediting a source scene
            # (e.g. tile 13CES, ~105°W–102°W) with coverage at a destination bbox
            # that the scene never actually imaged (e.g. Pine Island, ~100°W).
            if sid in scene_meta and scene_meta[sid].tile:
                if not _tile_zone_overlaps_bbox(scene_meta[sid].tile, bbox_wgs84):
                    logger.debug(
                        "Skipping scene %s: tile %s does not overlap bbox",
                        sid, scene_meta[sid].tile,
                    )
                    continue

            if sid not in scene_coverage:
                scene_coverage[sid] = pair_cov.copy()
            else:
                np.logical_or(scene_coverage[sid], pair_cov, out=scene_coverage[sid])

    if not scene_coverage:
        return []

    # Group scenes by (acquisition_date, orbit_key) where orbit_key is the
    # WRS path (Landsat) or MGRS tile (Sentinel-2).
    groups: dict[tuple, list[str]] = defaultdict(list)
    for sid, parsed in scene_meta.items():
        if parsed.path_row:
            orbit_key = parsed.path_row.split("/")[0]
        elif parsed.tile:
            orbit_key = parsed.tile
        else:
            continue
        key = (parsed.acquisition_date.date(), orbit_key)
        groups[key].append(sid)

    results: list[DatePathGroup] = []
    for (acq_date, orbit_key), scene_ids in groups.items():
        grids = [scene_coverage[s] for s in scene_ids if s in scene_coverage]
        if not grids:
            continue

        union = grids[0].copy()
        for g in grids[1:]:
            np.logical_or(union, g, out=union)

        coverage_frac = float(union.mean())

        # Determine whether this group is Landsat (path_row) or S2 (tile).
        sample = scene_meta[scene_ids[0]]
        is_landsat = bool(sample.path_row)

        scene_results = [
            SceneResult(
                scene_id=sid,
                platform=scene_meta[sid].platform,
                acquisition_date=scene_meta[sid].acquisition_date,
                path_row=scene_meta[sid].path_row,
                tile=scene_meta[sid].tile,
                pair_count=sum(
                    1 for item in granules
                    if sid in (
                        item.get("properties", {}).get("scene_1_id", ""),
                        item.get("properties", {}).get("scene_2_id", ""),
                    )
                ),
                mean_valid_pixels=0.0,
                cloud_free_score=round(coverage_frac * 100, 2),
            )
            for sid in scene_ids
            if sid in scene_meta
        ]

        results.append(DatePathGroup(
            acquisition_date=acq_date,
            scenes=scene_results,
            coverage_fraction=coverage_frac,
            coverage_grid=union,
            wrs_path=orbit_key if is_landsat else None,
            tile=orbit_key if not is_landsat else None,
        ))

    results.sort(key=lambda g: -g.coverage_fraction)
    return results
