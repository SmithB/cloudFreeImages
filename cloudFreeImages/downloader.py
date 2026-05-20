"""Download and crop optical scenes to a bounding box.

Each cloud-free :class:`~itslive_cloudfree.results.SceneResult` carries a
scene ID that can be looked up in the Element84 Earth Search STAC catalog.
Once the COG asset URLs are resolved, rasterio is used to do a windowed
read so only the pixels that fall within the requested bbox are downloaded.

Supported sensors
-----------------
* Landsat 8 / 9 (LC08, LC09) — collection ``landsat-c2-l1``
* Landsat 7 (LE07) — collection ``landsat-c2-l1``
* Sentinel-2 A/B (S2A, S2B) — collection ``sentinel-2-l2a``

Default bands downloaded
------------------------
* Landsat 8/9 : red (B4), green (B3), blue (B2), nir08 (B5)
* Landsat 7   : red (B3), green (B2), blue (B1), nir08 (B4)
* Sentinel-2  : red (B04), green (B03), blue (B02), nir (B08)

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import rasterio
from osgeo import gdal

gdal.UseExceptions()
from pyproj import CRS
from rasterio.windows import from_bounds

from .crs_utils import CRSInput, transform_bbox
from .results import SceneResult

logger = logging.getLogger(__name__)


_EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
_USGS_STAC_URL = "https://landsatlook.usgs.gov/stac-server"

# Maps ItsLive platform → Earth Search collection names to try (in order).
# Note: Earth Search only carries landsat-c2-l2 (L1 was dropped); L1/L1GT scenes
# must be retrieved from the USGS LandsatLook STAC (see _USGS_COLLECTIONS).
# Sentinel-2 is handled separately in _find_stac_item via _s2_earth_search_candidates.
_PLATFORM_COLLECTIONS: dict[str, list[str]] = {
    "LC08": ["landsat-c2-l2"],
    "LC09": ["landsat-c2-l2"],
    "LE07": ["landsat-c2-l2"],
    "LC05": [],
    "LT05": [],
}

# USGS LandsatLook STAC collection names (different naming convention from Earth Search).
_USGS_COLLECTIONS: dict[str, list[str]] = {
    "LC08": ["landsat-c2l1", "landsat-c2l2-sr"],
    "LC09": ["landsat-c2l1", "landsat-c2l2-sr"],
    "LE07": ["landsat-c2l1", "landsat-c2l2-sr"],
    "LC05": ["landsat-c2l1"],
    "LT05": ["landsat-c2l1"],
    "LC04": ["landsat-c2l1"],
    "LT04": ["landsat-c2l1"],
}

# Default band asset names per platform (Earth Search naming).
_DEFAULT_BANDS: dict[str, list[str]] = {
    "LC08": ["red", "green", "blue", "nir08"],
    "LC09": ["red", "green", "blue", "nir08"],
    "LE07": ["red", "green", "blue", "nir08"],
    "LC05": ["red", "green", "blue", "nir08"],
    "LT05": ["red", "green", "blue", "nir08"],
    "S2A":  ["red", "green", "blue", "nir"],
    "S2B":  ["red", "green", "blue", "nir"],
}

# Friendly labels shown in output filenames.
_BAND_LABELS: dict[str, list[str]] = {
    "LC08": ["B4", "B3", "B2", "B5"],
    "LC09": ["B4", "B3", "B2", "B5"],
    "LE07": ["B3", "B2", "B1", "B4"],
    "LC05": ["B3", "B2", "B1", "B4"],
    "LT05": ["B3", "B2", "B1", "B4"],
    "S2A":  ["B04", "B03", "B02", "B08"],
    "S2B":  ["B04", "B03", "B02", "B08"],
}


# ------------------------------------------------------------------ #
# Scene ID normalisation                                               #
# ------------------------------------------------------------------ #

def _s2_earth_search_candidates(scene_id: str) -> list[tuple[str, str]]:
    """Return (collection, earth_search_id) pairs to try for a Sentinel-2 scene.

    ItsLive stores the full ESA product name, e.g.:
    ``S2B_MSIL1C_20191223T151259_N0208_R139_T14CMB_20191223T180217``

    Earth Search v1 has two S2 collections:
    * ``sentinel-2-l2a`` — HTTPS COG GeoTIFFs, publicly accessible (preferred)
    * ``sentinel-2-l1c`` — old /vsis3/ JPEG2000 paths, requires credentials

    We therefore try the L2A collection first (substituting "L2A" in the ID),
    then fall back to the native L1C collection.
    """
    parts = scene_id.split("_")
    if len(parts) < 6:
        # Unexpected format; return as-is for both collections.
        return [("sentinel-2-l2a", scene_id), ("sentinel-2-l1c", scene_id)]

    satellite = parts[0]        # e.g. "S2B"
    sensing = parts[2][:8]      # e.g. "20191223"
    tile = parts[5].lstrip("T") # e.g. "T14CMB" → "14CMB"

    base = f"{satellite}_{tile}_{sensing}_0"
    return [
        ("sentinel-2-l2a", f"{base}_L2A"),
        ("sentinel-2-l1c", f"{base}_L1C"),
    ]


# ------------------------------------------------------------------ #
# Asset lookup                                                         #
# ------------------------------------------------------------------ #

def _landsat_path_row_date(scene_id: str) -> Optional[tuple[str, str, str]]:
    """Parse (path, row, 'YYYY-MM-DD') from a Landsat Collection 2 scene ID.

    Returns ``None`` if the ID does not match the expected format.
    """
    parts = scene_id.split("_")
    if len(parts) < 5:
        return None
    ppr = parts[2]          # e.g. "224115"
    raw_date = parts[3]     # e.g. "20191223"
    if len(ppr) != 6 or not ppr.isdigit():
        return None
    if len(raw_date) != 8 or not raw_date.isdigit():
        return None
    path = ppr[:3]          # "224"
    row = ppr[3:]           # "115"
    date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"  # "2019-12-23"
    return path, row, date_str


def _find_stac_item(scene_id: str, platform: str):
    """Return a pystac Item for *scene_id*, or None if not found."""
    import pystac_client

    client = pystac_client.Client.open(_EARTH_SEARCH_URL)

    if platform in ("S2A", "S2B"):
        # ItsLive uses full ESA product names; Earth Search uses shorter IDs.
        # Try L2A first (HTTPS COG assets, no credentials needed), then L1C.
        candidates = _s2_earth_search_candidates(scene_id)
        for collection, lookup_id in candidates:
            logger.debug("Trying %s in %s", lookup_id, collection)
            try:
                item = client.get_collection(collection).get_item(lookup_id)
                if item is not None:
                    logger.debug("Found %s in %s", lookup_id, collection)
                    return item
            except Exception:
                pass
            try:
                results = client.search(collections=[collection], ids=[lookup_id], max_items=1)
                for item in results.items():
                    logger.debug("Found %s in %s (via search)", lookup_id, collection)
                    return item
            except Exception as exc:
                logger.debug("Earth Search lookup failed for %s in %s: %s",
                             lookup_id, collection, exc)
        return None

    # ------------------------------------------------------------------ #
    # Landsat                                                              #
    # ------------------------------------------------------------------ #
    # ItsLive scene IDs are L1 IDs (e.g. LC08_L1GT_...).  Earth Search
    # carries Collection 2 L2 products whose asset hrefs are S3 URIs in the
    # requester-pays ``usgs-landsat`` bucket — not accessible from a laptop
    # without AWS credentials and payer configuration.  USGS LandsatLook
    # (landsatlook.usgs.gov) serves the same data via HTTPS with EarthData
    # cookie authentication and is therefore the only practical source here.
    # ------------------------------------------------------------------ #

    # USGS LandsatLook STAC (HTTPS assets; EarthData cookies for auth).
    usgs_collections = _USGS_COLLECTIONS.get(platform, [])
    if usgs_collections:
        try:
            usgs_client = pystac_client.Client.open(_USGS_STAC_URL)
            for collection in usgs_collections:
                logger.debug("Trying USGS LandsatLook: %s in %s", scene_id, collection)
                try:
                    item = usgs_client.get_collection(collection).get_item(scene_id)
                    if item is not None:
                        logger.debug("Found %s in USGS %s", scene_id, collection)
                        return item
                except Exception:
                    pass
                try:
                    results = usgs_client.search(
                        collections=[collection], ids=[scene_id], max_items=1
                    )
                    for item in results.items():
                        logger.debug("Found %s in USGS %s (via search)", scene_id, collection)
                        return item
                except Exception as exc:
                    logger.debug("USGS lookup failed for %s in %s: %s",
                                 scene_id, collection, exc)
        except Exception as exc:
            logger.debug("Could not connect to USGS LandsatLook STAC: %s", exc)

    logger.warning("Scene %s not found in Earth Search or USGS LandsatLook", scene_id)
    return None


# ------------------------------------------------------------------ #
# GDAL VFS path helpers                                                #
# ------------------------------------------------------------------ #

_DEFAULT_COOKIE_FILE = Path.home() / ".usgs_landsat_cookies.txt"


def _apply_cookie_config() -> None:
    """Set GDAL_HTTP_COOKIEFILE/COOKIEJAR if a cookie file is available.

    Checks (in order):
    1. The ``GDAL_HTTP_COOKIEFILE`` environment variable.
    2. The default cookie file written by ``itslive-usgs-login``.

    This is needed when calling ``gdal.Warp`` directly because GDAL config
    options set in the OS environment may not be inherited by a running
    Jupyter kernel.
    """
    import os
    cookie_file = os.environ.get("GDAL_HTTP_COOKIEFILE") or (
        str(_DEFAULT_COOKIE_FILE) if _DEFAULT_COOKIE_FILE.exists() else None
    )
    if cookie_file:
        gdal.SetConfigOption("GDAL_HTTP_COOKIEFILE", cookie_file)
        gdal.SetConfigOption("GDAL_HTTP_COOKIEJAR", cookie_file)
        logger.debug("Using USGS cookie file: %s", cookie_file)


def _to_gdal_path(href: str) -> str:
    """Convert a bare URL to a GDAL VFS path that gdal.Warp can open.

    rasterio.open() handles URL routing transparently, but gdal.Warp called
    directly with string arguments needs explicit VFS prefixes.
    """
    if href.startswith("https://") or href.startswith("http://"):
        return f"/vsicurl/{href}"
    if href.startswith("s3://"):
        return f"/vsis3/{href[5:]}"
    return href


# ------------------------------------------------------------------ #
# Single-band crop                                                     #
# ------------------------------------------------------------------ #

def _crop_band(
    href: str,
    bbox_native: tuple[float, float, float, float],
    dst_path: Path,
) -> None:
    """Window-read one COG asset and write a cropped GeoTIFF to *dst_path*.

    Raises
    ------
    ValueError
        If the requested bbox does not intersect the image extent.
    """
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(href) as src:
        window = from_bounds(*bbox_native, transform=src.transform)
        # Clip to the actual image extent so a bbox that only partially
        # overlaps the scene still produces a valid (smaller) output tile.
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"Requested bbox does not intersect image extent.\n"
                f"  Image bounds (native CRS): {src.bounds}\n"
                f"  Requested bbox (native CRS): {bbox_native}"
            )
        data = src.read(window=window)
        win_transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            width=data.shape[-1],
            height=data.shape[-2],
            transform=win_transform,
            driver="GTiff",
            compress="deflate",
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    logger.debug("Wrote %s", dst_path)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def download_scene(
    scene: SceneResult,
    bbox: tuple[float, float, float, float],
    bbox_crs: CRSInput = 4326,
    bands: Optional[list[str]] = None,
    out_dir: str | Path = ".",
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Download one scene cropped to *bbox* and return a dict of band→path.

    Parameters
    ----------
    scene:
        A :class:`~itslive_cloudfree.results.SceneResult` as returned by
        :func:`~itslive_cloudfree.search`.
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *bbox_crs* units.
    bbox_crs:
        CRS of *bbox*.  Accepts integer EPSG codes (e.g. ``3031``) or
        strings (e.g. ``"EPSG:3031"``).  Defaults to ``4326`` (WGS-84).
    bands:
        Earth Search asset names to download, e.g. ``["red", "nir08"]``.
        Defaults to the platform's standard visual + NIR set.
    out_dir:
        Directory where cropped GeoTIFFs are written.  Created if absent.
    skip_existing:
        When ``True`` (default), skip a band if its output file already
        exists.

    Returns
    -------
    dict mapping band name → ``Path`` of the written GeoTIFF.
    Bands that could not be found in the catalog are omitted.
    """
    out_dir = Path(out_dir)
    bands = bands or _DEFAULT_BANDS.get(scene.platform, ["red", "green", "blue"])
    labels = _BAND_LABELS.get(scene.platform, bands)
    band_label = dict(zip(bands, labels))

    # 1. Look up the scene in Earth Search.
    item = _find_stac_item(scene.scene_id, scene.platform)
    if item is None:
        raise LookupError(
            f"Scene {scene.scene_id!r} not found in Earth Search or USGS LandsatLook."
        )

    # 2. Determine the native CRS of the image (from the STAC item).
    native_crs_str = (
        item.properties.get("proj:epsg")
        or item.properties.get("proj:code")
    )
    if native_crs_str is None:
        # Fall back to parsing from the item's projection extension.
        ext = item.ext.get("proj") if hasattr(item, "ext") else None
        native_crs_str = getattr(ext, "epsg", None) or "EPSG:4326"

    native_crs = CRS.from_user_input(
        f"EPSG:{native_crs_str}" if str(native_crs_str).isdigit() else native_crs_str
    )

    # 3. Transform the user bbox to the image native CRS.
    bbox_native = transform_bbox(bbox, bbox_crs, native_crs)

    # 4. Download each requested band.
    written: dict[str, Path] = {}
    for band in bands:
        asset = item.assets.get(band)
        if asset is None:
            logger.warning("Band %r not found in assets for %s; skipping",
                           band, scene.scene_id)
            continue

        label = band_label.get(band, band)
        dst_path = out_dir / f"{scene.scene_id}_{label}.tif"

        if skip_existing and dst_path.exists():
            logger.info("Skipping existing file %s", dst_path)
            written[band] = dst_path
            continue

        logger.info("Downloading %s band %s → %s", scene.scene_id, band, dst_path)
        _crop_band(asset.href, bbox_native, dst_path)
        written[band] = dst_path

    return written


def download_scenes(
    scenes: list[SceneResult],
    bbox: tuple[float, float, float, float],
    bbox_crs: CRSInput = 4326,
    bands: Optional[list[str]] = None,
    out_dir: str | Path = ".",
    skip_existing: bool = True,
    max_scenes: Optional[int] = None,
) -> dict[str, dict[str, Path]]:
    """Download multiple scenes, returning a nested dict of scene_id→band→path.

    Scenes that cannot be found in Earth Search are skipped with a warning
    rather than raising an exception.

    Parameters
    ----------
    scenes:
        Ordered list of scenes (e.g. the top-N from :func:`~itslive_cloudfree.search`).
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *bbox_crs* units.
    bbox_crs:
        CRS of *bbox*.  Defaults to ``4326`` (WGS-84).
    max_scenes:
        Process at most this many scenes (useful when *scenes* is long).

    Returns
    -------
    ``{scene_id: {band: Path}}`` for every scene successfully downloaded.
    """
    results: dict[str, dict[str, Path]] = {}
    subset = scenes[:max_scenes] if max_scenes else scenes

    for scene in subset:
        try:
            band_paths = download_scene(
                scene,
                bbox=bbox,
                bbox_crs=bbox_crs,
                bands=bands,
                out_dir=out_dir,
                skip_existing=skip_existing,
            )
            results[scene.scene_id] = band_paths
        except LookupError as exc:
            logger.warning("%s", exc)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", scene.scene_id, exc)

    return results


def download_scenes_mosaic(
    scenes: list[SceneResult],
    bbox: tuple[float, float, float, float],
    bbox_crs: CRSInput = 4326,
    dst_crs: CRSInput = 3031,
    bands: Optional[list[str]] = None,
    resolution: Optional[float] = None,
    out_dir: str | Path = ".",
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Download, mosaic, and reproject multiple scenes in one GDAL warp pass.

    Uses :func:`gdal.Warp` with a list of COG hrefs so the mosaic and
    reprojection happen in a single pipeline — no intermediate files are
    written.

    Parameters
    ----------
    scenes:
        Scenes to mosaic (typically same-path, same-date adjacent rows).
    bbox:
        ``(min_x, min_y, max_x, max_y)`` in *bbox_crs* units.
    bbox_crs:
        CRS of *bbox*.  Defaults to ``4326`` (WGS-84).
    dst_crs:
        Output CRS.  Accepts integer EPSG codes or strings.
        Defaults to ``3031`` (Antarctic Polar Stereographic).  Avoid
        geographic CRS (e.g. EPSG:4326) for polar data — polar scenes
        occupy a non-rectangular footprint in geographic coordinates and
        will leave large nodata areas in the output.
    bands:
        Asset names to download.  Defaults to the platform's standard set.
    resolution:
        Output pixel size in *dst_crs* units.  If ``None``, GDAL chooses
        based on the native resolution of the inputs.
    out_dir:
        Directory where output GeoTIFFs are written.
    skip_existing:
        Skip a band if its output file already exists.

    Returns
    -------
    ``{band_name: Path}`` for every band successfully written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_crs_obj = CRS.from_user_input(dst_crs)
    dst_crs_wkt = dst_crs_obj.to_wkt()
    bbox_dst = transform_bbox(bbox, bbox_crs, dst_crs_obj)

    # Resolve STAC items for all scenes.
    scene_items: list[tuple[SceneResult, object]] = []
    for scene in scenes:
        item = _find_stac_item(scene.scene_id, scene.platform)
        if item is None:
            logger.warning("Scene %s not found, skipping", scene.scene_id)
            continue
        scene_items.append((scene, item))

    if not scene_items:
        return {}

    first_scene = scene_items[0][0]
    band_list = bands or _DEFAULT_BANDS.get(first_scene.platform, ["red", "green", "blue"])
    labels = _BAND_LABELS.get(first_scene.platform, band_list)
    band_label = dict(zip(band_list, labels))

    # Build an output filename that reflects all contributing scenes.
    date_str = first_scene.acquisition_date.strftime("%Y%m%d")
    platform = first_scene.platform
    rows = "_".join(
        (s.path_row.replace("/", "-") if s.path_row else s.tile or s.scene_id[:15])
        for s, _ in scene_items
    )

    written: dict[str, Path] = {}

    for band in band_list:
        label = band_label.get(band, band)
        out_path = out_dir / f"{platform}_{rows}_{date_str}_{label}.tif"

        if skip_existing and out_path.exists():
            logger.info("Skipping existing %s", out_path)
            written[band] = out_path
            continue

        hrefs = []
        for scene, item in scene_items:
            asset = item.assets.get(band)
            if asset is None:
                logger.warning("Band %r not found in %s, skipping scene",
                               band, scene.scene_id)
                continue
            hrefs.append(_to_gdal_path(asset.href))

        if not hrefs:
            continue

        # Read the nodata value from the first source so GDAL can mask edge
        # pixels during warping and stamp the value into the output metadata.
        gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        _apply_cookie_config()
        _src_ds = gdal.Open(hrefs[0])
        src_nodata = (
            _src_ds.GetRasterBand(1).GetNoDataValue()
            if _src_ds is not None else None
        )
        _src_ds = None
        if src_nodata is None:
            src_nodata = 0  # conventional nodata for Landsat and Sentinel-2

        warp_opts = gdal.WarpOptions(
            format="GTiff",
            outputBounds=bbox_dst,          # (minX, minY, maxX, maxY) in dst_crs
            dstSRS=dst_crs_wkt,
            xRes=resolution,
            yRes=resolution,
            srcNodata=src_nodata,
            dstNodata=src_nodata,
            resampleAlg="bilinear",
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
            multithread=True,
        )

        logger.info("Warping %d source(s) → %s", len(hrefs), out_path)
        try:
            result = gdal.Warp(str(out_path), hrefs, options=warp_opts)
        except Exception as exc:
            logger.warning("gdal.Warp failed for band %s: %s", band, exc)
            continue
        if result is None:
            logger.warning("gdal.Warp returned None for band %s", band)
            continue
        result = None  # close dataset

        written[band] = out_path

    return written
