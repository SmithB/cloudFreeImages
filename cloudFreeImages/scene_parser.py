"""Parse scene IDs from ItsLive granule metadata.

Supported formats
-----------------
Landsat Collection 2
    ``LC08_L1GT_224115_20201228_20210310_02_T2``
    Fields: sensor _ level _ PPPRRR _ acqdate _ procdate _ collection _ tier

Sentinel-2 (if/when present in ItsLive)
    ``S2A_MSIL1C_20200103T112459_N0208_R037_T29UMB_20200103T120053``
    Fields: platform _ level _ start_time _ baseline _ orbit _ tile _ proc_time

Sentinel-1 (SAR — excluded from cloud-free analysis, but parsed for completeness)
    ``S1B_IW_SLC__1SSH_20201231T092617_20201231T092645_024944_02F7F1_30C2``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

# ------------------------------------------------------------------ #
# Platform classification                                              #
# ------------------------------------------------------------------ #

#: Platforms whose images can be obscured by cloud (optical sensors).
OPTICAL_PLATFORMS: frozenset[str] = frozenset(
    ["LC04", "LT04", "LC05", "LT05", "LE07", "LC08", "LC09", "S2A", "S2B"]
)

#: SAR platforms — unaffected by cloud, excluded from cloud-free analysis.
SAR_PLATFORMS: frozenset[str] = frozenset(["S1A", "S1B", "S1C"])


def is_optical(platform: str) -> bool:
    return platform in OPTICAL_PLATFORMS


# ------------------------------------------------------------------ #
# Parsed scene dataclass                                               #
# ------------------------------------------------------------------ #

@dataclass
class ParsedScene:
    scene_id: str
    platform: str
    acquisition_date: datetime
    path_row: Optional[str]  # 'PPP/RRR' for Landsat
    tile: Optional[str]      # MGRS tile for Sentinel-2


# ------------------------------------------------------------------ #
# Landsat Collection 2                                                 #
# ------------------------------------------------------------------ #

# LC08_L1GT_224115_20201228_20210310_02_T2
_LANDSAT_RE = re.compile(
    r"^(?P<sensor>L[CETO]\d{2})_"   # LC08, LE07, LT05, …
    r"(?P<level>[A-Z0-9]+)_"        # L1GT, L1TP, L2SP, …
    r"(?P<path>\d{3})(?P<row>\d{3})_"
    r"(?P<acq>\d{8})_"              # YYYYMMDD
    r"\d{8}_"                       # processing date (ignored)
    r"\d{2}_"                       # collection
    r"[A-Z0-9]+"                    # tier
    r"$"
)


def _parse_landsat(scene_id: str) -> Optional[ParsedScene]:
    m = _LANDSAT_RE.match(scene_id)
    if not m:
        return None
    acq = datetime.strptime(m.group("acq"), "%Y%m%d").replace(tzinfo=timezone.utc)
    path_row = f"{int(m.group('path'))}/{int(m.group('row'))}"
    return ParsedScene(
        scene_id=scene_id,
        platform=m.group("sensor"),
        acquisition_date=acq,
        path_row=path_row,
        tile=None,
    )


# ------------------------------------------------------------------ #
# Sentinel-2                                                           #
# ------------------------------------------------------------------ #

# S2A_MSIL1C_20200103T112459_N0208_R037_T29UMB_20200103T120053
_S2_RE = re.compile(
    r"^(?P<platform>S2[AB])_"
    r"[A-Z0-9]+_"                         # level (MSIL1C, MSIL2A)
    r"(?P<acq>\d{8})T\d{6}_"             # YYYYMMDD
    r"N\d{4}_"                            # baseline
    r"R\d{3}_"                            # relative orbit
    r"T(?P<tile>[A-Z0-9]{5})_"           # MGRS tile
    r"\d{8}T\d{6}"                        # processing time
    r"$"
)


def _parse_sentinel2(scene_id: str) -> Optional[ParsedScene]:
    m = _S2_RE.match(scene_id)
    if not m:
        return None
    acq = datetime.strptime(m.group("acq"), "%Y%m%d").replace(tzinfo=timezone.utc)
    return ParsedScene(
        scene_id=scene_id,
        platform=m.group("platform"),
        acquisition_date=acq,
        path_row=None,
        tile=m.group("tile"),
    )


# ------------------------------------------------------------------ #
# Sentinel-1 (SAR)                                                    #
# ------------------------------------------------------------------ #

# S1B_IW_SLC__1SSH_20201231T092617_20201231T092645_024944_02F7F1_30C2
_S1_RE = re.compile(
    r"^(?P<platform>S1[ABC])_"
    r"[A-Z_]+_"                          # mode (IW_SLC__, EW_SLC__, …)
    r"\w+_"                              # polarisation
    r"(?P<acq>\d{8})T\d{6}_"           # YYYYMMDD
    r"\d{8}T\d{6}_"                     # stop time
    r"\d+_\w+_\w+"                      # orbit / datatake / checksum
    r"$"
)


def _parse_sentinel1(scene_id: str) -> Optional[ParsedScene]:
    m = _S1_RE.match(scene_id)
    if not m:
        return None
    acq = datetime.strptime(m.group("acq"), "%Y%m%d").replace(tzinfo=timezone.utc)
    return ParsedScene(
        scene_id=scene_id,
        platform=m.group("platform"),
        acquisition_date=acq,
        path_row=None,
        tile=None,
    )


# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #

def parse_scene_id(scene_id: str, platform_hint: str = "") -> Optional[ParsedScene]:
    """Parse a scene ID string into a :class:`ParsedScene`.

    *platform_hint* is the ``platform`` field from the STAC item properties
    and is used to select the right parser when the ID format alone is
    ambiguous.  Pass an empty string if not available.

    Returns ``None`` if the ID cannot be parsed.
    """
    # Dispatch by platform hint first (fast path), then by ID prefix.
    ph = platform_hint.upper()

    if ph.startswith("LC") or ph.startswith("LE") or ph.startswith("LT"):
        return _parse_landsat(scene_id)

    if ph.startswith("S2"):
        return _parse_sentinel2(scene_id)

    if ph.startswith("S1"):
        return _parse_sentinel1(scene_id)

    # No hint — try each parser in order.
    for parser in (_parse_landsat, _parse_sentinel2, _parse_sentinel1):
        result = parser(scene_id)
        if result is not None:
            return result

    return None
