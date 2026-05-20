"""Unit tests for itslive_cloudfree.scene_parser."""

import pytest
from datetime import datetime, timezone

from itslive_cloudfree.scene_parser import (
    parse_scene_id,
    is_optical,
    OPTICAL_PLATFORMS,
    SAR_PLATFORMS,
)


class TestLandsatParser:
    def test_lc08_basic(self):
        sid = "LC08_L1GT_224115_20201228_20210310_02_T2"
        result = parse_scene_id(sid, "LC08")
        assert result is not None
        assert result.platform == "LC08"
        assert result.path_row == "224/115"
        assert result.acquisition_date == datetime(2020, 12, 28, tzinfo=timezone.utc)
        assert result.tile is None

    def test_lc09(self):
        sid = "LC09_L1GT_160131_20220108_20220123_02_T2"
        result = parse_scene_id(sid, "LC09")
        assert result is not None
        assert result.platform == "LC09"
        assert result.path_row == "160/131"
        assert result.acquisition_date.year == 2022

    def test_le07(self):
        sid = "LE07_L1GT_224115_20010515_20170208_01_T2"
        result = parse_scene_id(sid, "LE07")
        assert result is not None
        assert result.platform == "LE07"
        assert result.acquisition_date == datetime(2001, 5, 15, tzinfo=timezone.utc)

    def test_no_hint_falls_back_to_autodetect(self):
        sid = "LC08_L1GT_224115_20201228_20210310_02_T2"
        result = parse_scene_id(sid)
        assert result is not None
        assert result.platform == "LC08"

    def test_invalid_returns_none(self):
        assert parse_scene_id("not_a_valid_scene_id") is None


class TestSentinel2Parser:
    def test_s2a(self):
        sid = "S2A_MSIL1C_20200103T112459_N0208_R037_T29UMB_20200103T120053"
        result = parse_scene_id(sid, "S2A")
        assert result is not None
        assert result.platform == "S2A"
        assert result.tile == "29UMB"
        assert result.acquisition_date == datetime(2020, 1, 3, tzinfo=timezone.utc)
        assert result.path_row is None

    def test_s2b(self):
        sid = "S2B_MSIL2A_20210615T103629_N0300_R008_T32UMF_20210615T130137"
        result = parse_scene_id(sid, "S2B")
        assert result is not None
        assert result.platform == "S2B"
        assert result.tile == "32UMF"


class TestSentinel1Parser:
    def test_s1b(self):
        sid = "S1B_IW_SLC__1SSH_20201231T092617_20201231T092645_024944_02F7F1_30C2"
        result = parse_scene_id(sid, "S1B")
        assert result is not None
        assert result.platform == "S1B"
        assert result.acquisition_date == datetime(2020, 12, 31, tzinfo=timezone.utc)
        assert result.path_row is None
        assert result.tile is None


class TestOpticalClassification:
    @pytest.mark.parametrize("platform", ["LC08", "LC09", "LE07", "S2A", "S2B"])
    def test_optical_platforms(self, platform):
        assert is_optical(platform)

    @pytest.mark.parametrize("platform", ["S1A", "S1B", "S1C"])
    def test_sar_platforms_not_optical(self, platform):
        assert not is_optical(platform)

    def test_unknown_platform_not_optical(self):
        assert not is_optical("UNKNOWN")
