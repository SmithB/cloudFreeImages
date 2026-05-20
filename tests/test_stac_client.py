"""Unit tests for itslive_cloudfree.stac_client.

Network calls are mocked using pytest-mock / unittest.mock so the test
suite runs offline.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------ #
# Helpers to build fake pystac items                                  #
# ------------------------------------------------------------------ #

def _fake_item(scene_1_id, scene_2_id, platform, valid_pct, date_dt=16.0):
    """Return an object that mimics a pystac Item."""
    item = MagicMock()
    item.properties = {
        "platform": platform,
        "scene_1_id": scene_1_id,
        "scene_2_id": scene_2_id,
        "percent_valid_pixels": valid_pct,
        "date_dt": date_dt,
    }
    item.to_dict.return_value = {
        "type": "Feature",
        "properties": item.properties,
    }
    return item


# Scene IDs with various acquisition dates for date-window tests.
# Acquisition dates encoded in the scene ID (field 4, YYYYMMDD):
_LC08_A   = "LC08_L1GT_224115_20200101_20200113_02_T2"  # 2020-01-01  inside
_LC08_B   = "LC08_L1GT_224115_20200117_20200129_02_T2"  # 2020-01-17  inside
_LC08_PRE = "LC08_L1GT_224115_20191225_20200106_02_T2"  # 2019-12-25  before window
_LC08_OLD = "LC08_L1GT_224115_20190901_20190913_02_T2"  # 2019-09-01  before window
_S1B_1    = "S1B_IW_SLC__1SSH_20200101T120000_20200101T120028_024000_02AAAA_1234"
_S1B_2    = "S1B_IW_SLC__1SSH_20200117T120000_20200117T120028_024100_02BBBB_5678"


class TestSearchGranules:
    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_yields_optical_items(self, mock_pc):
        fake_search = MagicMock()
        fake_search.items.return_value = [
            _fake_item(_LC08_A, _LC08_B, "LC08", 80),
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31"))
        assert len(results) == 1
        assert results[0]["properties"]["platform"] == "LC08"

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_sar_excluded_when_optical_only(self, mock_pc):
        fake_search = MagicMock()
        fake_search.items.return_value = [
            _fake_item(_S1B_1, _S1B_2, "S1B", 90),
            _fake_item(_LC08_A, _LC08_B, "LC08", 75),
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                                       optical_only=True))
        assert len(results) == 1
        assert results[0]["properties"]["platform"] == "LC08"

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_sar_included_when_optical_only_false(self, mock_pc):
        fake_search = MagicMock()
        fake_search.items.return_value = [
            _fake_item(_S1B_1, _S1B_2, "S1B", 90),
            _fake_item(_LC08_A, _LC08_B, "LC08", 75),
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                                       optical_only=False))
        assert len(results) == 2

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_min_valid_pixels_filter(self, mock_pc):
        fake_search = MagicMock()
        fake_search.items.return_value = [
            _fake_item(_LC08_A, _LC08_B, "LC08", 10),   # below threshold
            _fake_item(_LC08_A, _LC08_B, "LC08", 80),   # above threshold
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                                       min_valid_pixels=50))
        assert len(results) == 1
        assert results[0]["properties"]["percent_valid_pixels"] == 80

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_max_dt_days_filter(self, mock_pc):
        fake_search = MagicMock()
        fake_search.items.return_value = [
            _fake_item(_LC08_A, _LC08_B, "LC08", 80, date_dt=180),  # too long
            _fake_item(_LC08_A, _LC08_B, "LC08", 80, date_dt=16),   # ok
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                                       max_dt_days=30))
        assert len(results) == 1
        assert results[0]["properties"]["date_dt"] == 16

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_no_cql2_filter_sent_for_min_valid_pixels(self, mock_pc):
        """min_valid_pixels is applied client-side only — not included in server filter."""
        fake_search = MagicMock()
        fake_search.items.return_value = []
        mock_search = mock_pc.Client.open.return_value.search
        mock_search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                             min_valid_pixels=50))

        _, kwargs = mock_search.call_args
        # The filter sent is the platform filter, not a percent_valid_pixels filter.
        f = kwargs.get("filter", {})
        assert f.get("op") == "in"
        assert f["args"][0] == {"property": "platform"}
        assert "percent_valid_pixels" not in str(f)

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_platform_cql2_filter_sent_when_optical_only(self, mock_pc):
        """A CQL2-JSON platform filter is sent to the server when optical_only=True."""
        fake_search = MagicMock()
        fake_search.items.return_value = []
        mock_search = mock_pc.Client.open.return_value.search
        mock_search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                             optical_only=True))

        _, kwargs = mock_search.call_args
        assert "filter" in kwargs
        f = kwargs["filter"]
        assert f["op"] == "in"
        assert f["args"][0] == {"property": "platform"}
        platforms = f["args"][1]
        assert "LC08" in platforms
        assert "S2A" in platforms
        assert "S1B" not in platforms
        assert kwargs.get("filter_lang") == "cql2-json"

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_no_platform_filter_when_optical_only_false(self, mock_pc):
        """No CQL2 filter is sent when optical_only=False."""
        fake_search = MagicMock()
        fake_search.items.return_value = []
        mock_search = mock_pc.Client.open.return_value.search
        mock_search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                             optical_only=False))

        _, kwargs = mock_search.call_args
        assert "filter" not in kwargs
        assert "filter_lang" not in kwargs

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_date_window_straddling_pair_included(self, mock_pc):
        """A pair where one scene is inside the window and one is outside is kept."""
        fake_search = MagicMock()
        # scene_1: 2019-12-25 (before window), scene_2: 2020-01-01 (inside window)
        fake_search.items.return_value = [
            _fake_item(_LC08_PRE, _LC08_A, "LC08", 80),
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31"))
        assert len(results) == 1

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_date_window_both_scenes_outside_excluded(self, mock_pc):
        """A pair where both scenes are outside the window is dropped."""
        fake_search = MagicMock()
        # scene_1: 2019-12-25, scene_2: 2019-09-01 — both before window
        fake_search.items.return_value = [
            _fake_item(_LC08_PRE, _LC08_OLD, "LC08", 80),
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31"))
        assert len(results) == 0

    @patch("itslive_cloudfree.stac_client.pystac_client")
    def test_max_items_cap(self, mock_pc):
        fake_search = MagicMock()
        fake_search.items.return_value = [
            _fake_item(_LC08_A, _LC08_B, "LC08", 80),
            _fake_item(_LC08_A, _LC08_B, "LC08", 70),
            _fake_item(_LC08_A, _LC08_B, "LC08", 60),
        ]
        mock_pc.Client.open.return_value.search.return_value = fake_search

        from itslive_cloudfree.stac_client import search_granules
        results = list(search_granules((-102, -76, -99, -74), "2020-01-01", "2020-03-31",
                                       max_items=2))
        assert len(results) == 2


class TestIntegration:
    """Live API test — skipped unless --run-integration flag is passed."""

    @pytest.mark.integration
    def test_pine_island_returns_results(self):
        """Query the real ItsLive STAC API and check we get at least one scene."""
        from itslive_cloudfree import search
        results = search(
            bbox=(-102.0, -76.0, -99.0, -74.0),
            start_date="2019-10-01",
            end_date="2020-03-31",
            max_granules=50,
        )
        assert len(results) > 0
        assert all(r.cloud_free_score >= 0 for r in results)
        assert results[0].cloud_free_score >= results[-1].cloud_free_score
