"""Unit tests for itslive_cloudfree.scorer."""

import math
import pytest

from itslive_cloudfree.scorer import score_scenes
from itslive_cloudfree.results import SceneResult


def _make_granule(scene_1_id, scene_2_id, platform, valid_pct):
    return {
        "type": "Feature",
        "properties": {
            "platform": platform,
            "scene_1_id": scene_1_id,
            "scene_2_id": scene_2_id,
            "percent_valid_pixels": valid_pct,
            "date_dt": 16.0,
        },
    }


# Real-format Landsat scene IDs to exercise the parser
_SCENE_A = "LC08_L1GT_224115_20200101_20200113_02_T2"
_SCENE_B = "LC08_L1GT_224115_20200117_20200129_02_T2"
_SCENE_C = "LC08_L1GT_224115_20200202_20200214_02_T2"
_SCENE_D = "LC09_L1GT_224115_20200218_20200301_02_T2"


class TestScoreScenes:
    def test_empty_input(self):
        assert score_scenes([]) == []

    def test_sar_granules_excluded(self):
        """Sentinel-1 granules should not produce any SceneResult."""
        s1_sid = "S1B_IW_SLC__1SSH_20200101T120000_20200101T120028_024000_02AAAA_1234"
        granules = [_make_granule(s1_sid, s1_sid, "S1B", 80)]
        assert score_scenes(granules) == []

    def test_single_pair_returns_two_scenes(self):
        granules = [_make_granule(_SCENE_A, _SCENE_B, "LC08", 75)]
        results = score_scenes(granules)
        scene_ids = {r.scene_id for r in results}
        assert _SCENE_A in scene_ids
        assert _SCENE_B in scene_ids

    def test_highest_scoring_scene_gets_100(self):
        granules = [
            _make_granule(_SCENE_A, _SCENE_B, "LC08", 90),
            _make_granule(_SCENE_A, _SCENE_C, "LC08", 85),
        ]
        results = score_scenes(granules)
        assert results[0].cloud_free_score == pytest.approx(100.0)

    def test_scene_appearing_in_more_pairs_scores_higher(self):
        # SCENE_A appears in 3 pairs; SCENE_D appears in only 1
        granules = [
            _make_granule(_SCENE_A, _SCENE_B, "LC08", 80),
            _make_granule(_SCENE_A, _SCENE_C, "LC08", 80),
            _make_granule(_SCENE_A, _SCENE_D, "LC09", 80),
            _make_granule(_SCENE_B, _SCENE_D, "LC09", 80),
        ]
        results = score_scenes(granules)
        score_map = {r.scene_id: r.cloud_free_score for r in results}
        assert score_map[_SCENE_A] > score_map[_SCENE_D]

    def test_higher_valid_pixels_scores_higher(self):
        # Two scenes each appearing in exactly 1 pair; differ only in valid_pct
        sid_high = "LC08_L1GT_224115_20200101_20200113_02_T2"
        sid_low  = "LC08_L1GT_225115_20200101_20200113_02_T2"
        granules = [
            _make_granule(sid_high, _SCENE_B, "LC08", 95),
            _make_granule(sid_low,  _SCENE_C, "LC08", 20),
        ]
        results = score_scenes(granules)
        score_map = {r.scene_id: r.cloud_free_score for r in results}
        assert score_map[sid_high] > score_map[sid_low]

    def test_results_sorted_descending(self):
        granules = [
            _make_granule(_SCENE_A, _SCENE_B, "LC08", 90),
            _make_granule(_SCENE_A, _SCENE_C, "LC08", 85),
            _make_granule(_SCENE_D, _SCENE_B, "LC09", 30),
        ]
        results = score_scenes(granules)
        scores = [r.cloud_free_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_pair_count_correct(self):
        granules = [
            _make_granule(_SCENE_A, _SCENE_B, "LC08", 80),
            _make_granule(_SCENE_A, _SCENE_C, "LC08", 70),
            _make_granule(_SCENE_A, _SCENE_D, "LC09", 60),
        ]
        results = score_scenes(granules)
        scene_a = next(r for r in results if r.scene_id == _SCENE_A)
        assert scene_a.pair_count == 3

    def test_mean_valid_pixels_correct(self):
        granules = [
            _make_granule(_SCENE_A, _SCENE_B, "LC08", 80),
            _make_granule(_SCENE_A, _SCENE_C, "LC08", 60),
        ]
        results = score_scenes(granules)
        scene_a = next(r for r in results if r.scene_id == _SCENE_A)
        assert scene_a.mean_valid_pixels == pytest.approx(70.0)

    def test_platform_parsed_correctly(self):
        granules = [_make_granule(_SCENE_A, _SCENE_B, "LC08", 80)]
        results = score_scenes(granules)
        platforms = {r.platform for r in results}
        assert "LC08" in platforms

    def test_path_row_parsed_for_landsat(self):
        granules = [_make_granule(_SCENE_A, _SCENE_B, "LC08", 80)]
        results = score_scenes(granules)
        scene_a = next(r for r in results if r.scene_id == _SCENE_A)
        assert scene_a.path_row == "224/115"
