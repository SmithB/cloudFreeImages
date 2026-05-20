"""Unit tests for itslive_cloudfree.crs_utils."""

import pytest

pytest.importorskip("pyproj", reason="pyproj not installed")

from itslive_cloudfree.crs_utils import transform_bbox, bbox_to_wgs84


class TestTransformBbox:
    def test_identity_wgs84(self):
        """No-op when src and dst CRS are both WGS-84."""
        bbox = (-102.0, -76.0, -99.0, -74.0)
        result = transform_bbox(bbox, "EPSG:4326", "EPSG:4326")
        assert result == bbox

    def test_epsg3031_to_wgs84_contains_pine_island(self):
        """EPSG:3031 bbox around Pine Island Glacier transforms to sensible lon/lat."""
        from pyproj import Transformer
        # Convert the WGS-84 corner to EPSG:3031 to get a known input bbox
        t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        x0, y0 = t.transform(-102.0, -76.0)
        x1, y1 = t.transform(-99.0, -74.0)
        bbox_3031 = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        result = transform_bbox(bbox_3031, "EPSG:3031", "EPSG:4326")
        min_lon, min_lat, max_lon, max_lat = result

        # The result should enclose the original WGS-84 corners
        assert min_lon <= -102.0
        assert max_lon >= -99.0
        assert min_lat <= -76.0
        assert max_lat >= -74.0

    def test_integer_epsg_code(self):
        """Integer EPSG code (3031) is accepted as src_crs."""
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        x0, y0 = t.transform(-101.0, -75.5)
        x1, y1 = t.transform(-100.0, -74.5)
        bbox_3031 = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        result = transform_bbox(bbox_3031, 3031)  # integer, no "EPSG:" prefix
        min_lon, min_lat, max_lon, max_lat = result

        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 0   # southern hemisphere

    def test_bbox_to_wgs84_convenience(self):
        """bbox_to_wgs84 is equivalent to transform_bbox(..., dst_crs='EPSG:4326')."""
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        x0, y0 = t.transform(-101.0, -75.0)
        x1, y1 = t.transform(-100.0, -74.0)
        bbox_3031 = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        r1 = transform_bbox(bbox_3031, 3031, "EPSG:4326")
        r2 = bbox_to_wgs84(bbox_3031, 3031)
        assert r1 == r2

    def test_result_is_valid_wgs84_envelope(self):
        """Transformed result stays within WGS-84 bounds."""
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        # A mid-latitude UTM bbox
        x0, y0 = t.transform(10.0, 48.0)
        x1, y1 = t.transform(12.0, 50.0)
        bbox_utm = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        result = bbox_to_wgs84(bbox_utm, "EPSG:32633")
        min_lon, min_lat, max_lon, max_lat = result

        assert -180 <= min_lon < max_lon <= 180
        assert -90 <= min_lat < max_lat <= 90
