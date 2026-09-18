"""Uber H3 spatial indexing and GeoJSON conversion helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

try:
    import h3
except ImportError as exc:  # pragma: no cover - dependency failure is deployment-specific
    raise ImportError("veritrack_analytics requires the 'h3' package (h3-py >= 4)") from exc

__all__ = ["H3SpatialIndex"]


class H3SpatialIndex:
    """Thin, version-isolated adapter around the H3 v4 Python API."""

    @staticmethod
    def cell(latitude: float, longitude: float, resolution: int) -> str:
        H3SpatialIndex._validate_coordinates(latitude, longitude)
        H3SpatialIndex._validate_resolution(resolution)
        return str(h3.latlng_to_cell(latitude, longitude, resolution))

    @staticmethod
    def centroid(cell: str) -> Tuple[float, float]:
        H3SpatialIndex._validate_cell(cell)
        latitude, longitude = h3.cell_to_latlng(cell)
        return (float(latitude), float(longitude))

    @staticmethod
    def resolution(cell: str) -> int:
        H3SpatialIndex._validate_cell(cell)
        return int(h3.get_resolution(cell))

    @staticmethod
    def k_ring(cell: str, distance: int) -> Tuple[str, ...]:
        H3SpatialIndex._validate_cell(cell)
        if distance < 0:
            raise ValueError("k-ring distance cannot be negative")
        return tuple(sorted(str(value) for value in h3.grid_disk(cell, distance)))

    @staticmethod
    def boundary_lon_lat(cell: str) -> Tuple[Tuple[float, float], ...]:
        """Return a closed GeoJSON ring in longitude, latitude order."""
        H3SpatialIndex._validate_cell(cell)
        ring = tuple((float(lon), float(lat)) for lat, lon in h3.cell_to_boundary(cell))
        if not ring:
            raise ValueError(f"H3 cell {cell!r} has no boundary")
        return ring + (ring[0],)

    @classmethod
    def polygon_feature(cls, cell: str, properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "properties": {"h3_index": cell, "resolution": cls.resolution(cell), **(properties or {})},
            "geometry": {"type": "Polygon", "coordinates": [[list(point) for point in cls.boundary_lon_lat(cell)]]},
        }

    @classmethod
    def feature_collection(cls, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {"type": "FeatureCollection", "features": list(features)}

    @staticmethod
    def _validate_coordinates(latitude: float, longitude: float) -> None:
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError("coordinates must be valid WGS84 latitude/longitude")

    @staticmethod
    def _validate_resolution(resolution: int) -> None:
        if not 0 <= resolution <= 15:
            raise ValueError("H3 resolution must lie in [0, 15]")

    @staticmethod
    def _validate_cell(cell: str) -> None:
        if not isinstance(cell, str) or not h3.is_valid_cell(cell):
            raise ValueError(f"invalid H3 cell: {cell!r}")
