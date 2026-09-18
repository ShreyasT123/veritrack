"""Directed road network graph for Stage 3.

The network is a NetworkX :class:`~networkx.MultiDiGraph`. Both qualifiers
matter:

* **Directed**, because one-way compliance is real evidence. A path that
  requires driving the wrong way up a one-way street is not a path, and anomaly
  Condition C (``UNREACHABLE_TRANSITION``) depends on the graph refusing to
  find one.
* **Multi**, because two nodes can be joined by more than one road -- a
  surface street and the flyover above it, a carriageway and its service road.
  Collapsing those into a single edge would let the matcher silently place a
  vehicle on the wrong one.

Shortest-path performance
-------------------------
Network distance is the hottest query in Stage 3: the Viterbi transition model
calls it once per candidate pair per timestep, so a trellis with 8 candidates
over 20 sightings issues on the order of 1,200 queries for one trajectory.

Two things make that affordable:

1. **A\\* with a haversine heuristic** instead of plain Dijkstra. The heuristic
   is admissible -- a road segment is never shorter than the straight line
   between its endpoints -- so A\\* returns the same optimal distance while
   expanding far fewer nodes on a road network, whose geometry is strongly
   Euclidean.
2. **An LRU cache** keyed on the ordered node pair. Queries cluster hard on the
   camera set rather than spreading over all node pairs, so the hit rate in
   steady state is high and a repeated query is a dictionary lookup.

Being straight about the sub-millisecond target: a *cached* query is
sub-microsecond, and a cold A\\* query on a city-scale graph is typically a few
hundred microseconds. The cache is what delivers the headline number, and
``max_search_distance_m`` is what bounds the cold-path tail.

Candidate search
----------------
Finding the road segments within 50 m of a camera is done in a local
equirectangular projection rather than on the sphere. At city scale the
distortion is negligible, and it turns "distance from a point to a segment"
into plane geometry -- a projection onto a line, clamped to the segment -- which
is both exact and fast. A KD-tree over segment midpoints prunes the candidate
set before any exact distance is computed.
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import networkx as nx

try:  # pragma: no cover - optional acceleration
    from scipy.spatial import cKDTree

    _KDTREE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _KDTREE_AVAILABLE = False

from .config import GraphConfig
from .types import (
    EARTH_RADIUS_M,
    GeoPoint,
    RoadSegment,
    angular_difference_deg,
    bearing_deg,
    haversine_m,
)

__all__ = [
    "GraphError",
    "NodeNotFound",
    "EdgeCandidate",
    "RoadNetworkGraph",
    "build_synthetic_grid",
]

#: Returned by :meth:`RoadNetworkGraph.network_distance` when no directed path
#: exists. Chosen over ``None`` so that arithmetic downstream stays total, and
#: over a sentinel float so comparisons behave.
UNREACHABLE = float("inf")


class GraphError(RuntimeError):
    """Road network fault."""


class NodeNotFound(GraphError):
    """A referenced node is absent from the graph."""


class EdgeCandidate:
    """A road segment proposed as the true location of a camera sighting."""

    __slots__ = ("segment", "perpendicular_distance_m", "projection", "heading_difference_deg",
                 "along_fraction")

    def __init__(
        self,
        segment: RoadSegment,
        perpendicular_distance_m: float,
        projection: GeoPoint,
        heading_difference_deg: float,
        along_fraction: float,
    ) -> None:
        self.segment = segment
        self.perpendicular_distance_m = perpendicular_distance_m
        self.projection = projection
        self.heading_difference_deg = heading_difference_deg
        self.along_fraction = along_fraction

    def __repr__(self) -> str:
        return (
            f"EdgeCandidate({self.segment.segment_id}, "
            f"d_perp={self.perpendicular_distance_m:.1f}m, "
            f"dtheta={self.heading_difference_deg:.1f}deg)"
        )


class RoadNetworkGraph:
    """Directed multigraph of the road network, with spatial and metric queries."""

    __slots__ = (
        "_graph", "_config", "_segments", "_version", "_distance_cache",
        "_projection_origin", "_kdtree", "_kdtree_segments", "_index_dirty",
        "_segment_xy", "_cache_hits", "_cache_misses",
        "_node_kdtree", "_node_ids", "_nearest_node_cache", "_corridor_cache",
    )

    def __init__(self, config: Optional[GraphConfig] = None) -> None:
        self._config = config if config is not None else GraphConfig()
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._segments: Dict[str, RoadSegment] = {}
        self._version = 0
        self._projection_origin: Optional[Tuple[float, float]] = None
        self._kdtree: Optional[Any] = None
        self._kdtree_segments: List[RoadSegment] = []
        self._segment_xy: List[Tuple[float, float, float, float]] = []
        self._index_dirty = True
        self._cache_hits = 0
        self._cache_misses = 0
        self._node_kdtree: Optional[Any] = None
        self._node_ids: List[str] = []
        self._nearest_node_cache: Dict[Tuple[float, float], Tuple[str, float]] = {}
        self._corridor_cache: Dict[Tuple[str, str], float] = {}
        self._distance_cache = self._make_distance_cache()

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        latitude: float,
        longitude: float,
        *,
        node_type: str = "junction",
        **attributes: Any,
    ) -> None:
        """Add or update an intersection, camera pole or junction."""
        point = GeoPoint(latitude=latitude, longitude=longitude)
        self._graph.add_node(
            node_id,
            latitude=point.latitude,
            longitude=point.longitude,
            node_type=node_type,
            **attributes,
        )
        self._invalidate()

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        segment_id: Optional[str] = None,
        length_m: Optional[float] = None,
        speed_limit_kmh: Optional[float] = None,
        street_name: str = "",
        corridor_id: Optional[str] = None,
        heading_deg: Optional[float] = None,
        bidirectional: bool = False,
        **attributes: Any,
    ) -> RoadSegment:
        """Add a directed road segment.

        ``length_m`` and ``heading_deg`` default to the geodesic values between
        the endpoints. Real OSM geometry is polylined, so a production loader
        should pass the true ``length_m`` from the way's geometry rather than
        accept the straight-line default.
        """
        for node in (from_node, to_node):
            if node not in self._graph:
                raise NodeNotFound(f"node {node!r} must be added before an edge referencing it")
        if from_node == to_node:
            raise GraphError(f"self-loop rejected at node {from_node!r}")

        from_point = self.node_point(from_node)
        to_point = self.node_point(to_node)
        resolved_length = (
            float(length_m) if length_m is not None else from_point.distance_to(to_point)
        )
        if resolved_length <= 0.0:
            raise GraphError(f"segment {from_node}->{to_node} has non-positive length")
        resolved_speed = (
            float(speed_limit_kmh)
            if speed_limit_kmh is not None
            else self._config.default_speed_limit_kmh
        )
        resolved_heading = (
            float(heading_deg) if heading_deg is not None else from_point.bearing_to(to_point)
        )
        resolved_id = segment_id or f"{from_node}->{to_node}"

        key = self._graph.add_edge(
            from_node,
            to_node,
            segment_id=resolved_id,
            length_m=resolved_length,
            speed_limit_kmh=resolved_speed,
            heading_deg=resolved_heading,
            street_name=street_name,
            corridor_id=corridor_id,
            free_flow_time_sec=resolved_length / (resolved_speed / 3.6),
            **attributes,
        )
        segment = RoadSegment(
            segment_id=resolved_id,
            from_node=from_node,
            to_node=to_node,
            from_point=from_point,
            to_point=to_point,
            length_m=resolved_length,
            speed_limit_kmh=resolved_speed,
            heading_deg=resolved_heading,
            street_name=street_name,
            edge_key=key,
            corridor_id=corridor_id,
            is_oneway=not bidirectional,
        )
        self._segments[resolved_id] = segment
        self._invalidate()

        if bidirectional:
            self.add_edge(
                to_node,
                from_node,
                segment_id=f"{resolved_id}:rev",
                length_m=resolved_length,
                speed_limit_kmh=resolved_speed,
                street_name=street_name,
                corridor_id=corridor_id,
                bidirectional=False,
                **attributes,
            )
        return segment

    def _invalidate(self) -> None:
        """Drop derived state. Any mutation must call this.

        A stale distance cache after an edge is added would return a path
        length that the graph no longer supports -- the kind of bug that only
        shows up after a live road-closure update.
        """
        self._version += 1
        self._index_dirty = True
        self._kdtree = None
        self._node_kdtree = None
        self._nearest_node_cache.clear()
        self._corridor_cache.clear()
        self._distance_cache = self._make_distance_cache()
        self._cache_hits = 0
        self._cache_misses = 0

    # -----------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------

    @property
    def graph(self) -> nx.MultiDiGraph:
        return self._graph

    @property
    def config(self) -> GraphConfig:
        return self._config

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def has_node(self, node_id: str) -> bool:
        return node_id in self._graph

    def node_point(self, node_id: str) -> GeoPoint:
        try:
            data = self._graph.nodes[node_id]
        except KeyError as exc:
            raise NodeNotFound(f"unknown node {node_id!r}") from exc
        return GeoPoint(latitude=data["latitude"], longitude=data["longitude"])

    def segment(self, segment_id: str) -> RoadSegment:
        try:
            return self._segments[segment_id]
        except KeyError as exc:
            raise GraphError(f"unknown segment {segment_id!r}") from exc

    def segments(self) -> Iterator[RoadSegment]:
        return iter(self._segments.values())

    def cache_stats(self) -> Dict[str, Any]:
        info = self._distance_cache.cache_info()
        total = info.hits + info.misses
        return {
            "hits": info.hits,
            "misses": info.misses,
            "hit_rate": (info.hits / total) if total else 0.0,
            "size": info.currsize,
            "maxsize": info.maxsize,
        }

    # -----------------------------------------------------------------
    # Projection and spatial index
    # -----------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Build the local projection and the KD-tree over segment midpoints."""
        if not self._index_dirty:
            return
        if not self._segments:
            self._projection_origin = None
            self._kdtree = None
            self._node_kdtree = None
            self._node_ids = []
            self._kdtree_segments = []
            self._segment_xy = []
            self._index_dirty = False
            return

        latitudes = [self._graph.nodes[n]["latitude"] for n in self._graph.nodes]
        longitudes = [self._graph.nodes[n]["longitude"] for n in self._graph.nodes]
        origin = (sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes))
        self._projection_origin = origin

        self._kdtree_segments = list(self._segments.values())
        midpoints: List[Tuple[float, float]] = []
        self._segment_xy = []
        for segment in self._kdtree_segments:
            x1, y1 = self._project(segment.from_point)
            x2, y2 = self._project(segment.to_point)
            self._segment_xy.append((x1, y1, x2, y2))
            midpoints.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))

        self._node_ids = list(self._graph.nodes)
        node_points = [
            self._project(
                GeoPoint(
                    latitude=self._graph.nodes[node]["latitude"],
                    longitude=self._graph.nodes[node]["longitude"],
                )
            )
            for node in self._node_ids
        ]

        if _KDTREE_AVAILABLE and midpoints:
            self._kdtree = cKDTree(midpoints)
            self._node_kdtree = cKDTree(node_points) if node_points else None
        else:  # pragma: no cover - exercised only without scipy
            self._kdtree = None
            self._node_kdtree = None
        self._index_dirty = False

    def _project(self, point: GeoPoint) -> Tuple[float, float]:
        """Local equirectangular projection to metres.

        Valid for a city-sized extent. The cos(lat0) factor is what makes
        eastings and northings commensurate; omitting it is the classic bug
        that makes a 50 m radius behave like 50 m north and 35 m east in Mumbai.
        """
        if self._projection_origin is None:
            raise GraphError("projection origin is not established; add nodes first")
        lat0, lon0 = self._projection_origin
        cos_lat0 = math.cos(math.radians(lat0))
        x = math.radians(point.longitude - lon0) * cos_lat0 * EARTH_RADIUS_M
        y = math.radians(point.latitude - lat0) * EARTH_RADIUS_M
        return x, y

    def _unproject(self, x: float, y: float) -> GeoPoint:
        if self._projection_origin is None:
            raise GraphError("projection origin is not established")
        lat0, lon0 = self._projection_origin
        cos_lat0 = math.cos(math.radians(lat0))
        latitude = lat0 + math.degrees(y / EARTH_RADIUS_M)
        longitude = lon0 + math.degrees(x / (EARTH_RADIUS_M * cos_lat0))
        return GeoPoint(latitude=latitude, longitude=longitude)

    @staticmethod
    def _point_to_segment(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float, float, float]:
        """Perpendicular distance from a point to a finite segment.

        Returns (distance, projected_x, projected_y, along_fraction). The
        projection parameter is clamped to [0, 1] so that a camera beyond the
        end of a segment measures to the endpoint, not to the infinite line --
        without the clamp, a distant parallel road would appear adjacent.
        """
        dx, dy = x2 - x1, y2 - y1
        denominator = dx * dx + dy * dy
        if denominator <= 1e-12:
            return math.hypot(px - x1, py - y1), x1, y1, 0.0
        t = ((px - x1) * dx + (py - y1) * dy) / denominator
        t = max(0.0, min(1.0, t))
        projected_x = x1 + t * dx
        projected_y = y1 + t * dy
        return math.hypot(px - projected_x, py - projected_y), projected_x, projected_y, t

    def candidate_edges(
        self,
        location: GeoPoint,
        *,
        radius_m: float = 50.0,
        observed_heading_deg: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ) -> List[EdgeCandidate]:
        """Road segments within ``radius_m`` of ``location``, nearest first.

        This is the Viterbi candidate-pruning step. Keeping the radius tight
        matters more than it looks: the trellis is ``O(T |C|^2)``, so doubling
        the radius on a dense junction roughly quadruples the matching cost.
        """
        self._ensure_index()
        if not self._kdtree_segments:
            return []

        px, py = self._project(location)
        if self._kdtree is not None:
            # A segment's midpoint can be further from the camera than the
            # segment itself, by at most half the segment length. Inflate the
            # midpoint query radius accordingly so no true candidate is missed.
            longest = max(segment.length_m for segment in self._kdtree_segments)
            indices = self._kdtree.query_ball_point(
                [px, py], r=radius_m + longest * 0.5 + 1.0
            )
        else:  # pragma: no cover
            indices = range(len(self._kdtree_segments))

        candidates: List[EdgeCandidate] = []
        for index in indices:
            segment = self._kdtree_segments[index]
            x1, y1, x2, y2 = self._segment_xy[index]
            distance, proj_x, proj_y, fraction = self._point_to_segment(px, py, x1, y1, x2, y2)
            if distance > radius_m:
                continue
            if observed_heading_deg is None:
                heading_difference = 0.0
            else:
                heading_difference = angular_difference_deg(
                    observed_heading_deg, segment.heading_deg
                )
            candidates.append(
                EdgeCandidate(
                    segment=segment,
                    perpendicular_distance_m=distance,
                    projection=self._unproject(proj_x, proj_y),
                    heading_difference_deg=heading_difference,
                    along_fraction=fraction,
                )
            )

        candidates.sort(key=lambda c: (c.perpendicular_distance_m, c.heading_difference_deg))
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        return candidates

    def nearest_node(self, location: GeoPoint) -> Tuple[str, float]:
        """Closest graph node to a point, with its distance in metres.

        Backed by the same KD-tree as candidate search. The linear scan this
        replaced was the single hottest call in Stage 3: the anomaly detector
        and the fusion scorer each resolve both endpoints of every consecutive
        pair, so a 15-sighting trajectory issued hundreds of scans over the
        whole node set and spent most of its wall clock inside ``haversine``.

        Results are memoised on the rounded coordinate. Camera poles are
        fixed, so the same handful of positions is resolved over and over
        across a batch; rounding to roughly 1 cm collapses those to one entry
        without ever merging two genuinely distinct poles.
        """
        if not self._graph.nodes:
            raise GraphError("graph has no nodes")
        key = (round(location.latitude, 7), round(location.longitude, 7))
        cached = self._nearest_node_cache.get(key)
        if cached is not None:
            return cached

        self._ensure_index()
        result: Tuple[str, float]
        if self._node_kdtree is not None:
            px, py = self._project(location)
            _, index = self._node_kdtree.query([px, py], k=1)
            node_id = self._node_ids[int(index)]
            data = self._graph.nodes[node_id]
            result = (
                node_id,
                haversine_m(
                    location.latitude, location.longitude, data["latitude"], data["longitude"]
                ),
            )
        else:  # pragma: no cover - exercised only without scipy
            best_node = ""
            best_distance = UNREACHABLE
            for node_id, data in self._graph.nodes(data=True):
                distance = haversine_m(
                    location.latitude, location.longitude, data["latitude"], data["longitude"]
                )
                if distance < best_distance:
                    best_node, best_distance = node_id, distance
            result = (best_node, best_distance)

        # Bound the memo: a pathological caller passing continuously varying
        # coordinates must not grow this without limit.
        if len(self._nearest_node_cache) < 50_000:
            self._nearest_node_cache[key] = result
        return result

    # -----------------------------------------------------------------
    # Shortest paths
    # -----------------------------------------------------------------

    def _make_distance_cache(self) -> Any:
        """Build the LRU-wrapped distance function bound to this graph version."""

        @lru_cache(maxsize=self._config.distance_cache_size)
        def cached_distance(from_node: str, to_node: str) -> float:
            return self._shortest_path_length(from_node, to_node)

        return cached_distance

    def _heuristic(self, node: str, target: str) -> float:
        if not self._config.use_astar_heuristic:
            return 0.0
        source_data = self._graph.nodes[node]
        target_data = self._graph.nodes[target]
        return haversine_m(
            source_data["latitude"],
            source_data["longitude"],
            target_data["latitude"],
            target_data["longitude"],
        )

    def _shortest_path_length(self, from_node: str, to_node: str) -> float:
        """A* over the directed multigraph. Returns UNREACHABLE if no path exists."""
        if from_node == to_node:
            return 0.0
        if from_node not in self._graph:
            raise NodeNotFound(f"unknown node {from_node!r}")
        if to_node not in self._graph:
            raise NodeNotFound(f"unknown node {to_node!r}")

        limit = self._config.max_search_distance_m
        best_cost: Dict[str, float] = {from_node: 0.0}
        frontier: List[Tuple[float, float, str]] = [
            (self._heuristic(from_node, to_node), 0.0, from_node)
        ]
        visited: set[str] = set()

        while frontier:
            _, cost, node = heapq.heappop(frontier)
            if node in visited:
                continue
            if node == to_node:
                return cost
            visited.add(node)
            if cost > limit:
                # Everything still in the frontier costs at least this much,
                # so the target is beyond the search horizon.
                break
            for _, neighbour, data in self._graph.out_edges(node, data=True):
                if neighbour in visited:
                    continue
                # A MultiDiGraph can hold parallel edges; out_edges yields each
                # one, and the relaxation below naturally keeps the cheapest.
                new_cost = cost + float(data["length_m"])
                if new_cost > limit:
                    continue
                if new_cost < best_cost.get(neighbour, UNREACHABLE):
                    best_cost[neighbour] = new_cost
                    heapq.heappush(
                        frontier,
                        (new_cost + self._heuristic(neighbour, to_node), new_cost, neighbour),
                    )
        return UNREACHABLE

    def network_distance(self, from_node: str, to_node: str) -> float:
        """Cached shortest directed path length in metres, or ``inf``."""
        return self._distance_cache(from_node, to_node)

    def is_reachable(self, from_node: str, to_node: str) -> bool:
        """True when a directed path exists inside the search horizon."""
        return math.isfinite(self.network_distance(from_node, to_node))

    def shortest_path_nodes(self, from_node: str, to_node: str) -> Optional[List[str]]:
        """The node sequence of the optimal path, or None if unreachable."""
        if from_node == to_node:
            return [from_node]
        for node in (from_node, to_node):
            if node not in self._graph:
                raise NodeNotFound(f"unknown node {node!r}")
        try:
            if self._config.use_astar_heuristic:
                return nx.astar_path(
                    self._graph,
                    from_node,
                    to_node,
                    heuristic=lambda a, b: self._heuristic(a, b),
                    weight="length_m",
                )
            return nx.shortest_path(self._graph, from_node, to_node, weight="length_m")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def shortest_path_segments(self, from_node: str, to_node: str) -> Optional[List[RoadSegment]]:
        """The segment sequence of the optimal path, or None if unreachable.

        Between each consecutive node pair the cheapest parallel edge is
        selected, which is what resolves the surface-street-versus-flyover
        ambiguity in favour of the shorter route.
        """
        nodes = self.shortest_path_nodes(from_node, to_node)
        if nodes is None:
            return None
        segments: List[RoadSegment] = []
        for left, right in zip(nodes, nodes[1:]):
            edge_bundle = self._graph.get_edge_data(left, right)
            if not edge_bundle:
                return None
            best_key = min(edge_bundle, key=lambda k: edge_bundle[k]["length_m"])
            segments.append(self._segments[edge_bundle[best_key]["segment_id"]])
        return segments

    def travel_time_s(self, from_node: str, to_node: str) -> float:
        """Free-flow travel time along the shortest path, in seconds."""
        segments = self.shortest_path_segments(from_node, to_node)
        if segments is None:
            return UNREACHABLE
        return sum(float(segment.free_flow_time_sec or 0.0) for segment in segments)

    def corridor_free_flow_kmh(self, from_node: str, to_node: str, fallback: float) -> float:
        """Length-weighted mean free-flow speed along the shortest path.

        Length-weighted, not arithmetic: a 2 km arterial at 60 km/h followed by
        a 50 m slip road at 20 km/h is a 60 km/h corridor, and a plain mean
        would call it 40.
        """
        key = (from_node, to_node)
        cached = self._corridor_cache.get(key)
        if cached is not None:
            return cached if cached > 0.0 else fallback

        segments = self.shortest_path_segments(from_node, to_node)
        if not segments:
            # Memoise the miss too: an unreachable pair is re-queried once per
            # trellis cell, and re-running a failed A* each time is pure waste.
            if len(self._corridor_cache) < 200_000:
                self._corridor_cache[key] = -1.0
            return fallback
        total_length = sum(segment.length_m for segment in segments)
        if total_length <= 0.0:
            return fallback
        weighted = sum(s.speed_limit_kmh * s.length_m for s in segments) / total_length
        if len(self._corridor_cache) < 200_000:
            self._corridor_cache[key] = weighted
        return weighted

    # -----------------------------------------------------------------
    # Loaders
    # -----------------------------------------------------------------

    @classmethod
    def from_osm_like(
        cls,
        nodes: Sequence[Mapping[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        config: Optional[GraphConfig] = None,
    ) -> "RoadNetworkGraph":
        """Build from OSM-shaped records.

        ``nodes`` need ``id``/``lat``/``lon``; ``edges`` need ``from``/``to``
        plus any of ``length_m``, ``maxspeed``, ``name``, ``oneway``,
        ``corridor_id``. This is the shape an ``osmnx`` export or an Overpass
        query reduces to, so a production loader is a field-mapping exercise
        rather than a rewrite.
        """
        network = cls(config)
        for node in nodes:
            network.add_node(
                str(node["id"]),
                float(node["lat"]),
                float(node["lon"]),
                node_type=str(node.get("type", "junction")),
            )
        for edge in edges:
            oneway = bool(edge.get("oneway", True))
            network.add_edge(
                str(edge["from"]),
                str(edge["to"]),
                segment_id=str(edge["id"]) if "id" in edge else None,
                length_m=float(edge["length_m"]) if "length_m" in edge else None,
                speed_limit_kmh=(
                    float(edge["maxspeed"]) if edge.get("maxspeed") is not None else None
                ),
                street_name=str(edge.get("name", "")),
                corridor_id=edge.get("corridor_id"),
                bidirectional=not oneway,
            )
        return network


def build_synthetic_grid(
    rows: int = 5,
    columns: int = 5,
    *,
    spacing_m: float = 500.0,
    origin_lat: float = 19.0330,
    origin_lon: float = 73.0297,
    speed_limit_kmh: float = 50.0,
    bidirectional: bool = True,
    config: Optional[GraphConfig] = None,
) -> RoadNetworkGraph:
    """A Manhattan grid for testing, benchmarking and demos.

    Defaults are anchored on Navi Mumbai so that synthetic coordinates land in
    a plausible place on a map during a demo. Nodes are named ``n{row}_{col}``
    and streets are named so that a reconstructed path reads like a real route
    ("5th Avenue, 3rd Street").

    With ``bidirectional=False`` the grid becomes a strictly one-way lattice
    (all eastbound and southbound), which is the fixture that makes anomaly
    Condition C reachable: travelling north-west is then topologically
    impossible rather than merely long.
    """
    if rows < 2 or columns < 2:
        raise ValueError("a grid needs at least 2 rows and 2 columns")

    network = RoadNetworkGraph(config)
    # Metres-per-degree at the origin latitude; the longitude scale shrinks by
    # cos(lat), which is what keeps the grid square on the ground.
    lat_step = math.degrees(spacing_m / EARTH_RADIUS_M)
    lon_step = math.degrees(spacing_m / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat))))

    for row in range(rows):
        for column in range(columns):
            network.add_node(
                f"n{row}_{column}",
                origin_lat + row * lat_step,
                origin_lon + column * lon_step,
                node_type="junction",
            )

    for row in range(rows):
        for column in range(columns):
            if column + 1 < columns:
                network.add_edge(
                    f"n{row}_{column}",
                    f"n{row}_{column + 1}",
                    length_m=spacing_m,
                    speed_limit_kmh=speed_limit_kmh,
                    street_name=f"{row + 1}th Avenue",
                    corridor_id=f"avenue-{row}",
                    bidirectional=bidirectional,
                )
            if row + 1 < rows:
                network.add_edge(
                    f"n{row}_{column}",
                    f"n{row + 1}_{column}",
                    length_m=spacing_m,
                    speed_limit_kmh=speed_limit_kmh,
                    street_name=f"{column + 1}th Street",
                    corridor_id=f"street-{column}",
                    bidirectional=bidirectional,
                )
    return network
