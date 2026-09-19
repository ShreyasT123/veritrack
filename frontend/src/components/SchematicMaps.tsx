import type { StopItem } from "../data/veritrackData";
import {
  AREAS,
  BLOCKS,
  NODES,
  PRESSURE,
  ROADS,
  SEGMENTS,
} from "../data/veritrackData";

const line = (p: [number, number][]) =>
  p.map((q, i) => (i ? "L" : "M") + q[0] + " " + q[1]).join(" ");

const len = (p: [number, number][]) =>
  p.reduce((s, q, i) => {
    if (i === 0) return 0;
    const prev = p[i - 1];
    return prev ? s + Math.hypot(q[0] - prev[0], q[1] - prev[1]) : s;
  }, 0);

export function BaseMapLayers() {
  return (
    <>
      <rect width="760" height="620" fill="var(--land)" />
      {BLOCKS.map((b, i) => (
        <rect
          key={i}
          className="map-block"
          x={b[0]}
          y={b[1]}
          width={b[2]}
          height={b[3]}
        />
      ))}
      {ROADS.map((r, i) => (
        <path
          key={`edge-${i}`}
          className="map-road-edge"
          d={line(r.pts)}
          strokeWidth={r.w + 5}
        />
      ))}
      {ROADS.map((r, i) => (
        <path
          key={`road-${i}`}
          className="map-road"
          d={line(r.pts)}
          strokeWidth={r.w}
        />
      ))}
      {AREAS.map((a) => (
        <text
          key={a.t}
          className="map-area"
          x={a.x}
          y={a.y}
          textAnchor="middle"
        >
          {a.t}
        </text>
      ))}
      <text
        className="map-road-label"
        x="130"
        y="492"
        textAnchor="middle"
        transform="rotate(-28 130 492)"
      >
        National Highway 48
      </text>
      <text
        className="map-road-label"
        x="112"
        y="348"
        textAnchor="middle"
        transform="rotate(-76 112 348)"
      >
        Dwarka Expressway
      </text>
      <text
        className="map-road-label"
        x="690"
        y="218"
        textAnchor="middle"
        transform="rotate(-28 690 218)"
      >
        towards Delhi
      </text>
      <text className="map-road-label" x="46" y="586" textAnchor="middle">
        towards Jaipur
      </text>
    </>
  );
}

export function NodeLayer({ routeIds = [] }: { routeIds?: string[] }) {
  const on = new Set(routeIds);
  return (
    <>
      {Object.entries(NODES).map(([k, d]) => (
        <g key={k}>
          {!on.has(k) && <circle className="map-node" cx={d.x} cy={d.y} r="5" />}
          <text
            className="map-node-label"
            x={d.x + d.dx}
            y={d.y + d.dy}
            textAnchor={d.a}
          >
            {d.n}
          </text>
        </g>
      ))}
    </>
  );
}

export function TraceMap({ stops = [] }: { stops: StopItem[] }) {
  const pts: [number, number][] = stops
    .filter((s) => Boolean(NODES[s.at]))
    .map((s) => [NODES[s.at]!.x, NODES[s.at]!.y]);
  const totalLen = Math.round(len(pts));
  const routeIds = stops.map((s) => s.at);

  return (
    <svg
      viewBox="0 0 760 620"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Map of the route this vehicle took across the city"
      xmlns="http://www.w3.org/2000/svg"
    >
      <BaseMapLayers />
      <NodeLayer routeIds={routeIds} />

      {pts.length > 1 && (
        <>
          <path
            className="map-route-halo draw"
            style={{ "--len": totalLen } as React.CSSProperties}
            d={line(pts)}
          />
          <path
            className="map-route draw"
            style={{ "--len": totalLen } as React.CSSProperties}
            d={line(pts)}
          />
        </>
      )}

      {pts.map((p, i) => {
        const last = i === pts.length - 1;
        return (
          <g key={i}>
            <circle
              className={`map-pin${last ? " last" : ""}`}
              cx={p[0]}
              cy={p[1]}
              r="11.5"
            />
            <text className="map-pin-text" x={p[0]} y={p[1] + 4}>
              {i + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function TrafficMap() {
  return (
    <svg
      viewBox="0 0 760 620"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Map showing how freely traffic is moving on each road"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="press">
          <stop offset="0%" stopColor="#FF073A" stopOpacity="0.26" />
          <stop offset="100%" stopColor="#FF073A" stopOpacity="0" />
        </radialGradient>
      </defs>
      <BaseMapLayers />
      {PRESSURE.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r={p[2]} fill="url(#press)" />
      ))}
      {SEGMENTS.map((g, i) => (
        <path key={i} className={`map-seg ${g.s}`} d={line(g.p)} />
      ))}
      <NodeLayer routeIds={[]} />
    </svg>
  );
}
