import { BUSY, OD, ROADSTATE } from "../data/veritrackData";
import { TrafficMap } from "./SchematicMaps";

export function TrafficView() {
  return (
    <section className="view" id="view-traffic">
      <div className="head">
        <h1 className="serif">How the city is moving</h1>
        <p>
          Speeds come from vehicles seen at two or more camera points in the last
          twenty minutes, so they describe real journeys rather than a single spot
          reading.
        </p>
      </div>

      <div className="body">
        {/* Left Column: Map & OD Flows */}
        <div className="traffic-left">
          <div className="panel">
            <div className="panel-head">
              <h2>Main roads right now</h2>
              <span className="note">Updated a moment ago</span>
            </div>
            <div className="map-hold">
              <TrafficMap />
            </div>
            <div className="legend">
              <span>
                <i style={{ background: "var(--green)" }} /> Moving freely
              </span>
              <span>
                <i style={{ background: "var(--amber)" }} /> Slowing down
              </span>
              <span>
                <i style={{ background: "var(--red)" }} /> Heavy
              </span>
              <span>Shaded circles mark where vehicles are backing up</span>
            </div>
          </div>

          <div className="panel" style={{ flex: "none" }}>
            <div className="panel-head">
              <h2>Where the traffic is going</h2>
              <span className="note">Most common journeys this morning</span>
            </div>
            <div className="od-scroll">
              {OD.map((o, i) => (
                <div className="od" key={i}>
                  <div className="rt">
                    {o.a}
                    <em>to</em>
                    {o.b}
                  </div>
                  <div className="pc">{o.pc}</div>
                  <div className="sub">{o.sub}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Road by road & Busiest points */}
        <div className="traffic-right">
          <div className="panel">
            <div className="panel-head">
              <h2>Road by road</h2>
              <span className="note">Average speed</span>
            </div>
            <div className="scroll">
              {ROADSTATE.map((r, i) => {
                const word =
                  r.s === "free"
                    ? "Moving freely"
                    : r.s === "slow"
                    ? "Slowing down"
                    : "Heavy";
                const widthPercent = Math.min(
                  100,
                  Math.round((r.now / r.usual) * 100)
                );
                return (
                  <div className="road-row" key={i}>
                    <div className="top">
                      <span className="nm">{r.n}</span>
                      <span className={`st ${r.s}`}>{word}</span>
                    </div>
                    <div className="track">
                      <i className={r.s} style={{ width: `${widthPercent}%` }} />
                    </div>
                    <div className="sp">
                      <b>{r.now} km/h</b> now, usually {r.usual} km/h at this hour
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Busiest points</h2>
              <span className="note">Vehicles an hour</span>
            </div>
            <div className="scroll">
              {BUSY.map((b, i) => (
                <div className="busy" key={i}>
                  <span className="rank">{i + 1}</span>
                  <span className="nm">
                    {b.n}
                    <small>{b.rd}</small>
                  </span>
                  <span className="ct">
                    {b.c}
                    <em>{b.d}</em>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
