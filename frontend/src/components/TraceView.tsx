import { useEffect, useState } from "react";
import type { Journey } from "../data/veritrackData";
import { JOURNEYS, NODES } from "../data/veritrackData";
import { TraceMap } from "./SchematicMaps";

interface TraceViewProps {
  initialPlate?: string;
}

export function TraceView({ initialPlate = "HR 26 DK 8337" }: TraceViewProps) {
  const [searchInput, setSearchInput] = useState(initialPlate);
  const [currentKey, setCurrentKey] = useState<string>("HR26DK8337");

  useEffect(() => {
    if (initialPlate) {
      setSearchInput(initialPlate);
      const cleanKey = initialPlate.replace(/[^A-Za-z0-9]/g, "").toUpperCase();
      setCurrentKey(cleanKey);
    }
  }, [initialPlate]);

  const handleSearch = () => {
    const cleanKey = (searchInput || "").replace(/[^A-Za-z0-9]/g, "").toUpperCase();
    setCurrentKey(cleanKey);
  };

  const handleClear = () => {
    setSearchInput("");
    setCurrentKey("");
  };

  const journey: Journey | undefined = JOURNEYS[currentKey];

  return (
    <section className="view" id="view-trace">
      <div className="head">
        <h1 className="serif">Trace a vehicle</h1>
        <p>
          Every camera point the vehicle passed, in order, with the time and the direction it was travelling.
        </p>
      </div>

      <div className="body">
        <div className="trace-controls">
          <div className="search">
            <div className="field">
              <label htmlFor="plate">Registration</label>
              <input
                id="plate"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="e.g. HR 26 DK 8337"
                autoComplete="off"
                spellCheck={false}
              />
            </div>
            <button className="btn" type="button" onClick={handleSearch}>
              Show journey
            </button>
            <button className="btn ghost" type="button" onClick={handleClear}>
              Clear
            </button>
          </div>
          <p className="hint">
            Searches are entered in the access register against your name.
          </p>
        </div>

        {/* Map panel */}
        <div className="panel map-panel">
          <div className="panel-head">
            <h2>
              {journey ? `Journey of ${journey.plate}` : "Journey across the city"}
            </h2>
            <span className="note">
              {journey ? journey.window : "Last 30 days searched"}
            </span>
          </div>
          <div className="map-hold">
            <TraceMap stops={journey?.stops || []} />
          </div>
          <div className="bar">
            {journey ? (
              journey.bar.map((b, i) => (
                <span key={i} dangerouslySetInnerHTML={{ __html: b }} />
              ))
            ) : (
              <span>No journey to show</span>
            )}
          </div>
        </div>

        {/* Vehicle and stops side panel */}
        <div className="panel side">
          <div className="panel-head">
            <h2>Vehicle and stops</h2>
            <span className="note">
              {journey ? `${journey.stops.length} stops` : "No stops"}
            </span>
          </div>

          {journey ? (
            <>
              <div className="vhead">
                <span className="plate">{journey.plate}</span>
                <p className="vmeta">
                  {journey.vehicle} &nbsp;&middot;&nbsp; <em>{journey.where}</em>
                </p>
                {/* Red alert card: Solid red with white text */}
                {journey.flag && <p className="flagline">{journey.flag}</p>}
              </div>

              <div className="scroll">
                {journey.stops.map((s, i) => (
                  <div className="stop" key={i}>
                    <span className="stop-n">{i + 1}</span>
                    <span className="stop-where">
                      {NODES[s.at]?.n || s.at}
                    </span>
                    <span className="stop-time">{s.t}</span>
                    <span className="stop-what">{s.what}</span>
                    <span className={`stop-speed${s.slow ? " slow" : ""}`}>
                      Passing at <b>{s.sp}</b>
                    </span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty">
              <h3 className="serif">No sightings for that number</h3>
              <p>
                Nothing was read at any camera point in the last 30 days. Check
                the number, or try HR 26 DK 8337, UP 16 BT 2290, or DL 8C AF 4521.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
