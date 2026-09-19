import { useState } from "react";
import type { CameraPoint } from "../data/veritrackData";
import { CAMERAS, QWORD, READS } from "../data/veritrackData";

export function CamerasView() {
  const [selectedId, setSelectedId] = useState<string>("shankar");

  const cam: CameraPoint =
    CAMERAS.find((c) => c.id === selectedId) ?? CAMERAS[0]!;
  const reads = READS[selectedId] || [];

  return (
    <section className="view" id="view-cameras">
      <div className="head">
        <h1 className="serif">Camera points</h1>
        <p>
          Choose a camera point to see what it is reading now. Readings that need
          a second look are marked before they reach the register.
        </p>
      </div>

      <div className="body">
        {/* Left column: Camera points list */}
        <div className="panel">
          <div className="panel-head">
            <h2>Across the city</h2>
            <span className="note">{CAMERAS.length} points</span>
          </div>
          <div className="scroll">
            {CAMERAS.map((c) => {
              const isActive = c.id === selectedId;
              return (
                <button
                  type="button"
                  key={c.id}
                  className="cam-row"
                  aria-current={isActive}
                  onClick={() => setSelectedId(c.id)}
                >
                  <span>
                    <span className="nm">{c.n}</span>
                    <span className="rd">{c.rd}</span>
                  </span>
                  <span className="cnt">{c.hr}/hr</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Center column: Live surveillance video stream without media player controls */}
        <div className="panel">
          <div className="panel-head">
            <h2>{cam.n}</h2>
            <span className="note">{cam.dir}</span>
          </div>
          <div className="viewer">
            {/* Live Surveillance Feed Overlay */}
            <div className="cam-live-badge">
              <span className="pulse-dot" />
              <span>LIVE RTSP · 1080p@30FPS · GANTRY 5.8m</span>
            </div>
            <div className="cam-telemetry-overlay">
              NODE: GMDA-CAM-{cam.id.toUpperCase()} · 14.2ms FP16
            </div>

            {/* Video element: playback controls disabled, looks strictly like live feed */}
            <video
              src="/annotated.mp4"
              autoPlay
              muted
              loop
              playsInline
              disablePictureInPicture
              controlsList="nodownload nofullscreen noremoteplayback"
              aria-label={`Live surveillance camera feed for ${cam.n}`}
            />
          </div>
          <div className="bar">
            <span>
              Reading <b>{cam.lanes} lanes</b>
            </span>
            <span>
              <b>{cam.hr}</b> vehicles an hour
            </span>
            <span>{cam.rd}</span>
            <span>
              Recording since <b>6:00 am</b>
            </span>
          </div>
        </div>

        {/* Right column: Reads panel */}
        <div className="panel reads-panel">
          <div className="panel-head">
            <h2>Just read here</h2>
            <span className="note">{reads.length} in the last minute</span>
          </div>
          <div className="scroll">
            {reads.map((r, i) => (
              <div className="read" key={i}>
                <span className={`p${r.hot ? " hot" : ""}`}>{r.plate}</span>
                <span className={`q ${r.q}`}>
                  <span>{QWORD[r.q]}</span>
                  <span className={`bars ${r.q}`}>
                    <i />
                    <i />
                    <i />
                  </span>
                </span>
                <span className="d">{r.v}</span>
                <span className="m">
                  {r.lane}, read at {r.t}
                  {r.hot ? ", on an alert" : ""}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
