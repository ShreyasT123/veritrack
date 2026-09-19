import { Camera, Maximize2, ShieldCheck } from "lucide-react";

const detections = [
  { plate: "GX15 OGJ", confidence: 92.6, vehicle_class: "White Sedan / Saloon", lane: "LANE 01", timestamp: "10:14:02 IST" },
  { plate: "AP05 JEO", confidence: 88.8, vehicle_class: "Silver Sedan / Saloon", lane: "LANE 02", timestamp: "10:14:02 IST" },
  { plate: "KH05 ZZK", confidence: 84.6, vehicle_class: "Blue Hatchback", lane: "LANE 03", timestamp: "10:14:02 IST" },
];

export function VideoFeedView() {
  return (
    <div className="feed-layout">
      {/* Top Video Stage Panel */}
      <section className="feed-panel">
        <div className="panel-title">
          <div className="title-with-icon">
            <Camera size={14} />
            <span>OVERHEAD GANTRY SURVEILLANCE FEED · NH-48 EXPRESSWAY</span>
          </div>
          <div className="title-meta">
            <span className="live-dot" />
            <span>RTSP-CH01 · 1080p@30FPS · GANTRY 5.8m ELEVATION</span>
            <button className="panel-icon-btn" title="Fullscreen">
              <Maximize2 size={12} />
            </button>
          </div>
        </div>

        <div className="feed-video-container">
          <video
            src="/annotated.mp4"
            controls
            autoPlay
            muted
            loop
            playsInline
            poster="/cascade_showcase.png"
          />
        </div>
      </section>

      {/* Extracted Cascade Detections Panel */}
      <section className="feed-panel">
        <div className="panel-title">
          <span>CASCADE INFERENCE TELEMETRY · RECTIFIED OCR DETECTIONS</span>
          <span>MULTI-LANE FREE-FLOW (MLFF) BENCHMARK</span>
        </div>

        <div className="feed-detection-row">
          {detections.map((item) => (
            <article className="sighting feed-card" key={item.plate}>
              <div className="feed-card-left">
                <div className="plate-line">
                  <b className="mono">{item.plate}</b>
                  <span className="lane-pill">{item.lane}</span>
                </div>
                <em>{item.vehicle_class}</em>
                <p>GMDA-CAM-01 · Shankar Chowk Gantry</p>
                <time className="mono">{item.timestamp}</time>
              </div>

              <div className="confidence">
                <b>{item.confidence.toFixed(1)}%</b>
                <i>
                  <span style={{ width: `${item.confidence}%` }} />
                </i>
                <small className="mono">CASCADE PASS</small>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* Advisory Footer Matching Tab 1 */}
      <footer className="command-disclaimer">
        <span>
          <ShieldCheck size={14} />
          <b>STATUTORY COMPLIANCE:</b> Multi-lane video ingestion executes under DPDP Act 2023 warrant scope. Plate text validated via Indian positional regex grammar.
        </span>
        <span className="mono">LATENCY: 14.2 ms · FP16 TensorRT</span>
      </footer>
    </div>
  );
}