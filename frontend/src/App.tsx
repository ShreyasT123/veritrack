import { useEffect, useState } from "react";
import { AlertPanel } from "./components/AlertPanel";
import { CorridorSummary } from "./components/CorridorSummary";
import { MapView } from "./components/MapView";
import { SightingTicker } from "./components/SightingTicker";
import { TopNav } from "./components/TopNav";
import { VehicleSearch } from "./components/VehicleSearch";
import { mockAlerts, mockCorridors, mockSightings } from "./data/mockData";
import { api, gatewayConnected } from "./services/api";
import type { AlertRecord, CorridorMetric, Sighting, Trajectory } from "./types/telemetry";

export default function App() {
  const [sightings, setSightings] = useState<Sighting[]>(mockSightings);
  const [corridors, setCorridors] = useState<CorridorMetric[]>(mockCorridors);
  const [alerts, setAlerts] = useState<AlertRecord[]>(mockAlerts);
  const [trajectory, setTrajectory] = useState<Trajectory>();
  const [connected, setConnected] = useState(false);
  const [now, setNow] = useState(new Date());
  const refresh = async () => { const [s, c, a] = await Promise.all([api.sightings(), api.corridors(), api.alerts()]); setSightings(s); setCorridors(c); setAlerts(a); setConnected(gatewayConnected()); };
  useEffect(() => { void refresh(); const poll = window.setInterval(() => void refresh(), 3000); const clock = window.setInterval(() => setNow(new Date()), 1000); return () => { clearInterval(poll); clearInterval(clock); }; }, []);
  const search = async (plate: string) => { setTrajectory(await api.trajectory(plate)); setConnected(gatewayConnected()); };
  return <div className="app"><TopNav connected={connected} now={now}/><main><aside className="left"><VehicleSearch onSearch={search}/><SightingTicker sightings={sightings}/></aside><section className="center"><MapView corridors={corridors} trajectory={trajectory}/><div className="map-footer"><b>Command advisory:</b> Search is warrant-scoped. CCTV observations are DPDP pseudonymised at ingestion; audit ledger write enabled.</div></section><aside className="right"><AlertPanel alert={alerts[0]}/><CorridorSummary corridors={corridors}/></aside></main></div>;
}
