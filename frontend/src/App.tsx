import { useState } from "react";
import { AlertsView } from "./components/AlertsView";
import { CamerasView } from "./components/CamerasView";
import { RecordsView } from "./components/RecordsView";
import { Sidebar } from "./components/Sidebar";
import { TopStrip } from "./components/TopStrip";
import { TraceView } from "./components/TraceView";
import { TrafficView } from "./components/TrafficView";

export default function App() {
  const [tab, setTab] = useState<string>("trace");
  const [tracePlate, setTracePlate] = useState<string>("HR 26 DK 8337");

  const handleOpenJourney = (plate: string) => {
    setTracePlate(plate);
    setTab("trace");
  };

  return (
    <div className="shell">
      {/* 1. Sidebar with flipped colors: Solid blue background (#0464FF) and white text */}
      <Sidebar tab={tab} onTabChange={setTab} alertCount={2} />

      {/* 2. Main operational stage */}
      <div className="stage">
        <TopStrip />

        {tab === "trace" && <TraceView initialPlate={tracePlate} />}
        {tab === "records" && <RecordsView />}
        {tab === "cameras" && <CamerasView />}
        {tab === "traffic" && <TrafficView />}
        {tab === "alerts" && <AlertsView onOpenJourney={handleOpenJourney} />}
      </div>
    </div>
  );
}