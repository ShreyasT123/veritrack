interface SidebarProps {
  tab: string;
  onTabChange: (tab: string) => void;
  alertCount?: number;
}

export function Sidebar({ tab, onTabChange, alertCount = 2 }: SidebarProps) {
  return (
    <aside className="rail">
      <div className="rail-top">
        <div className="mark">
          Veri<span>track</span>
        </div>
        <p className="mark-sub">
          Traffic Command Centre
          <br />
          Gurugram
        </p>
      </div>

      <nav id="nav" aria-label="Sections">
        <p className="nav-label">Find a vehicle</p>
        <button
          type="button"
          data-view="trace"
          aria-current={tab === "trace"}
          onClick={() => onTabChange("trace")}
        >
          <span>Trace a vehicle</span>
        </button>
        <button
          type="button"
          data-view="records"
          aria-current={tab === "records"}
          onClick={() => onTabChange("records")}
        >
          <span>Sighting records</span>
        </button>

        <p className="nav-label">Watch the city</p>
        <button
          type="button"
          data-view="cameras"
          aria-current={tab === "cameras"}
          onClick={() => onTabChange("cameras")}
        >
          <span>Camera points</span>
        </button>
        <button
          type="button"
          data-view="traffic"
          aria-current={tab === "traffic"}
          onClick={() => onTabChange("traffic")}
        >
          <span>City traffic</span>
        </button>
        <button
          type="button"
          data-view="alerts"
          aria-current={tab === "alerts"}
          onClick={() => onTabChange("alerts")}
        >
          <span>Alerts</span>
          {alertCount > 0 && <span className="pip">{alertCount}</span>}
        </button>
      </nav>

      <div className="rail-foot">
        <p className="who">
          <small>Sector 29 police station</small>
        </p>
      </div>
    </aside>
  );
}