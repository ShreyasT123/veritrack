import { useState } from "react";
import type { AlertItem } from "../data/veritrackData";
import { ALERTS } from "../data/veritrackData";

interface AlertsViewProps {
  onOpenJourney: (plate: string) => void;
}

export function AlertsView({ onOpenJourney }: AlertsViewProps) {
  const [selectedId, setSelectedId] = useState<string>("clone");
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());
  const [dispatchedIds, setDispatchedIds] = useState<Set<string>>(new Set());

  const currentAlert: AlertItem =
    ALERTS.find((a) => a.id === selectedId) ?? ALERTS[0]!;

  const handleMarkChecked = (id: string) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleDispatch = (id: string) => {
    setDispatchedIds((prev) => new Set(prev).add(id));
  };

  const isChecked = checkedIds.has(currentAlert.id);
  const isDispatched = dispatchedIds.has(currentAlert.id);

  return (
    <section className="view" id="view-alerts">
      <div className="head">
        <h1 className="serif">Alerts</h1>
        <p>
          Vehicles on a watchlist, and journeys the cameras recorded that do not
          add up.
        </p>
      </div>

      <div className="body">
        {/* Left Column: Waiting list */}
        <div className="panel">
          <div className="panel-head">
            <h2>Waiting for you</h2>
            <span className="note">{ALERTS.length - checkedIds.size} open</span>
          </div>
          <div className="scroll">
            {ALERTS.map((a) => {
              const isSelected = a.id === selectedId;
              const checked = checkedIds.has(a.id);
              return (
                <button
                  type="button"
                  key={a.id}
                  className="alert-row"
                  aria-current={isSelected}
                  onClick={() => setSelectedId(a.id)}
                  style={checked ? { opacity: 0.6 } : undefined}
                >
                  <span className={`sev ${a.sev}`}>
                    {checked ? "Resolved" : a.sevw}
                  </span>
                  <div className="ttl">{a.ttl}</div>
                  <div className="sub">{a.sub}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Alert detail */}
        <div className="panel">
          <div className="panel-head">
            <h2>{currentAlert.ttl}</h2>
            <span className="note">{currentAlert.when}</span>
          </div>
          <div className="scroll">
            <div className="detail">
              <span className={`sev ${currentAlert.sev}`}>
                {isChecked ? "Resolved" : currentAlert.sevw}
              </span>
              <h2>{currentAlert.ttl}</h2>
              <p
                className="txt"
                dangerouslySetInnerHTML={{ __html: currentAlert.body }}
              />

              <dl className="facts">
                {currentAlert.facts.map((f, i) => (
                  <div className="fact" key={i}>
                    <dt>{f[0]}</dt>
                    <dd>{f[1]}</dd>
                  </div>
                ))}
              </dl>

              <div className="actions">
                <button
                  type="button"
                  className={`btn${
                    currentAlert.sev === "urgent" ? " danger" : ""
                  }`}
                  onClick={() => handleDispatch(currentAlert.id)}
                  disabled={isDispatched}
                >
                  {isDispatched
                    ? "✓ Transmitted to ERSS-112 CAD"
                    : "Send to the control room"}
                </button>

                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => onOpenJourney(currentAlert.plate)}
                >
                  Open the journey
                </button>

                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => handleMarkChecked(currentAlert.id)}
                >
                  {isChecked ? "Reopen incident" : "Mark as checked"}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
