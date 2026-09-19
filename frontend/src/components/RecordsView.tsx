import { useState } from "react";
import type { SightingRecord } from "../data/veritrackData";
import { QWORD, RECORDS } from "../data/veritrackData";

export function RecordsView() {
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<"all" | "flag" | "check">("all");

  const q = searchQuery.replace(/\s+/g, " ").trim().toLowerCase();
  const rows: SightingRecord[] = RECORDS.filter((r) => {
    if (filterType === "flag" && !r[7]) return false;
    if (filterType === "check" && r[6] === "good") return false;
    if (!q) return true;
    return `${r[3]} ${r[1]} ${r[2]} ${r[4]}`.toLowerCase().includes(q);
  });

  const handleExport = () => {
    const header = [
      "Time",
      "Camera Point",
      "Road",
      "Registration",
      "Vehicle",
      "Travelling",
      "Reading Quality",
      "Watchlist Flag",
    ];
    const csvContent = [
      header.join(","),
      ...rows.map((row) =>
        [
          JSON.stringify(row[0]),
          JSON.stringify(row[1]),
          JSON.stringify(row[2]),
          JSON.stringify(row[3]),
          JSON.stringify(row[4]),
          JSON.stringify(row[5]),
          JSON.stringify(QWORD[row[6]]),
          JSON.stringify(row[7] ? "YES" : "NO"),
        ].join(",")
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `veritrack_sightings_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section className="view" id="view-records">
      <div className="head">
        <h1 className="serif">Sighting records</h1>
        <p>
          Every reading from every camera point, newest first. Search by
          registration number or by the name of a place.
        </p>
      </div>

      <div className="body">
        {/* Filter bar */}
        <div className="filters">
          <div className="field" style={{ flex: "0 1 280px" }}>
            <label htmlFor="rec-search">Search</label>
            <input
              id="rec-search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Registration or place"
              autoComplete="off"
              style={{ fontWeight: 500, letterSpacing: "0.03em" }}
            />
          </div>

          <button
            type="button"
            className="chip"
            data-filter="all"
            aria-pressed={filterType === "all"}
            onClick={() => setFilterType("all")}
          >
            Everything
          </button>
          <button
            type="button"
            className="chip"
            data-filter="flag"
            aria-pressed={filterType === "flag"}
            onClick={() => setFilterType("flag")}
          >
            On a watchlist
          </button>
          <button
            type="button"
            className="chip"
            data-filter="check"
            aria-pressed={filterType === "check"}
            onClick={() => setFilterType("check")}
          >
            Needs a check
          </button>

          <button
            type="button"
            className="btn ghost"
            style={{ marginLeft: "auto" }}
            onClick={handleExport}
          >
            Export for the case file
          </button>
        </div>

        {/* Table panel */}
        <div className="panel">
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Camera point</th>
                  <th>Road</th>
                  <th>Registration</th>
                  <th>Vehicle</th>
                  <th>Travelling</th>
                  <th>Reading</th>
                </tr>
              </thead>
              <tbody>
                {rows.length > 0 ? (
                  rows.map((r, i) => (
                    <tr key={i}>
                      <td className="t">{r[0]}</td>
                      <td>{r[1]}</td>
                      <td className="t">{r[2]}</td>
                      <td>
                        <span className={`pl${r[7] ? " hot" : ""}`}>{r[3]}</span>
                        {r[7] && <small>On a watchlist</small>}
                      </td>
                      <td className="t">{r[4]}</td>
                      <td className="t">{r[5]}</td>
                      <td>
                        <span className={`rdq ${r[6]}`}>{QWORD[r[6]]}</span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7}>
                      <div className="empty">
                        <h3 className="serif">Nothing matches that</h3>
                        <p>
                          Try a shorter search, such as the last four digits or
                          the name of a junction.
                        </p>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="bar">
            <span>
              {rows.length === RECORDS.length ? (
                <>
                  <b>{rows.length}</b> readings shown
                </>
              ) : (
                <>
                  <b>{rows.length}</b> of {RECORDS.length} readings shown
                </>
              )}
            </span>
            <span>Readings are removed automatically after 30 days</span>
            <span>Open only to officers assigned to the case</span>
          </div>
        </div>
      </div>
    </section>
  );
}
