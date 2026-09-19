import { Database, Download, Search, ShieldCheck } from "lucide-react";
import { useState } from "react";
import type { DatabaseRecord } from "../types/telemetry";

export function DatabaseExplorer({ records }: { records: DatabaseRecord[] }) {
  const [query, setQuery] = useState("GX15 OGJ");
  const [filter, setFilter] = useState("GX15 OGJ");

  const visible = records.filter((record) =>
    `${record.plate} ${record.camera_id} ${record.node_id}`
      .toLowerCase()
      .includes(filter.toLowerCase())
  );

  return (
    <div className="db-container">
      {/* 1. Header */}
      <div className="db-header">
        <div>
          <span className="db-badge">
            <Database size={12} /> PERSISTENCE & STATUTORY AUDIT
          </span>
          <h1 className="db-title">Raw Sighting Telemetry & DPDP Ledger</h1>
          <p className="db-subtitle">
            Warrant-scoped query boundary. Non-hotlist records pseudonymised with rotating HMAC-SHA256 salts.
          </p>
        </div>
        <button className="db-export-btn">
          <Download size={14} /> Export Audit Log
        </button>
      </div>

      {/* 2. Command Console Query Bar */}
      <div className="db-query-bar">
        <div className="db-query-prefix">
          <Search size={14} />
          <span>SELECT * FROM sightings WHERE</span>
        </div>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && setFilter(query)}
          placeholder="Filter by plate or camera..."
        />
        <button onClick={() => setFilter(query)}>Filter Records</button>
        <span className="db-query-count">{visible.length} sightings returned</span>
      </div>

      {/* 3. Structured Data Table */}
      <div className="db-table-wrapper">
        <table>
          <thead>
            <tr>
              <th>TIMESTAMP (IST)</th>
              <th>SURVEILLANCE NODE</th>
              <th>RESOLVED PLATE</th>
              <th>CONFIDENCE</th>
              <th>DPDP ROTATING HMAC (SHA-256)</th>
              <th>SPEED & HEADING</th>
              <th>CLASSIFICATION</th>
              <th>RETENTION</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((record) => (
              <tr key={`${record.timestamp}-${record.camera_id}`}>
                <td className="db-mono-cell">{record.timestamp}</td>
                <td>
                  <b>{record.node_id}</b>
                  <small>{record.camera_id}</small>
                </td>
                <td>
                  <span className="db-plate-chip">{record.plate}</span>
                </td>
                <td>
                  <span className={`db-conf-chip ${record.confidence > 90 ? "high" : "med"}`}>
                    {record.confidence.toFixed(1)}%
                  </span>
                </td>
                <td className="db-hash-cell">{record.dpdp_hash}</td>
                <td className="db-mono-cell">{record.source}</td>
                <td>{record.vehicle_class}</td>
                <td>
                  <span className="db-retention-tag">30-Day Auto-Purge</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 4. Footer */}
      <div className="db-footer">
        <span>
          <ShieldCheck size={14} /> DPDP ACT 2023 SECTION 8 COMPLIANT · IDENTITY ENCRYPTED AT REST
        </span>
        <b>STORAGE ENGINE: TIMESCALEDB HYPERTABLE · POSTGIS GIST INDEXED</b>
      </div>

      {/* Self-Contained Scoped Styling */}
      <style>{`
        .db-container {
          max-width: 1400px;
          margin: 0 auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 14px;
          font-family: 'Inter', -apple-system, sans-serif;
          color: #0f172a;
          height: calc(100vh - 40px);
        }

        .db-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          padding-bottom: 12px;
          border-bottom: 1px solid #e2e8f0;
        }

        .db-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.08em;
          color: #007ba7;
          text-transform: uppercase;
          margin-bottom: 4px;
        }

        .db-title {
          font-size: 20px;
          font-weight: 700;
          color: #0f172a;
          letter-spacing: -0.01em;
        }

        .db-subtitle {
          font-size: 12px;
          color: #64748b;
          margin-top: 2px;
        }

        .db-export-btn {
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 4px;
          padding: 7px 12px;
          font-size: 12px;
          font-weight: 600;
          color: #0f172a;
          display: inline-flex;
          align-items: center;
          gap: 6px;
          cursor: pointer;
        }

        /* COMMAND QUERY BAR */
        .db-query-bar {
          display: flex;
          align-items: center;
          gap: 10px;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          padding: 8px 12px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }

        .db-query-prefix {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          color: #007ba7;
          white-space: nowrap;
        }

        .db-query-bar input {
          flex: 1;
          border: 1px solid #e2e8f0;
          background: #f8fafc;
          border-radius: 4px;
          padding: 6px 10px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 13px;
          font-weight: 600;
          color: #0f172a;
          outline: none;
          text-transform: uppercase;
        }

        .db-query-bar button {
          background: #007ba7;
          color: #ffffff;
          border: none;
          border-radius: 4px;
          padding: 7px 14px;
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
        }

        .db-query-count {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          color: #64748b;
          white-space: nowrap;
          background: #f1f5f9;
          padding: 4px 8px;
          border-radius: 4px;
        }

        /* TABLE WRAPPER */
        .db-table-wrapper {
          flex: 1;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 6px;
          overflow: auto;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }

        .db-table-wrapper table {
          width: 100%;
          border-collapse: collapse;
          text-align: left;
        }

        .db-table-wrapper thead th {
          background: #f8fafc;
          color: #64748b;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          padding: 10px 14px;
          border-bottom: 1px solid #e2e8f0;
          white-space: nowrap;
          position: sticky;
          top: 0;
          z-index: 5;
        }

        .db-table-wrapper tbody td {
          padding: 10px 14px;
          font-size: 11.5px;
          border-bottom: 1px solid #f1f5f9;
          color: #0f172a;
          vertical-align: middle;
        }

        .db-table-wrapper tbody tr:hover {
          background-color: #f8fafc;
        }

        .db-mono-cell {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
        }

        .db-table-wrapper td b {
          font-weight: 600;
          display: block;
        }

        .db-table-wrapper td small {
          font-size: 10px;
          color: #64748b;
          font-family: 'JetBrains Mono', monospace;
        }

        .db-plate-chip {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 700;
          color: #0f172a;
          background: #fef08a;
          padding: 3px 8px;
          border-radius: 3px;
          border: 1px solid #fde047;
          display: inline-block;
          font-size: 12px;
        }

        .db-conf-chip {
          padding: 2px 6px;
          border-radius: 3px;
          font-weight: 700;
          font-family: 'JetBrains Mono', monospace;
          font-size: 10.5px;
        }

        .db-conf-chip.high {
          color: #16a34a;
          background: #f0fdf4;
        }

        .db-conf-chip.med {
          color: #d97706;
          background: #fffbeb;
        }

        .db-hash-cell {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          color: #64748b;
        }

        .db-retention-tag {
          font-size: 10px;
          font-family: 'JetBrains Mono', monospace;
          color: #16a34a;
          font-weight: 600;
        }

        /* FOOTER */
        .db-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: #ffffff;
          border: 1px solid #e2e8f0;
          padding: 8px 14px;
          border-radius: 4px;
          font-size: 10.5px;
          color: #64748b;
        }

        .db-footer span {
          display: inline-flex;
          align-items: center;
          gap: 6px;
        }

        .db-footer b {
          font-family: 'JetBrains Mono', monospace;
          color: #007ba7;
        }
      `}</style>
    </div>
  );
}