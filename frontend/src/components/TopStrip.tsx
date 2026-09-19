import { useEffect, useState } from "react";

export function TopStrip() {
  const [timeStr, setTimeStr] = useState("");
  const [dateStr, setDateStr] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        new Intl.DateTimeFormat("en-IN", {
          timeZone: "Asia/Kolkata",
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        }).format(now)
      );
      setDateStr(
        new Intl.DateTimeFormat("en-IN", {
          timeZone: "Asia/Kolkata",
          weekday: "short",
          day: "numeric",
          month: "short",
        }).format(now)
      );
    };

    updateTime();
    const interval = setInterval(updateTime, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="strip">
      <span className="cell">
        <i className="led" /> <b>14</b> camera points working
      </span>
      <span className="cell">
        <b>18,420</b> readings since midnight
      </span>
      <span className="cell flag">
        <i className="led red" /> <b>2</b> alerts need action
      </span>
      <span className="cell">
        City average <b>34 km/h</b>
      </span>
      <span className="cell spacer" />
      <span className="cell time">
        <b>{timeStr || "10:14"}</b>
        <span style={{ color: "var(--muted)", marginLeft: "6px" }}>
          {dateStr || "Sat 19 Sep"}
        </span>
      </span>
    </div>
  );
}
