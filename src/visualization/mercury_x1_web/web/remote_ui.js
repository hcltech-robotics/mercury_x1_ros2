// Unitree BLE remote indicators shared by the control page (:8080) and the head display (:8081):
// battery icon, 1-5 bar signal icon from RSSI, and the drive lock state (Start unlocks / locks).
(function () {
  const C = { ok: "#3ddc84", warn: "#ffb020", bad: "#ff5468", on: "#eef0f3", off: "#3a3f4b", line: "#cfd3da" };

  // h = icon height in px; the battery is drawn 2:1
  function battery(pct, h = 14) {
    const p = pct == null ? null : Math.max(0, Math.min(100, pct));
    const fill = p == null ? C.off : p > 30 ? C.ok : p > 15 ? C.warn : C.bad;
    const w = p == null ? 0 : (p / 100) * 16;
    return `<svg class="rm-ic" width="${h * 2}" height="${h}" viewBox="0 0 24 12" aria-label="battery ${p ?? "?"} %">` +
      `<rect x="0.75" y="0.75" width="19.5" height="10.5" rx="2" fill="none" stroke="${C.line}" stroke-width="1.5"/>` +
      `<rect x="21" y="3.5" width="2.2" height="5" rx="0.8" fill="${C.line}"/>` +
      `<rect x="2.5" y="2.5" width="${w.toFixed(1)}" height="7" rx="1" fill="${fill}"/></svg>`;
  }

  // BLE RSSI -> bars: >= -55 dBm 5, -65 4, -72 3, -80 2, weaker 1; not connected 0
  function bars(rssi) {
    if (rssi == null) return 0;
    return rssi >= -55 ? 5 : rssi >= -65 ? 4 : rssi >= -72 ? 3 : rssi >= -80 ? 2 : 1;
  }
  function signal(rssi, h = 14) {
    const n = bars(rssi), col = n >= 3 ? C.on : n === 2 ? C.warn : C.bad;
    const r = [0, 1, 2, 3, 4].map((i) =>
      `<rect x="${i * 4.4}" y="${11 - (i + 1) * 2.2}" width="3.2" height="${(i + 1) * 2.2}" rx="0.6" fill="${i < n ? col : C.off}"/>`).join("");
    return `<svg class="rm-ic" width="${h * 1.6}" height="${h}" viewBox="0 0 21.2 11.2" aria-label="signal ${n}/5">${r}</svg>`;
  }

  const LOCK = '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>';
  const OPEN = '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 7.5-2"/>';
  function lockIcon(open, color, h = 14) {
    return `<svg class="rm-ic" width="${h}" height="${h}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="2.2" ` +
      `stroke-linecap="round" stroke-linejoin="round">${open ? OPEN : LOCK}</svg>`;
  }

  // drive lock: {cls: ok|warn|muted, short, long, icon(h)}; null when not connected / unknown
  function lock(r) {
    if (!r || r.state !== "connected") return null;
    if (r.driving) return { cls: "warn", short: "DRIVING", long: "DRIVING · sticks move the base", open: true, color: C.warn };
    if (r.drive_enabled === true) return { cls: "ok", short: "unlocked", long: "Control allowed · Start locks", open: true, color: C.ok };
    if (r.drive_enabled === false) return { cls: "muted", short: "locked", long: "Locked · press Start to drive", open: false, color: "#959cab" };
    return null;
  }

  window.RemoteUI = { battery, signal, bars, lock, lockIcon };
})();
