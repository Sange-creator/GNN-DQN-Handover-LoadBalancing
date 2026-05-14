// =========================================================================
// primitives.jsx — small shared UI atoms used by sections.
// =========================================================================

const { useState, useEffect, useRef, useMemo, useCallback } = React;

function Eyebrow({ children, num }) {
  return (
    <div className="section-eyebrow">
      {num ? <span style={{ fontFeatureSettings: '"tnum"' }}>{num}</span> : null}
      <span>{children}</span>
    </div>
  );
}

function MethodSwatch({ id, size = 10 }) {
  const m = window.METHOD_BY_ID[id];
  if (!m) return null;
  return (
    <span
      className="swatch"
      aria-hidden="true"
      style={{
        display: "inline-block",
        width: size,
        height: size,
        borderRadius: 2,
        background: m.dotted ? "transparent" : m.raw,
        border: m.dotted ? `1.5px dotted ${m.raw}` : "none",
      }}
    />
  );
}

function MethodChip({ id }) {
  const m = window.METHOD_BY_ID[id];
  if (!m) return null;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontFamily: "var(--mono)", fontSize: 11.5 }}>
      <MethodSwatch id={id} />
      <span style={{ color: m.ours ? "var(--signal)" : "var(--ink-2)", fontWeight: m.ours ? 600 : 400 }}>{m.label}</span>
    </span>
  );
}

function NumStat({ k, v, u }) {
  return (
    <div className="scn-stat">
      <span className="scn-stat-k">{k}</span>
      <span className="scn-stat-v">
        {v}
        {u ? <span style={{ color: "var(--ink-3)", fontWeight: 400, marginLeft: 3 }}>{u}</span> : null}
      </span>
    </div>
  );
}

window.Eyebrow = Eyebrow;
window.MethodSwatch = MethodSwatch;
window.MethodChip = MethodChip;
window.NumStat = NumStat;
