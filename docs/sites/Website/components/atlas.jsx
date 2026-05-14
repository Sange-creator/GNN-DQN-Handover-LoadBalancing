// =========================================================================
// atlas.jsx — Scenario Atlas: 11 cards with full conditions, click to load.
// =========================================================================

function ScenarioThumb({ scenarioId }) {
  // Tiny SVG sketch of the scenario topology.
  const data = window.getScenarioPositions(scenarioId);
  if (!data) return null;
  const { positions, bbox: bb, landmarks } = data;
  const pad = 16;
  const W = 240, H = 150;
  const w = bb.w || 1, h = bb.h || 1;
  const sx = (x) => pad + ((x - bb.minX) / w) * (W - pad * 2);
  const sy = (y) => H - pad - ((y - bb.minY) / h) * (H - pad * 2);
  const s = window.SCENARIO_BY_ID[scenarioId];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" width="100%" height="100%">
      <defs>
        <pattern id={`grid-${scenarioId}`} width="12" height="12" patternUnits="userSpaceOnUse">
          <path d="M 12 0 L 0 0 0 12" fill="none" stroke="oklch(0.88 0.006 90)" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width={W} height={H} fill={`url(#grid-${scenarioId})`} />
      {landmarks ? landmarks.filter((l) => l.kind === "water").map((l, i) => (
        <ellipse key={i} cx={sx(l.x)} cy={sy(l.y)} rx="22" ry="11"
                 fill="oklch(0.55 0.135 245 / 0.10)" stroke="oklch(0.55 0.135 245 / 0.25)" />
      )) : null}
      {/* Coverage rings */}
      {positions.map(([x, y], i) => (
        <circle key={`r-${i}`} cx={sx(x)} cy={sy(y)} r="14"
                fill="oklch(0.55 0.135 245 / 0.06)"
                stroke="oklch(0.55 0.135 245 / 0.18)"
                strokeWidth="0.6" />
      ))}
      {/* Towers */}
      {positions.map(([x, y], i) => (
        <circle key={`t-${i}`} cx={sx(x)} cy={sy(y)} r="2.4" fill="oklch(0.18 0.012 250)" />
      ))}
      {/* Highway connector */}
      {scenarioId === "highway" ? (
        <polyline
          points={positions.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")}
          fill="none"
          stroke="oklch(0.72 0.130 65 / 0.45)"
          strokeWidth="1.2"
          strokeDasharray="3 3"
        />
      ) : null}
      {/* Real data badge */}
      {s.real ? (
        <g>
          <rect x={W - 60} y={8} width="52" height="14" rx="7" fill="oklch(0.50 0.150 245 / 0.10)" stroke="oklch(0.50 0.150 245 / 0.4)" />
          <text x={W - 34} y={17.5} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="8" fill="oklch(0.50 0.150 245)" letterSpacing="0.5">REAL DATA</text>
        </g>
      ) : null}
    </svg>
  );
}

function ScenarioCard({ scenario, isLoaded, onLoad }) {
  const s = scenario;
  return (
    <article className={`scn-card ${isLoaded ? "is-loaded" : ""}`} onClick={() => onLoad(s.id)}>
      <div className="scn-thumb">
        <ScenarioThumb scenarioId={s.id} />
      </div>

      <header className="scn-card-head">
        <div>
          <h3 className="scn-card-name">{s.name}</h3>
          <div className="scn-card-sub">{s.locale}</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
          {s.group === "training" ? <span className="tag">training</span> : <span className="tag tag--amber">test · unseen</span>}
          {s.real ? <span className="tag tag--signal">real coords</span> : null}
          {s.headline ? <span className="tag tag--moss">headline</span> : null}
        </div>
      </header>

      <p className="scn-card-desc">{s.description}</p>

      <div className="scn-stats">
        <NumStat k="Cells" v={s.cells} />
        <NumStat k="UEs" v={s.ues} />
        <NumStat k="UE / cell" v={(s.ues / s.cells).toFixed(1)} />
        <NumStat k="ISD" v={s.isd_m ? `${s.isd_m}` : "real"} u={s.isd_m ? "m" : ""} />
        <NumStat k="Speed" v={`${s.speed_min.toFixed(1)}–${s.speed_max.toFixed(1)}`} u="m/s" />
        <NumStat k="Demand" v={`${s.demand_min}–${s.demand_max}`} u="Mbps" />
        <NumStat k="Mobility" v={s.mobility} />
        <NumStat k="σ shadow" v={`${s.shadow_db}`} u="dB" />
      </div>

      {s.notes ? (
        <div style={{
          fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)",
          borderTop: "1px dashed var(--rule)", paddingTop: 10
        }}>
          ↳ {s.notes}
        </div>
      ) : null}

      <div className="scn-card-foot">
        <button
          className="btn btn-ghost"
          style={{ padding: "6px 12px", fontSize: 12.5, marginTop: 4 }}
          onClick={(e) => { e.stopPropagation(); onLoad(s.id); }}
        >
          {isLoaded ? "Loaded in geographic lab" : s.real ? "Open geographic lab" : "Load in live lab"}
        </button>
      </div>
    </article>
  );
}

function ScenarioAtlas({ loadedId, onLoad }) {
  const [filter, setFilter] = React.useState("all");
  const filters = [
    { id: "all", label: "All scenarios" },
    { id: "training", label: "Training set" },
    { id: "test", label: "Unseen test set" },
    { id: "real", label: "Real coordinates" },
    { id: "headline", label: "Headline scenarios" },
  ];
  const filtered = window.SCENARIOS.filter((s) => {
    if (filter === "all") return true;
    if (filter === "training") return s.group === "training";
    if (filter === "test") return s.group === "test";
    if (filter === "real") return s.real;
    if (filter === "headline") return s.headline;
    return true;
  });
  return (
    <section className="section section--alt" id="atlas">
      <div className="section-inner">
        <Eyebrow num="04">Scenario atlas</Eyebrow>
        <h2 className="section-title">Real Pokhara first, synthetic layouts as stress tests.</h2>
        <p className="section-lede">
          The live lab defaults to the Pokhara map topology because it is the clearest defense scenario.
          Synthetic hex, highway, event, and coverage-hole layouts remain in the atlas as controlled
          generalization and stress tests. Click a card to load the scenario into the live lab below.
        </p>

        <div className="atlas-toolbar">
          <span className="atlas-toolbar-label">Filter</span>
          <div className="chip-row">
            {filters.map((f) => (
              <button
                key={f.id}
                className={`chip ${filter === f.id ? "is-active" : ""}`}
                onClick={() => setFilter(f.id)}
              >{f.label}</button>
            ))}
          </div>
          <span style={{ marginLeft: "auto", fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>
            {filtered.length} of {window.SCENARIOS.length}
          </span>
        </div>

        <div className="atlas-grid">
          {filtered.map((s) => (
            <ScenarioCard key={s.id} scenario={s} isLoaded={loadedId === s.id} onLoad={onLoad} />
          ))}
        </div>
      </div>
    </section>
  );
}

window.ScenarioAtlas = ScenarioAtlas;
