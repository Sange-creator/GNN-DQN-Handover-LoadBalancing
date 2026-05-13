// =========================================================================
// lab.jsx — Live Scenario Lab. Toolbar, multi-method panes, KPI strip.
// =========================================================================

function LabPane({ scenarioId, methodId, isHighlight, speed, paused }) {
  const cvRef = React.useRef(null);
  const mapRef = React.useRef(null);
  const leafletRef = React.useRef(null);
  const simRef = React.useRef(null);
  const [kpi, setKpi] = React.useState({
    thr_avg: 0, thr_p5: 0, jain: 0,
    pingpong_pct: 0, ho_per_min: 0, outage_pct: 0,
  });
  // Stable kpi setter
  const onKpi = React.useCallback((k) => setKpi(k), []);

  // Build sim when scenario or method changes
  React.useEffect(() => {
    simRef.current = new window.SimEngine(scenarioId, methodId, { kpiOnUpdate: () => {} });
  }, [scenarioId, methodId]);

  // Real-coordinate scenarios get an OpenStreetMap basemap. The simulator
  // still owns policy state; Leaflet only supplies geographic context.
  React.useEffect(() => {
    const scenario = window.SCENARIO_BY_ID[scenarioId];
    const data = window.getScenarioPositions(scenarioId);

    if (leafletRef.current) {
      leafletRef.current.remove();
      leafletRef.current = null;
    }

    if (!scenario?.real || !data?.boundsLatLon || !mapRef.current || !window.L) return undefined;

    const map = window.L.map(mapRef.current, {
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false,
      touchZoom: false,
      tap: false,
      fadeAnimation: false,
      zoomAnimation: false,
      markerZoomAnimation: false,
      preferCanvas: true,
    });

    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      opacity: 0.82,
      updateWhenIdle: true,
      updateWhenZooming: false,
      keepBuffer: 1,
    }).addTo(map);

    map.fitBounds(window.L.latLngBounds(data.boundsLatLon).pad(0.35), {
      animate: false,
      padding: [22, 22],
    });

    leafletRef.current = map;
    setTimeout(() => {
      if (leafletRef.current) leafletRef.current.invalidateSize();
    }, 80);

    return () => {
      map.remove();
      if (leafletRef.current === map) leafletRef.current = null;
    };
  }, [scenarioId]);

  // Animation loop
  React.useEffect(() => {
    let raf;
    let last = performance.now();
    let lastDraw = 0;
    let lastKpi = 0;
    const tick = (now) => {
      const dt = Math.min(0.1, ((now - last) / 1000)) * speed;
      last = now;
      if (!paused && simRef.current) simRef.current.step(dt);
      if (cvRef.current && simRef.current && now - lastDraw > 33) {
        window.drawSim(cvRef.current, simRef.current, { map: leafletRef.current });
        lastDraw = now;
      }
      if (simRef.current && now - lastKpi > 350) {
        setKpi(simRef.current.kpi);
        lastKpi = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [speed, paused, scenarioId]);

  const m = window.METHOD_BY_ID[methodId];
  const scenario = window.SCENARIO_BY_ID[scenarioId];

  return (
    <div className={`lab-pane ${isHighlight ? "is-highlight" : ""} ${scenario.real ? "lab-pane--map" : ""}`}>
      {scenario.real ? <div className="lab-pane-map" ref={mapRef} aria-hidden="true" /> : null}
      <canvas ref={cvRef} />
      <div className="lab-pane-label" style={isHighlight ? { borderColor: m.raw, color: "#fff" } : undefined}>
        <span className="swatch" style={{
          background: m.dotted ? "transparent" : m.raw,
          border: m.dotted ? `1.5px dotted ${m.raw}` : "none",
        }} />
        <span style={{ color: m.ours ? "oklch(0.85 0.10 245)" : undefined, fontWeight: m.ours ? 600 : 400 }}>
          {m.label}
        </span>
        {m.ours ? <span style={{
          padding: "1px 6px", borderRadius: 100,
          background: "oklch(0.50 0.150 245 / 0.20)",
          border: "1px solid oklch(0.50 0.150 245 / 0.5)",
          fontSize: 9.5, marginLeft: 4, letterSpacing: "0.06em"
        }}>OURS</span> : null}
      </div>
      {scenario.real ? (
        <div className="lab-map-source">
          <span>OpenStreetMap basemap</span>
          <span>UE-only overlay</span>
        </div>
      ) : null}
      {/* Per-pane mini KPI */}
      <div style={{
        position: "absolute", right: 12, top: 12,
        zIndex: 4,
        display: "flex", flexDirection: "column", gap: 3,
        padding: "6px 9px",
        background: "oklch(0.10 0.014 248 / 0.75)",
        border: "1px solid var(--rule-deep)",
        borderRadius: 6,
        fontFamily: "var(--mono)", fontSize: 10.5,
        color: "var(--ink-on-deep)",
        minWidth: 130,
      }}>
        <KpiLine k="Thr avg" v={kpi.thr_avg.toFixed(2)} u="Mbps" />
        <KpiLine k="P5 thr" v={kpi.thr_p5.toFixed(2)} u="Mbps" />
        <KpiLine k="Ping-pong" v={kpi.pingpong_pct.toFixed(1)} u="%"
                 tone={kpi.pingpong_pct > 8 ? "bad" : kpi.pingpong_pct < 1 ? "good" : null} />
        <KpiLine k="Jain" v={kpi.jain.toFixed(2)} />
      </div>
    </div>
  );
}

function KpiLine({ k, v, u, tone }) {
  const color = tone === "bad" ? "oklch(0.78 0.130 25)"
              : tone === "good" ? "oklch(0.78 0.110 145)"
              : "var(--ink-on-deep)";
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
      <span style={{ color: "oklch(0.65 0.014 250)" }}>{k}</span>
      <span style={{ color, fontVariantNumeric: "tabular-nums" }}>
        {v}{u ? <span style={{ color: "oklch(0.65 0.014 250)", marginLeft: 3 }}>{u}</span> : null}
      </span>
    </div>
  );
}

function LiveLab({ scenarioId, onLoad }) {
  const [methods, setMethods] = React.useState(["a3_ttt", "son_gnn_dqn"]);
  const [speed, setSpeed] = React.useState(1.5);
  const [paused, setPaused] = React.useState(false);
  const s = window.SCENARIO_BY_ID[scenarioId];

  const toggleMethod = (id) => {
    if (methods.includes(id)) {
      if (methods.length > 1) setMethods(methods.filter((x) => x !== id));
    } else {
      if (methods.length < 3) setMethods([...methods, id]);
      else setMethods([...methods.slice(1), id]);
    }
  };

  const cols = methods.length;

  return (
    <section className="lab" id="lab">
      <div className="section-inner">
        <Eyebrow num="05">Live scenario lab</Eyebrow>
        <h2 className="section-title">Watch handover policies run side-by-side.</h2>
        <p className="section-lede">
          A map-backed in-browser visualisation of the cell environment defined in
          <span className="code"> src/handover_gnn_dqn/env/simulator.py</span>. Pick a scenario, compare up
          to three methods, and see UE trajectories, handover events, ping-pongs, and live load
          redistribution. This is a visualisation, not real-time inference — KPIs are computed from a
          simplified physics model and converge to the published evaluation table.
        </p>

        <div className="lab-shell">
          <div className="lab-toolbar">
            <div className="lab-toolbar-row">
              <span className="lab-toolbar-label">Scenario</span>
              <select className="lab-select"
                      value={scenarioId}
                      onChange={(e) => onLoad(e.target.value)}>
                {window.SCENARIOS.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name} · {s.cells} cells · {s.ues} UEs
                  </option>
                ))}
              </select>
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "oklch(0.65 0.014 250)" }}>
                {s.locale} · {s.mobility}
              </span>
            </div>
            <div className="lab-toolbar-row">
              <span className="lab-toolbar-label">Compare</span>
              {window.METHODS.map((m) => (
                <button
                  key={m.id}
                  className={`lab-method-toggle ${methods.includes(m.id) ? "is-active" : ""}`}
                  onClick={() => toggleMethod(m.id)}
                  title={m.desc}
                >
                  <span className="swatch" style={{
                    background: m.dotted ? "transparent" : m.raw,
                    border: m.dotted ? `1.5px dotted ${m.raw}` : "none",
                  }} />
                  {m.short}
                </button>
              ))}
            </div>
            <div className="lab-toolbar-row">
              <span className="lab-toolbar-label">Speed</span>
              {[0.5, 1, 2, 4].map((sp) => (
                <button key={sp}
                        className={`lab-btn ${speed === sp ? "is-active" : ""}`}
                        onClick={() => setSpeed(sp)}>{sp}×</button>
              ))}
              <button className="lab-btn"
                      onClick={() => setPaused(!paused)}
                      style={{ marginLeft: 8 }}>
                {paused ? "▶ Play" : "⏸ Pause"}
              </button>
            </div>
          </div>

          <div className="lab-stage">
            <div className={`lab-canvas-grid cols-${cols}`}>
              {methods.map((mId) => (
                <LabPane
                  key={`${scenarioId}-${mId}`}
                  scenarioId={scenarioId}
                  methodId={mId}
                  isHighlight={mId === "son_gnn_dqn"}
                  speed={speed}
                  paused={paused}
                />
              ))}
            </div>
            <ScenarioConditionsBar scenario={s} />
          </div>
        </div>

        <div style={{
          marginTop: 16, fontFamily: "var(--mono)", fontSize: 11,
          color: "oklch(0.65 0.014 250)", textAlign: "center"
        }}>
          ↳ Visualisation only. Final numerical results are produced by
          <span style={{ color: "var(--ink-on-deep)" }}> scripts/evaluate.py</span> and rendered in the Results section below.
        </div>
      </div>
    </section>
  );
}

function ScenarioConditionsBar({ scenario }) {
  const s = scenario;
  if (!s) return null;
  const topologyLabel = s.id.includes("pokhara")
    ? "Pokhara map"
    : s.id.includes("kathmandu")
      ? "Kathmandu map"
      : s.layout.replace(/_/g, " ");
  const items = [
    { k: "Topology", v: topologyLabel },
    { k: "Cells", v: s.cells },
    { k: "UEs", v: s.ues },
    { k: "ISD", v: s.isd_m ? `${s.isd_m} m` : "real" },
    { k: "Speed", v: `${s.speed_min.toFixed(1)}–${s.speed_max.toFixed(1)} m/s` },
    { k: "Mobility", v: s.mobility },
    { k: "Demand", v: `${s.demand_min}–${s.demand_max} Mbps` },
    { k: "σ shadow", v: `${s.shadow_db} dB` },
    { k: "Group", v: s.group },
  ];
  return (
    <div className="lab-kpi-strip" style={{ gridTemplateColumns: `repeat(${items.length}, 1fr)` }}>
      {items.map((it, i) => (
        <div className="lab-kpi" key={i}>
          <span className="lab-kpi-k">{it.k}</span>
          <span className="lab-kpi-v" style={{ fontSize: 13 }}>{it.v}</span>
        </div>
      ))}
    </div>
  );
}

window.LiveLab = LiveLab;
