// =========================================================================
// architecture.jsx — Three-layer system diagram with hover detail.
// =========================================================================

function ArchitectureSection() {
  const [active, setActive] = React.useState("son");

  const layers = [
    {
      id: "gnn",
      num: "Layer 1",
      title: "GNN-DQN preference model",
      tags: ["learned", "topology-invariant"],
      body:
        "Represents the cell network as a graph. Each node carries UE-observable radio features. Two GCN layers + a Q-head emit a per-UE preference distribution over candidate target cells.",
      detail: {
        head: "Inputs",
        list: [
          "RSRP and RSRQ from each candidate cell",
          "RSRQ-derived load proxy (no PRB required)",
          "RSRP / RSRQ trend features (Δ over window)",
          "UE speed and time since last handover",
          "Serving cell indicator + previous serving cell",
          "Signal usability flag",
        ],
        outputs: [
          "Q-values over candidate target cells",
          "Per-UE target preference (argmax)",
        ],
        codeRef: "src/handover_gnn_dqn/models/gnn_dqn.py",
      },
    },
    {
      id: "son",
      num: "Layer 2",
      title: "SON translation layer",
      tags: ["safety-bounded", "ours"],
      body:
        "Aggregates per-UE preferences over (serving, target) cell pairs. Translates the aggregate into bounded CIO offsets and conservative TTT updates. Monitors KPIs and rolls back on regression.",
      detail: {
        head: "Bounds & rules",
        list: [
          "CIO update bound · ±6 dB per pair",
          "TTT scaled up when ping-pong is high",
          "Update rate limited to operator policy",
          "KPI rollback when throughput or fairness drops",
          "Reports update count, CIO magnitude, rollback count",
        ],
        outputs: [
          "Bounded CIO deltas per cell pair",
          "Scaled TTT setpoint",
          "Rollback signal on KPI regression",
        ],
        codeRef: "src/handover_gnn_dqn/son/controller.py",
      },
    },
    {
      id: "exec",
      num: "Layer 3",
      title: "3GPP execution engine",
      tags: ["standard", "carrier-grade"],
      body:
        "Runs the unmodified 3GPP Event A3 / TTT handover with the SON-supplied parameters. Emits standard X2/Xn handover commands. Telemetry feeds back into the SON controller.",
      detail: {
        head: "Behaviour",
        list: [
          "Standard A3 entering / leaving conditions",
          "Hysteresis + offset applied per pair",
          "TTT timer per UE",
          "KPM telemetry returned to SON layer",
          "Future: O-RAN E2 KPM via the oran_e2 feature profile",
        ],
        outputs: [
          "Handover command (X2/Xn)",
          "KPM counters (PRB, throughput, ping-pong)",
        ],
        codeRef: "src/handover_gnn_dqn/oran/adapter.py",
      },
    },
  ];

  const det = layers.find((l) => l.id === active).detail;
  const detLayer = layers.find((l) => l.id === active);

  return (
    <section className="section" id="architecture">
      <div className="section-inner">
        <Eyebrow num="03">Architecture</Eyebrow>
        <h2 className="section-title">Three layers between the agent and the radio.</h2>
        <p className="section-lede">
          The learned preference model never directly issues handover commands. Instead, a safety-bounded
          SON controller turns its preferences into 3GPP-compliant CIO and TTT updates. Hover any layer
          to inspect its inputs, outputs, and code path.
        </p>

        <div className="two-col" style={{ marginTop: 24 }}>
          <div>
            <div className="arch-stack">
              {layers.map((l, i) => (
                <React.Fragment key={l.id}>
                  <div
                    className={`arch-layer ${active === l.id ? "is-active" : ""}`}
                    onMouseEnter={() => setActive(l.id)}
                    onClick={() => setActive(l.id)}
                  >
                    <div className="arch-layer-head">
                      <span className="arch-layer-num">{l.num}</span>
                      <h3 className="arch-layer-title">{l.title}</h3>
                      <div className="arch-layer-tags">
                        {l.tags.map((t) => (
                          <span key={t} className={`tag ${t === "ours" ? "tag--signal" : ""}`}>{t}</span>
                        ))}
                      </div>
                    </div>
                    <p className="arch-layer-body">{l.body}</p>
                  </div>
                  {i < layers.length - 1 ? <div className="arch-flow">↓ preferences ↓</div> : null}
                </React.Fragment>
              ))}
            </div>
          </div>

          <aside className="arch-detail">
            <div className="arch-detail-title">Detail · {detLayer.num}</div>
            <h4>{detLayer.title}</h4>
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--ink-3)", marginTop: 8 }}>{det.head}</div>
            <ul className="arch-list" style={{ marginTop: 8 }}>
              {det.list.map((x, i) => <li key={i}><span>{x}</span></li>)}
            </ul>
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--ink-3)", marginTop: 14 }}>Outputs</div>
            <ul className="arch-list" style={{ marginTop: 8 }}>
              {det.outputs.map((x, i) => <li key={i}><span>{x}</span></li>)}
            </ul>
            <div className="arch-code-ref">
              <b>code</b> · <span>{det.codeRef}</span>
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}

window.ArchitectureSection = ArchitectureSection;
