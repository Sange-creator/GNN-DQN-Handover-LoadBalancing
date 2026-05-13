// =========================================================================
// closing.jsx — Topology generalization deep-dive, Ablations, Methods/Features,
// and Footer.
// =========================================================================

function GeneralizationSection() {
  const trainedRef = React.useRef(null);
  const deployedRef = React.useRef(null);
  const [deployed, setDeployed] = React.useState(false);

  React.useEffect(() => {
    let raf;
    const trainedSim = new window.SimEngine("suburban", "son_gnn_dqn");
    const deployedSim = new window.SimEngine("kathmandu_real", deployed ? "son_gnn_dqn" : "a3_ttt");

    let last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.08, (now - last) / 1000) * 1.4;
      last = now;
      trainedSim.step(dt);
      deployedSim.step(dt);
      if (trainedRef.current) window.drawSim(trainedRef.current, trainedSim);
      if (deployedRef.current) window.drawSim(deployedRef.current, deployedSim);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [deployed]);

  return (
    <section className="section section--alt" id="generalization">
      <div className="section-inner">
        <Eyebrow num="07">Topology generalization</Eyebrow>
        <h2 className="section-title">From 3 cells to 25 cells, with no retraining.</h2>
        <p className="section-lede">
          The defining test of a graph-based handover policy: train on a small synthetic deployment, drop
          it onto a real-world topology of a completely different size and shape, and ask whether the
          policy is still useful. The GNN-DQN's per-cell representation is topology-invariant, so the
          same checkpoint runs unchanged across cell counts.
        </p>

        <div className="generalization-stage">
          <div className="generalization-side">
            <h4 className="generalization-side-title">Trained on</h4>
            <div className="generalization-mini-canvas">
              <canvas ref={trainedRef} />
              <div className="generalization-side-badge">3 cells · synthetic</div>
            </div>
            <ul className="arch-list" style={{ marginTop: 12 }}>
              <li><span /><span>3-cell triangular synthetic layout</span></li>
              <li><span /><span>20 UEs · uniform mobility</span></li>
              <li><span /><span>Training reward signal</span></li>
            </ul>
          </div>

          <div className="generalization-arrow">
            <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)", textAlign: "center" }}>
              zero-shot
              <br />
              <span style={{ color: "var(--signal)", fontSize: 20 }}>→</span>
              <br />
              same checkpoint
            </div>
            <button
              className="btn btn-primary"
              onClick={() => setDeployed(!deployed)}
              style={{ marginTop: 18 }}
            >
              {deployed ? "✓ Deployed" : "Deploy →"}
            </button>
          </div>

          <div className="generalization-side">
            <h4 className="generalization-side-title">Deployed on</h4>
            <div className="generalization-mini-canvas">
              <canvas ref={deployedRef} />
              <div className="generalization-side-badge generalization-side-badge--real">25 cells · Kathmandu · real coords</div>
            </div>
            <ul className="arch-list" style={{ marginTop: 12 }}>
              <li><span /><span>OpenCellID Kathmandu valley extract</span></li>
              <li><span /><span>250 UEs · mixed mobility</span></li>
              <li><span /><span>{deployed ? <b style={{ color: "var(--signal)" }}>GNN-DQN+SON active</b> : "A3/TTT baseline"}</span></li>
            </ul>
          </div>
        </div>

        <p style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--ink-3)", textAlign: "center", marginTop: 20 }}>
          ↳ The same network weights handle 3 cells and 25 cells. A flat-MLP baseline would require retraining for each new layout.
        </p>
      </div>
    </section>
  );
}

// =========================================================================

function AblationsSection() {
  const ablations = [
    {
      n: "08.1",
      title: "SON safety layer on / off",
      sub: "Does the bounded CIO controller earn its place?",
      rows: [
        { variant: "GNN-DQN raw (no SON)", thr: 8.2, ping: 9.7, jain: 0.79 },
        { variant: "GNN-DQN + SON", thr: 8.6, ping: 0.4, jain: 0.91, best: true },
      ],
      caption: "Removing the SON layer recovers most of the throughput but ping-pong jumps almost 25×. The SON layer is essentially free on throughput and decisive on stability.",
    },
    {
      n: "08.2",
      title: "GNN encoder vs flat MLP",
      sub: "Does the graph structure pay off?",
      rows: [
        { variant: "Flat-MLP DQN (3-cell train)", thr: 5.9, ping: 4.2, jain: 0.81 },
        { variant: "GNN-DQN (3-cell train)", thr: 8.6, ping: 0.4, jain: 0.91, best: true },
        { variant: "Flat-MLP DQN (Kathmandu)", thr: 2.1, ping: 19.3, jain: 0.61 },
        { variant: "GNN-DQN (Kathmandu)", thr: 8.4, ping: 0.5, jain: 0.90, best: true },
      ],
      caption: "Flat-MLP cannot generalize beyond its training cell count. GNN-DQN holds throughput on a 25-cell unseen topology.",
    },
    {
      n: "08.3",
      title: "Feature profile · UE-only vs O-RAN/E2",
      sub: "Is the RSRQ-as-load-proxy compromise acceptable?",
      rows: [
        { variant: "UE-only profile (deployable today)", thr: 8.6, ping: 0.4, jain: 0.91, best: true },
        { variant: "O-RAN/E2 profile (PRB telemetry)", thr: 8.9, ping: 0.3, jain: 0.93 },
      ],
      caption: "True PRB counters add ~3% throughput. We ship the UE-only profile because it runs on networks today; the E2 profile is reserved for future deployment.",
    },
  ];
  return (
    <section className="section" id="ablations">
      <div className="section-inner">
        <Eyebrow num="08">Ablations</Eyebrow>
        <h2 className="section-title">Which parts of the system actually matter?</h2>
        <p className="section-lede">
          Three controlled comparisons isolate the SON safety layer, the graph encoder, and the feature
          profile. Each ablation runs the same training budget on the same scenario mix.
        </p>

        <div className="ablation-stack">
          {ablations.map((a) => (
            <div className="ablation-card card" key={a.n}>
              <div className="ablation-card-head">
                <div>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>{a.n}</span>
                  <h3 style={{ fontFamily: "var(--serif)", fontSize: 20, fontWeight: 500, margin: "2px 0 0" }}>{a.title}</h3>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--ink-3)" }}>{a.sub}</div>
                </div>
              </div>
              <table className="ablation-table">
                <thead>
                  <tr>
                    <th>Variant</th>
                    <th>Thr (Mbps/UE)</th>
                    <th>Ping-pong %</th>
                    <th>Jain</th>
                  </tr>
                </thead>
                <tbody>
                  {a.rows.map((r) => (
                    <tr key={r.variant} className={r.best ? "is-best" : ""}>
                      <td>{r.variant}{r.best ? <span style={{ marginLeft: 8, color: "var(--signal)", fontFamily: "var(--mono)", fontSize: 10 }}>BEST</span> : null}</td>
                      <td>{r.thr.toFixed(2)}</td>
                      <td>{r.ping.toFixed(2)}</td>
                      <td>{r.jain.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p style={{ marginTop: 12, fontSize: 13.5, color: "var(--ink-2)" }}>↳ {a.caption}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// =========================================================================

function MethodsSection() {
  const [tab, setTab] = React.useState("ue_only");
  const profiles = {
    ue_only: {
      title: "UE-only profile · deployable today",
      desc: "Uses only measurements available on commercial UEs. Requires no network-side telemetry, no O-RAN compliant infrastructure. Compatible with standard drive-test data collection.",
      status: "Active",
      features: [
        { f: "RSRP from each candidate cell", src: "UE measurement report", units: "dBm" },
        { f: "RSRQ from each candidate cell", src: "UE measurement report", units: "dB" },
        { f: "Load proxy", src: "Derived from RSRQ (interference + load)", units: "0–1" },
        { f: "RSRP trend (window Δ)", src: "Computed from recent reports", units: "dBm/s" },
        { f: "RSRQ trend (window Δ)", src: "Computed from recent reports", units: "dB/s" },
        { f: "UE speed", src: "UE positioning or RSRP rate-of-change", units: "m/s" },
        { f: "Time since last handover", src: "UE state", units: "s" },
        { f: "Serving-cell one-hot", src: "UE state", units: "—" },
        { f: "Previous serving cell", src: "UE state", units: "—" },
        { f: "Usability flag", src: "Derived from RSRP threshold", units: "0/1" },
      ],
    },
    oran_e2: {
      title: "O-RAN / E2 profile · future deployment",
      desc: "Extends the UE-only profile with network-side telemetry exposed via O-RAN E2 KPM service models. Requires an O-RAN compliant gNB/eNB. Reserved for a future deployment where this telemetry is available.",
      status: "Reserved",
      features: [
        { f: "All UE-only features", src: "↑ above", units: "—" },
        { f: "PRB utilization · serving cell", src: "E2 KPM · UEThpDl", units: "0–1" },
        { f: "PRB utilization · candidates", src: "E2 KPM · UEThpDl per neighbor", units: "0–1" },
        { f: "Attached UE count per cell", src: "E2 KPM · RRC.ConnEstabAtt", units: "count" },
        { f: "Cell-level mean throughput", src: "E2 KPM · DRB.UEThpDl", units: "Mbps" },
      ],
    },
  };
  const p = profiles[tab];
  return (
    <section className="section section--alt" id="methods">
      <div className="section-inner">
        <Eyebrow num="09">Methods · feature profiles</Eyebrow>
        <h2 className="section-title">Two profiles. One agent. Honest about deployability.</h2>
        <p className="section-lede">
          Many academic methods quietly assume PRB counters that real networks do not expose. The
          framework supports two feature profiles. The UE-only profile is what we publish and what could
          deploy today. The O-RAN profile is documented and tested but reserved for a future deployment.
        </p>

        <div className="methods-tabs">
          <button className={`methods-tab ${tab === "ue_only" ? "is-active" : ""}`} onClick={() => setTab("ue_only")}>
            <span className="methods-tab-badge methods-tab-badge--active">✓ deployable today</span>
            UE_ONLY
          </button>
          <button className={`methods-tab ${tab === "oran_e2" ? "is-active" : ""}`} onClick={() => setTab("oran_e2")}>
            <span className="methods-tab-badge methods-tab-badge--future">⊙ future</span>
            ORAN_E2
          </button>
        </div>

        <div className="methods-body card">
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
            <h3 style={{ fontFamily: "var(--serif)", fontSize: 22, fontWeight: 500, margin: 0 }}>{p.title}</h3>
            <span className="methods-status">{p.status}</span>
          </div>
          <p style={{ marginTop: 6, color: "var(--ink-2)", fontSize: 14.5 }}>{p.desc}</p>

          <table className="methods-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Source</th>
                <th>Units</th>
              </tr>
            </thead>
            <tbody>
              {p.features.map((f) => (
                <tr key={f.f}>
                  <td><span className="code">{f.f}</span></td>
                  <td>{f.src}</td>
                  <td><span style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--ink-3)" }}>{f.units}</span></td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="methods-note">
            <b>On the RSRQ-as-load-proxy decision.</b> RSRQ depends on both signal level and interference,
            which is in turn driven by cell load. For UE-only deployment, this is the best load signal a
            UE can observe. We acknowledge this is a proxy, validate it against PRB counters in
            simulation, and clearly mark the E2 profile as the path to direct measurement.
          </div>
        </div>
      </div>
    </section>
  );
}

// =========================================================================

function ProjectFooter() {
  return (
    <footer className="site-footer">
      <div className="section-inner">
        <div className="footer-grid">
          <div>
            <div className="footer-brand">
              <span className="brand-dot" />
              <span style={{ fontFamily: "var(--mono)", letterSpacing: "0.08em", fontSize: 12.5, fontWeight: 600 }}>GNN-DQN · HANDOVER</span>
            </div>
            <p style={{ marginTop: 12, fontSize: 13.5, color: "var(--ink-3)", maxWidth: 320 }}>
              GNN-DQN driven predictive handover and load optimization for LTE networks.
              A topology-invariant, safety-bounded mobility-management framework.
            </p>
          </div>

          <div>
            <h5 className="footer-h">Project</h5>
            <ul className="footer-list">
              <li><a href="#abstract">Abstract</a></li>
              <li><a href="#architecture">Architecture</a></li>
              <li><a href="#atlas">Scenario atlas</a></li>
              <li><a href="#lab">Live scenario lab</a></li>
              <li><a href="#results">Results dashboard</a></li>
              <li><a href="#generalization">Topology generalization</a></li>
              <li><a href="#ablations">Ablations</a></li>
              <li><a href="#methods">Feature profiles</a></li>
              <li><a href="#reproduce">Reproduce</a></li>
            </ul>
          </div>

          <div>
            <h5 className="footer-h">Authors</h5>
            <ul className="footer-list">
              <li>Rojin Puri</li>
              <li>Rujan Subedi</li>
              <li>Saange Tamang</li>
              <li>Sulav Kandel</li>
            </ul>
            <h5 className="footer-h" style={{ marginTop: 18 }}>Department</h5>
            <ul className="footer-list">
              <li>Electronics & Computer Engineering</li>
              <li>IOE Paschimanchal Campus</li>
            </ul>
          </div>

          <div>
            <h5 className="footer-h">Stack</h5>
            <ul className="footer-list">
              <li>PyTorch · Graph Convolutional Network</li>
              <li>NumPy · Pandas · Matplotlib</li>
              <li>OpenCellID for real coordinates</li>
              <li>Custom ns-3-flavoured LTE simulator</li>
              <li>3GPP Event A3 / TTT compliant SON layer</li>
            </ul>
          </div>
        </div>

        <div className="footer-foot">
          <span>© 2026 · Major Project · IOE Paschimanchal Campus</span>
          <span style={{ marginLeft: "auto" }}>Built with HTML + Canvas. No tracking.</span>
        </div>
      </div>
    </footer>
  );
}

window.GeneralizationSection = GeneralizationSection;
window.AblationsSection = AblationsSection;
window.MethodsSection = MethodsSection;
window.ProjectFooter = ProjectFooter;
