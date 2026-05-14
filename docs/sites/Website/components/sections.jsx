// =========================================================================
// sections.jsx — Hero, Abstract, Problem, Reproduce, and other static
// narrative sections.
// =========================================================================

function HeroFigure() {
  // Animated mini-network: cells + UEs orbiting, with handover arcs.
  const ref = React.useRef(null);
  React.useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let raf;
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    const cells = [
      { x: 0.20, y: 0.30, load: 0.4 },
      { x: 0.55, y: 0.25, load: 0.7 },
      { x: 0.78, y: 0.55, load: 0.55 },
      { x: 0.30, y: 0.72, load: 0.45 },
      { x: 0.62, y: 0.78, load: 0.5 },
    ];

    const ues = Array.from({ length: 22 }, (_, i) => ({
      id: i,
      x: Math.random(),
      y: Math.random(),
      vx: (Math.random() - 0.5) * 0.0008,
      vy: (Math.random() - 0.5) * 0.0008,
      cell: 0,
      handover_t: 0,
    }));

    const arcs = [];

    const resize = () => {
      const r = cv.getBoundingClientRect();
      cv.width = Math.round(r.width * dpr);
      cv.height = Math.round(r.height * dpr);
    };
    resize();
    window.addEventListener("resize", resize);

    const ink = "#1a2330";
    const signal = "oklch(0.50 0.150 245)";
    const amber = "oklch(0.72 0.130 65)";

    let t0 = performance.now();
    const draw = (now) => {
      const dt = Math.min(0.05, (now - t0) / 1000);
      t0 = now;
      const W = cv.width, H = cv.height;
      ctx.clearRect(0, 0, W, H);

      // Subtle grid
      ctx.strokeStyle = "rgba(20,30,50,0.05)";
      ctx.lineWidth = 1;
      for (let i = 1; i < 12; i++) {
        ctx.beginPath();
        ctx.moveTo((i / 12) * W, 0);
        ctx.lineTo((i / 12) * W, H);
        ctx.stroke();
      }
      for (let i = 1; i < 9; i++) {
        ctx.beginPath();
        ctx.moveTo(0, (i / 9) * H);
        ctx.lineTo(W, (i / 9) * H);
        ctx.stroke();
      }

      // Cell coverage rings
      cells.forEach((c, idx) => {
        const cx = c.x * W, cy = c.y * H;
        const baseR = Math.min(W, H) * 0.18;
        for (let k = 3; k > 0; k--) {
          ctx.beginPath();
          ctx.arc(cx, cy, baseR * k * 0.6, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(40,90,200,${0.05 / k})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        // Tower
        ctx.fillStyle = ink;
        ctx.beginPath();
        ctx.arc(cx, cy, 4 * dpr, 0, Math.PI * 2);
        ctx.fill();
        // Load bar
        ctx.fillStyle = c.load > 0.7 ? amber : signal;
        ctx.fillRect(cx - 12 * dpr, cy - 14 * dpr, 24 * dpr * c.load, 2 * dpr);
        ctx.strokeStyle = "rgba(20,30,50,0.25)";
        ctx.strokeRect(cx - 12 * dpr, cy - 14 * dpr, 24 * dpr, 2 * dpr);
      });

      // UE motion + cell assignment
      ues.forEach((u) => {
        u.x += u.vx;
        u.y += u.vy;
        if (u.x < 0.05 || u.x > 0.95) u.vx *= -1;
        if (u.y < 0.05 || u.y > 0.95) u.vy *= -1;
        // Find best cell
        let best = 0, bestD = Infinity;
        cells.forEach((c, i) => {
          const dx = c.x - u.x, dy = c.y - u.y;
          const d = dx * dx + dy * dy;
          if (d < bestD) { bestD = d; best = i; }
        });
        if (best !== u.cell) {
          arcs.push({
            from: cells[u.cell], to: cells[best], t: 0,
          });
          u.cell = best;
          u.handover_t = 1;
        }
        u.handover_t = Math.max(0, u.handover_t - dt * 2);

        const px = u.x * W, py = u.y * H;
        ctx.fillStyle = u.handover_t > 0 ? amber : signal;
        ctx.beginPath();
        ctx.arc(px, py, (1.8 + u.handover_t * 1.5) * dpr, 0, Math.PI * 2);
        ctx.fill();
      });

      // Handover arcs
      for (let i = arcs.length - 1; i >= 0; i--) {
        const a = arcs[i];
        a.t += dt * 1.4;
        if (a.t > 1) { arcs.splice(i, 1); continue; }
        const fx = a.from.x * W, fy = a.from.y * H;
        const tx = a.to.x * W, ty = a.to.y * H;
        const mx = (fx + tx) / 2, my = (fy + ty) / 2 - 30 * dpr;
        ctx.strokeStyle = `rgba(216,150,40,${(1 - a.t) * 0.7})`;
        ctx.lineWidth = 1.4 * dpr;
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.quadraticCurveTo(mx, my, tx, ty);
        ctx.stroke();
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="hero-figure">
      <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} />
      <div style={{
        position: "absolute", left: 14, bottom: 12, fontFamily: "var(--mono)",
        fontSize: 11, color: "var(--ink-3)", display: "flex", gap: 14
      }}>
        <span><span style={{ color: "var(--signal)" }}>●</span> UE</span>
        <span><span style={{ color: "var(--amber)" }}>●</span> Handover event</span>
        <span style={{ color: "var(--ink-3)" }}>5 cells · 22 UEs · live</span>
      </div>
    </div>
  );
}

function HeroSection() {
  return (
    <section className="hero">
      <div className="hero-inner">
        <div>
          <span className="hero-tag">
            <span className="hero-tag-dot" />
            <span>Major Project · IOE Paschimanchal Campus · 2026</span>
          </span>
          <h1 className="hero-title">
            <em>GNN-DQN</em> driven predictive handover and load optimization for <em>LTE</em> networks.
          </h1>
          <p className="hero-sub">
            A topology-invariant graph reinforcement learning agent paired with a safety-bounded
            self-organising network controller. Trained on small synthetic deployments, it generalises
            zero-shot to a 25-cell Kathmandu topology and balances real-world peak-hour load across
            Lakeside and Mahendrapul without ping-pong handovers.
          </p>
          <div className="hero-actions">
            <a className="btn btn-primary" href="#lab">View live scenario lab →</a>
            <a className="btn btn-ghost" href="#results">Results dashboard</a>
            <a className="btn btn-ghost" href="#architecture">System architecture</a>
          </div>
          <div className="hero-meta">
            <span><b>Authors</b> · Rojin Puri · Rujan Subedi · Saange Tamang · Sulav Kandel</span>
            <span><b>Dept.</b> · Electronics, Communication and Information Engineering</span>
            <span><b>Status</b> · Evaluation complete · 20 seeds</span>
          </div>
        </div>
        <HeroFigure />
      </div>
    </section>
  );
}

// =========================================================================

function AbstractSection() {
  return (
    <section className="section" id="abstract">
      <div className="section-inner two-col">
        <div>
          <Eyebrow num="01">Abstract</Eyebrow>
          <h2 className="section-title">A relational, safety-bounded approach to mobility management.</h2>
        </div>
        <div>
          <p style={{ fontSize: 16, lineHeight: 1.65, color: "var(--ink-2)", margin: 0 }}>
            As LTE and 5G networks densify to meet next-generation throughput and latency demands,
            traditional handover mechanisms based on signal-strength triggers like 3GPP Event A3 become
            reactive and load-blind. They produce ping-pong handovers, congested cells, and degraded
            user throughput in dense regions such as Lakeside and Mahendrapul during peak hours.
          </p>
          <p style={{ fontSize: 16, lineHeight: 1.65, color: "var(--ink-2)", marginTop: 18 }}>
            This project proposes <b style={{ color: "var(--signal)" }}>GNN-DQN driven predictive handover</b>:
            a topology-invariant graph reinforcement learning preference model whose recommendations are
            translated into bounded CIO and TTT updates by a self-organising network controller. The
            framework operates with a UE-only feature profile so it remains deployable without network-side
            PRB counters, and uses RSRQ as a practical load proxy. A separate O-RAN/E2 profile is reserved
            for a future deployment that exposes true PRB telemetry.
          </p>
          <p style={{ fontSize: 16, lineHeight: 1.65, color: "var(--ink-2)", marginTop: 18 }}>
            Evaluation across eleven scenarios — covering urban, highway, rural, event, and real
            OpenCellID-derived deployments in Pokhara and Kathmandu — characterises throughput, fairness,
            and stability against six baselines. Final numbers populate this report after the current
            training run completes.
          </p>
        </div>
      </div>
    </section>
  );
}

// =========================================================================

function ProblemSection() {
  const items = [
    {
      n: "01",
      t: "Reactive triggers",
      b: "A3 / TTT only fires when a UE is already in a degraded radio condition. By the time the handover completes, the user may have suffered an outage, especially on highways where mobility is fast.",
    },
    {
      n: "02",
      t: "Load-blind decisions",
      b: "Strongest-RSRP attaches every UE to the same congested cell during peak hours. In dense areas like Lakeside, this saturates the central tower while neighbours sit half-empty.",
    },
    {
      n: "03",
      t: "Ping-pong instability",
      b: "Naive load-aware heuristics over-correct and bounce UEs between cells. We measured 11.27% ping-pong rate on a load-aware baseline — unacceptable for a real network.",
    },
    {
      n: "04",
      t: "Topology fragility",
      b: "An MLP-based DRL agent trained on a fixed cell count collapses when deployed on a different layout. Inputs change shape; the policy stops working.",
    },
    {
      n: "05",
      t: "Carrier-grade safety",
      b: "Operators cannot deploy a black-box policy that might issue arbitrary handover commands. Any change must respect 3GPP parameter ranges and roll back when KPIs regress.",
    },
    {
      n: "06",
      t: "Deployability",
      b: "Many academic methods need PRB counters not exposed on today's networks. A practical solution must run on UE-observable measurements and be drive-test compatible.",
    },
  ];
  return (
    <section className="section section--alt" id="problem">
      <div className="section-inner">
        <Eyebrow num="02">The problem</Eyebrow>
        <h2 className="section-title">Six failure modes of conventional handover.</h2>
        <p className="section-lede">
          Handover is the heartbeat of mobility management. When it misfires, every other LTE optimisation
          downstream pays the cost. The proposed framework explicitly addresses each of these failure modes.
        </p>
        <div className="problem-grid">
          {items.map((it) => (
            <div className="problem-card" key={it.n}>
              <span className="problem-card-num">{it.n}</span>
              <h3>{it.t}</h3>
              <p>{it.b}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// =========================================================================

function ReproduceSection() {
  const cmds = [
    {
      title: "Smoke test · UE-only",
      sub: "configs/experiments/smoke_ue.json",
      cmd: "python3 scripts/train.py --config configs/experiments/smoke_ue.json",
    },
    {
      title: "Smoke test · O-RAN/E2",
      sub: "configs/experiments/smoke_oran.json",
      cmd: "python3 scripts/train.py --config configs/experiments/smoke_oran.json",
    },
    {
      title: "Multi-scenario training",
      sub: "Main publishable path",
      cmd: "python3 scripts/train.py --config configs/experiments/multiscenario_ue.json",
    },
    {
      title: "Pokhara peak hour",
      sub: "configs/experiments/pokhara_ue.json",
      cmd: "python3 scripts/train.py --config configs/experiments/pokhara_ue.json",
    },
    {
      title: "Evaluation · 20 seeds",
      sub: "scripts/evaluate.py",
      cmd: "python3 scripts/evaluate.py \\\n  --checkpoint results/runs/multiscenario_ue/checkpoints/gnn_dqn.pt \\\n  --out-dir results/runs/multiscenario_ue/eval_20seed \\\n  --seeds 20",
    },
    {
      title: "Generate figures",
      sub: "scripts/generate_figures.py",
      cmd: "python3 scripts/generate_figures.py \\\n  --eval-dir results/runs/multiscenario_ue/eval_20seed",
    },
  ];

  return (
    <section className="section" id="reproduce">
      <div className="section-inner">
        <Eyebrow num="09">Reproduce</Eyebrow>
        <h2 className="section-title">Run any experiment from a single config.</h2>
        <p className="section-lede">
          The training and evaluation pipeline is config-driven. Each experiment is a JSON file under
          <span className="code"> configs/experiments/</span> so that runs are deterministic and
          version-controlled. Long runs are resumable from generated checkpoints.
        </p>
        <div className="repro-grid">
          {cmds.map((c) => (
            <div className="repro-card card" key={c.title}>
              <div className="repro-card-head">
                <h3 className="repro-card-title">{c.title}</h3>
                <span className="repro-card-sub">{c.sub}</span>
              </div>
              <pre className="code-block"><code>{c.cmd}</code></pre>
            </div>
          ))}
        </div>

        <h3 style={{ fontFamily: "var(--serif)", fontWeight: 500, fontSize: 22, margin: "40px 0 12px" }}>Repository layout</h3>
        <pre className="code-block" style={{ fontSize: 12.5 }}><code>{`configs/experiments/      JSON configs for smoke, UE-only, Pokhara, multi-scenario
data/raw/                 OpenCellID, ns-3, synthetic, drive-test inputs
data/processed/           Cleaned/generated datasets (gitignored)
docs/                     Project notes, guides, paper material
scripts/                  Config-driven train / evaluate / data / figure entrypoints
src/handover_gnn_dqn/     Reusable package code
  └── env/                LTE simulator (cell + UE physics)
  └── models/             GNN-DQN and flat MLP baselines
  └── policies/           Heuristic baselines (a3_ttt, load_aware, …)
  └── son/                SON translation layer + bounded CIO controller
  └── topology/           Scenario definitions + OpenCellID loader
  └── rl/                 Replay buffer, target network, training loop
tests/                    Unit / integration / regression acceptance tests
results/runs/             Generated runs (checkpoints, figures, eval CSVs)`}</code></pre>
      </div>
    </section>
  );
}

window.HeroSection = HeroSection;
window.AbstractSection = AbstractSection;
window.ProblemSection = ProblemSection;
window.ReproduceSection = ReproduceSection;
