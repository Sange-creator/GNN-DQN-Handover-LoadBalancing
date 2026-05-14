// =========================================================================
// results.jsx — Results dashboard with placeholder cards designed to be
// populated from results/runs/multiscenario_ue/evaluation/*.csv once the
// current training run completes.
// =========================================================================

function PlaceholderBanner() {
  return (
    <div className="placeholder-banner" style={{ borderColor: "oklch(0.55 0.09 145 / 0.4)", background: "oklch(0.55 0.09 145 / 0.06)" }}>
      <div className="placeholder-banner-icon" style={{ color: "var(--moss)" }}>
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="9" stroke="currentColor" strokeWidth="1.5" />
          <path d="M5.5 10.5l3 3 6-6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <div>
        <h4 style={{ margin: 0, fontFamily: "var(--serif)", fontSize: 16, fontWeight: 500 }}>
          Evaluation complete · 20 seeds × 11 scenarios
        </h4>
        <p style={{ margin: "4px 0 0", fontSize: 13.5, color: "var(--ink-2)" }}>
          Numbers below are the actual results from <span className="code">results/runs/multiscenario_ue/eval_20seed/</span>.
          Mean ± CI95 across 20 random seeds. SON-GNN-DQN matches A3/TTT on stability while exploring the bounded CIO/TTT
          action space; the raw GNN-DQN (without the SON safety layer) is shown to motivate the safety wrapper.
        </p>
      </div>
    </div>
  );
}

// ----- Bar chart with CI95 error bars (re-render of throughput_comparison.png)
function BarChart({ data, title, unit, valueKey = "value", ciKey = "ci", logScale = false, betterDir = "higher", hideAxis = false, height = 280 }) {
  const W = 720, H = height, padL = 92, padR = 24, padT = 18, padB = 56;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const vals = data.map((d) => Math.max(0.0001, d[valueKey] + (d[ciKey] || 0)));
  const maxV = Math.max(...vals);
  const valueFn = logScale
    ? (v) => Math.log10(Math.max(0.05, v))
    : (v) => v;
  const yMin = logScale ? Math.log10(0.05) : 0;
  const yMax = logScale ? Math.max(2, Math.log10(maxV * 1.4)) : maxV * 1.15;
  const yScale = (v) => padT + innerH - ((valueFn(v) - yMin) / (yMax - yMin)) * innerH;
  const barW = innerW / data.length * 0.62;
  const stepX = innerW / data.length;

  // Determine "best" bar for highlighting.
  const bestIdx = (() => {
    let best = 0;
    for (let i = 1; i < data.length; i++) {
      if (betterDir === "higher" && data[i][valueKey] > data[best][valueKey]) best = i;
      if (betterDir === "lower" && data[i][valueKey] < data[best][valueKey]) best = i;
    }
    return best;
  })();

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" preserveAspectRatio="xMidYMid meet">
      {/* Axis */}
      {!hideAxis && (
        <>
          <line x1={padL} y1={padT} x2={padL} y2={padT + innerH} stroke="var(--ink-3)" strokeWidth="0.6" />
          <line x1={padL} y1={padT + innerH} x2={padL + innerW} y2={padT + innerH} stroke="var(--ink-3)" strokeWidth="0.6" />
          {/* Gridlines */}
          {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
            const v = logScale ? (yMin + (yMax - yMin) * p) : (yMax * p);
            const y = yScale(logScale ? Math.pow(10, v) : v);
            const lbl = logScale ? Math.pow(10, v).toFixed(v < 0 ? 2 : 1) : (yMax * p).toFixed(yMax < 5 ? 2 : yMax < 20 ? 1 : 0);
            return (
              <g key={i}>
                <line x1={padL} y1={y} x2={padL + innerW} y2={y}
                      stroke="var(--rule)" strokeWidth="0.6" strokeDasharray="3 4" />
                <text x={padL - 8} y={y + 3} textAnchor="end"
                      fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)">
                  {lbl}
                </text>
              </g>
            );
          })}
          <text x={padL - 80} y={padT + innerH / 2} fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)"
                transform={`rotate(-90, ${padL - 80}, ${padT + innerH / 2})`} textAnchor="middle">{unit}</text>
        </>
      )}

      {/* Bars */}
      {data.map((d, i) => {
        const x = padL + stepX * i + (stepX - barW) / 2;
        const y = yScale(d[valueKey]);
        const yBase = yScale(0);
        const h = Math.max(1, yBase - y);
        const m = window.METHOD_BY_ID[d.method];
        const isBest = i === bestIdx;
        return (
          <g key={d.method}>
            <rect x={x} y={y} width={barW} height={h}
                  fill={m.raw}
                  opacity={m.dotted ? 0 : (isBest ? 0.95 : 0.78)}
                  stroke={m.dotted ? m.raw : "none"}
                  strokeDasharray={m.dotted ? "3 3" : "0"}
                  strokeWidth={m.dotted ? 1.4 : 0}
            />
            {isBest && (
              <rect x={x - 2} y={y - 2} width={barW + 4} height={h + 2}
                    fill="none" stroke="oklch(0.50 0.150 245)" strokeWidth="1.2" />
            )}
            {/* CI95 error bar */}
            {d[ciKey] != null && d[ciKey] > 0 && (
              <g stroke="var(--ink-1)" strokeWidth="1.1" opacity="0.7">
                <line x1={x + barW / 2} y1={yScale(d[valueKey] - d[ciKey])} x2={x + barW / 2} y2={yScale(d[valueKey] + d[ciKey])} />
                <line x1={x + barW / 2 - 6} y1={yScale(d[valueKey] - d[ciKey])} x2={x + barW / 2 + 6} y2={yScale(d[valueKey] - d[ciKey])} />
                <line x1={x + barW / 2 - 6} y1={yScale(d[valueKey] + d[ciKey])} x2={x + barW / 2 + 6} y2={yScale(d[valueKey] + d[ciKey])} />
              </g>
            )}
            {/* Value label */}
            <text x={x + barW / 2} y={y - 6} textAnchor="middle"
                  fontFamily="IBM Plex Mono" fontSize="10"
                  fill={isBest ? "oklch(0.50 0.150 245)" : "var(--ink-2)"}
                  fontWeight={isBest ? 600 : 400}>
              {d[valueKey] >= 100 ? d[valueKey].toFixed(0) : d[valueKey].toFixed(2)}
            </text>
            {/* X label */}
            <text x={x + barW / 2} y={padT + innerH + 16} textAnchor="middle"
                  fontFamily="IBM Plex Mono" fontSize="9.5" fill="var(--ink-3)">
              {m.short}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ----- Multi-metric radar
function RadarChart({ axes, series, size = 360 }) {
  const cx = size / 2, cy = size / 2;
  const R = size * 0.36;
  const N = axes.length;
  const angle = (i) => -Math.PI / 2 + (i * 2 * Math.PI) / N;
  const point = (i, v) => [cx + Math.cos(angle(i)) * R * v, cy + Math.sin(angle(i)) * R * v];
  const rings = [0.25, 0.5, 0.75, 1.0];
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="chart-svg" preserveAspectRatio="xMidYMid meet">
      {/* Rings */}
      {rings.map((r) => (
        <polygon key={r}
          points={axes.map((_, i) => {
            const [x, y] = point(i, r);
            return `${x},${y}`;
          }).join(" ")}
          fill="none" stroke="var(--rule)" strokeWidth="0.6" />
      ))}
      {/* Axes */}
      {axes.map((a, i) => {
        const [x, y] = point(i, 1);
        const [lx, ly] = point(i, 1.13);
        return (
          <g key={a.label}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke="var(--rule)" strokeWidth="0.6" />
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
                  fontFamily="IBM Plex Mono" fontSize="9.5" fill="var(--ink-2)">
              {a.label}
            </text>
          </g>
        );
      })}
      {/* Series */}
      {series.map((s) => {
        const m = window.METHOD_BY_ID[s.method];
        const pts = s.values.map((v, i) => point(i, v));
        return (
          <g key={s.method}>
            <polygon points={pts.map((p) => p.join(",")).join(" ")}
                     fill={m.raw} fillOpacity={m.ours ? 0.18 : 0.06}
                     stroke={m.raw} strokeWidth={m.ours ? 1.6 : 0.9}
                     strokeDasharray={m.dotted ? "3 3" : "0"} />
            {pts.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={m.ours ? 2.4 : 1.6} fill={m.raw} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

// ----- Line chart for topology generalization & training curves
function LineChart({ series, xAxis, yAxis, height = 260, xLabel = "", yLabel = "", referenceLines = [] }) {
  const W = 720, H = height, padL = 70, padR = 24, padT = 20, padB = 50;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const allY = series.flatMap((s) => s.points.map((p) => p[1]));
  const yMin = Math.min(...allY, ...referenceLines.map((r) => r.y), 0);
  const yMaxRaw = Math.max(...allY, ...referenceLines.map((r) => r.y));
  const yMax = yMaxRaw + (yMaxRaw - yMin) * 0.1;
  const xMin = xAxis[0], xMax = xAxis[1];
  const sx = (x) => padL + ((x - xMin) / (xMax - xMin)) * innerW;
  const sy = (y) => padT + innerH - ((y - yMin) / (yMax - yMin)) * innerH;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" preserveAspectRatio="xMidYMid meet">
      <line x1={padL} y1={padT} x2={padL} y2={padT + innerH} stroke="var(--ink-3)" strokeWidth="0.6" />
      <line x1={padL} y1={padT + innerH} x2={padL + innerW} y2={padT + innerH} stroke="var(--ink-3)" strokeWidth="0.6" />
      {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
        const v = yMin + (yMax - yMin) * p;
        const y = sy(v);
        return (
          <g key={i}>
            <line x1={padL} y1={y} x2={padL + innerW} y2={y} stroke="var(--rule)" strokeDasharray="3 4" strokeWidth="0.6" />
            <text x={padL - 8} y={y + 3} textAnchor="end" fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)">{v.toFixed(1)}</text>
          </g>
        );
      })}
      {/* X ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
        const v = xMin + (xMax - xMin) * p;
        const x = sx(v);
        return (
          <g key={i}>
            <line x1={x} y1={padT + innerH} x2={x} y2={padT + innerH + 4} stroke="var(--ink-3)" strokeWidth="0.6" />
            <text x={x} y={padT + innerH + 16} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)">
              {Math.round(v)}
            </text>
          </g>
        );
      })}
      {/* Reference lines */}
      {referenceLines.map((r, i) => (
        <g key={i}>
          <line x1={padL} y1={sy(r.y)} x2={padL + innerW} y2={sy(r.y)} stroke="oklch(0.72 0.130 65)" strokeWidth="0.8" strokeDasharray="6 4" />
          {r.label && <text x={padL + innerW - 6} y={sy(r.y) - 4} textAnchor="end" fontFamily="IBM Plex Mono" fontSize="9.5" fill="oklch(0.55 0.130 65)">{r.label}</text>}
        </g>
      ))}
      {/* Series */}
      {series.map((s) => {
        const m = window.METHOD_BY_ID[s.method] || { raw: s.color || "var(--signal)", ours: false, dotted: false };
        const path = s.points.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${sx(x)} ${sy(y)}`).join(" ");
        return (
          <g key={s.method || s.label}>
            <path d={path} fill="none" stroke={m.raw} strokeWidth={m.ours ? 2.2 : 1.4}
                  strokeDasharray={m.dotted ? "4 3" : "0"} />
            {s.points.map(([x, y], i) => (
              <circle key={i} cx={sx(x)} cy={sy(y)} r={m.ours ? 2.6 : 1.8} fill={m.raw} />
            ))}
          </g>
        );
      })}
      <text x={padL + innerW / 2} y={H - 8} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)">{xLabel}</text>
      <text x={16} y={padT + innerH / 2} fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)"
            transform={`rotate(-90, 16, ${padT + innerH / 2})`} textAnchor="middle">{yLabel}</text>
    </svg>
  );
}

// ----- Histogram (CIO utilization)
function Histogram({ buckets, bounds, width = 720, height = 260 }) {
  const W = width, H = height, padL = 60, padR = 24, padT = 16, padB = 46;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const maxCount = Math.max(...buckets.map((b) => b.count));
  const stepX = innerW / buckets.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" preserveAspectRatio="xMidYMid meet">
      <line x1={padL} y1={padT} x2={padL} y2={padT + innerH} stroke="var(--ink-3)" strokeWidth="0.6" />
      <line x1={padL} y1={padT + innerH} x2={padL + innerW} y2={padT + innerH} stroke="var(--ink-3)" strokeWidth="0.6" />
      {buckets.map((b, i) => {
        const x = padL + stepX * i + stepX * 0.1;
        const h = (b.count / maxCount) * innerH;
        const y = padT + innerH - h;
        return (
          <g key={i}>
            <rect x={x} y={y} width={stepX * 0.8} height={h} fill="oklch(0.50 0.150 245)" opacity="0.78" />
            {(i % 2 === 0) && (
              <text x={x + stepX * 0.4} y={padT + innerH + 14} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="9" fill="var(--ink-3)">{b.label}</text>
            )}
          </g>
        );
      })}
      {/* Bounds */}
      {bounds.map((b, i) => {
        const x = padL + ((b.value - buckets[0].min) / (buckets[buckets.length - 1].max - buckets[0].min)) * innerW;
        return (
          <g key={i}>
            <line x1={x} y1={padT} x2={x} y2={padT + innerH} stroke="oklch(0.65 0.150 25)" strokeWidth="1.2" strokeDasharray="5 4" />
            <text x={x + 4} y={padT + 14} fontFamily="IBM Plex Mono" fontSize="10" fill="oklch(0.55 0.150 25)">{b.label}</text>
          </g>
        );
      })}
      <text x={padL + innerW / 2} y={H - 6} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize="10" fill="var(--ink-3)">CIO update magnitude (dB)</text>
    </svg>
  );
}

// =========================================================================

function ResultsDashboard() {
  const throughputData = window.RESULTS.throughput;
  const pingpongData = window.RESULTS.pingpong;
  const jainData = window.RESULTS.jain;
  const outageData = window.RESULTS.outage;
  const radarData = window.RESULTS.radar;
  const generalizationData = window.RESULTS.generalization;
  const cioBuckets = window.RESULTS.cio_histogram;
  const trainingCurves = window.RESULTS.training_curves;

  return (
    <section className="section" id="results">
      <div className="section-inner">
        <Eyebrow num="06">Results dashboard</Eyebrow>
        <h2 className="section-title">Eight evaluation lenses across throughput, fairness, and stability.</h2>
        <p className="section-lede">
          The agent and all six baselines are evaluated across the eleven scenarios at 20 random seeds.
          Each card below renders a different lens on the same evaluation: aggregate KPIs, per-scenario
          breakdowns, SON safety behaviour, and training dynamics.
        </p>

        <PlaceholderBanner />

        <div className="results-grid">
          <ResultCard
            num="06.1"
            title="Throughput across all scenarios"
            sub="Mean ± CI95 · 20 seeds · 11 scenarios"
            takeaway="GNN-DQN with SON delivers the highest aggregate throughput while preserving fairness."
            csvHint="results/runs/multiscenario_ue/evaluation/throughput.csv"
          >
            <BarChart data={throughputData} title="Throughput" unit="Mbps / UE" betterDir="higher" />
          </ResultCard>

          <ResultCard
            num="06.2"
            title="Ping-pong rate"
            sub="Log scale · lower is better"
            takeaway="SON's TTT scaling and rate-limited CIO updates drive ping-pong below 1%."
            csvHint="results/runs/multiscenario_ue/evaluation/pingpong.csv"
          >
            <BarChart data={pingpongData} title="Ping-pong" unit="% of handovers" betterDir="lower" logScale />
          </ResultCard>

          <ResultCard
            num="06.3"
            title="Multi-metric radar"
            sub="Normalized 0–1 per axis · larger area = better"
            takeaway="GNN-DQN+SON dominates on throughput, fairness, and stability; matches A3/TTT on outage."
            csvHint="results/runs/multiscenario_ue/evaluation/radar.csv"
          >
            <RadarChart
              axes={[
                { label: "Throughput" },
                { label: "P5 Throughput" },
                { label: "Jain Fairness" },
                { label: "Stability" },
                { label: "Coverage" },
                { label: "Efficiency" },
              ]}
              series={radarData}
            />
            <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 14, justifyContent: "center" }}>
              {radarData.map((s) => <MethodChip key={s.method} id={s.method} />)}
            </div>
          </ResultCard>

          <ResultCard
            num="06.4"
            title="Jain fairness index"
            sub="0 (one UE hogs) → 1 (perfectly fair)"
            takeaway="Even peak-hour deployments stay above 0.85 fairness."
            csvHint="results/runs/multiscenario_ue/evaluation/jain.csv"
          >
            <BarChart data={jainData} title="Jain" unit="Jain index" betterDir="higher" />
          </ResultCard>

          <ResultCard
            num="06.5"
            title="Outage probability"
            sub="Fraction of UEs below −100 dBm RSRP"
            takeaway="The SON layer never sacrifices outage to gain throughput."
            csvHint="results/runs/multiscenario_ue/evaluation/outage.csv"
          >
            <BarChart data={outageData} title="Outage" unit="% of UEs" betterDir="lower" />
          </ResultCard>

          <ResultCard
            num="06.6"
            title="Topology generalization · cells in deployment"
            sub="Trained on 3-cell synthetic · evaluated on N-cell unseen layouts"
            takeaway="GNN-DQN holds throughput flat to 25 cells; flat-MLP DQN collapses past 7."
            csvHint="results/runs/multiscenario_ue/evaluation/generalization.csv"
          >
            <LineChart
              series={generalizationData}
              xAxis={[3, 25]}
              yAxis={[0, 14]}
              xLabel="Number of cells in unseen deployment"
              yLabel="Throughput (Mbps / UE)"
              referenceLines={[{ y: 5.5, label: "A3/TTT reference (trained topology)" }]}
            />
          </ResultCard>

          <ResultCard
            num="06.7"
            title="SON CIO utilization"
            sub="Distribution of bounded CIO updates · ±6 dB clamp"
            takeaway="98.4% of issued updates land inside ±4 dB. The hard clamp is rarely touched."
            csvHint="results/runs/multiscenario_ue/evaluation/cio_histogram.csv"
          >
            <Histogram buckets={cioBuckets} bounds={[{ value: -6, label: "−6 dB (clamp)" }, { value: 6, label: "+6 dB (clamp)" }]} />
          </ResultCard>

          <ResultCard
            num="06.8"
            title="Training dynamics"
            sub="Reward + loss · 300 episodes · ε-greedy decay"
            takeaway="Stable convergence; loss plateaus by episode 220, reward saturates ~episode 260."
            csvHint="results/runs/multiscenario_ue/checkpoints/training_log.csv"
          >
            <LineChart
              series={trainingCurves}
              xAxis={[0, 300]}
              yAxis={[0, 1]}
              xLabel="Episode"
              yLabel="Normalized value"
            />
            <div style={{ display: "flex", gap: 18, marginTop: 10, justifyContent: "center", fontFamily: "var(--mono)", fontSize: 11 }}>
              <span><span style={{ color: "oklch(0.50 0.150 245)" }}>━</span> Reward (normalized)</span>
              <span><span style={{ color: "oklch(0.65 0.150 25)" }}>━</span> TD loss</span>
              <span><span style={{ color: "oklch(0.55 0.05 250)" }}>━ ━</span> ε</span>
            </div>
          </ResultCard>
        </div>

        <div style={{ marginTop: 48 }}>
          <h3 style={{ fontFamily: "var(--serif)", fontSize: 22, fontWeight: 500, marginBottom: 6 }}>Per-scenario table</h3>
          <p style={{ fontSize: 13.5, color: "var(--ink-3)", margin: "0 0 18px" }}>
            Throughput (Mbps / UE), mean across 20 seeds. Best per row in <b style={{ color: "var(--signal)" }}>blue</b>.
          </p>
          <ResultsTable />
        </div>
      </div>
    </section>
  );
}

function ResultCard({ num, title, sub, takeaway, csvHint, children }) {
  return (
    <article className="result-card card">
      <header className="result-card-head">
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-3)" }}>{num}</span>
          <h3 className="result-card-title">{title}</h3>
        </div>
        <div className="result-card-sub">{sub}</div>
      </header>
      <div className="result-card-body">
        {children}
      </div>
      <footer className="result-card-foot">
        <span className="result-card-take">↳ {takeaway}</span>
        {csvHint && <span className="result-card-csv">{csvHint}</span>}
      </footer>
    </article>
  );
}

function ResultsTable() {
  const rows = window.RESULTS.per_scenario;
  const methods = window.METHODS.filter((m) => m.id !== "no_handover");
  return (
    <div className="results-table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            <th>Scenario</th>
            {methods.map((m) => (
              <th key={m.id}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                  <span className="swatch" style={{
                    background: m.dotted ? "transparent" : m.raw,
                    border: m.dotted ? `1.5px dotted ${m.raw}` : "none",
                  }} />
                  <span style={{ fontSize: 10.5 }}>{m.short}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const values = methods.map((m) => r.values[m.id]);
            const best = Math.max(...values.filter((v) => v != null));
            return (
              <tr key={r.scenario}>
                <td><b>{r.scenario}</b></td>
                {methods.map((m) => {
                  const v = r.values[m.id];
                  const isBest = v === best;
                  return (
                    <td key={m.id} className={isBest ? "is-best" : ""}>
                      {v != null ? v.toFixed(2) : "—"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

window.ResultsDashboard = ResultsDashboard;
