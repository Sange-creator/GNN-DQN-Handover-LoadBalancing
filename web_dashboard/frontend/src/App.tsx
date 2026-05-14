import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Polyline, Circle, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell as RCell,
} from 'recharts';
import {
  Activity, ShieldCheck, Zap, Play, Pause, RotateCcw,
  TrendingUp, Layers, Radio, GitCompare, BarChart3,
} from 'lucide-react';
import {
  fetchTopology, fetchTraining, fetchComparison,
  fetchSimulation, fetchScenarios, fetchModelInfo,
} from './api';

// Fix Leaflet icons
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

/* ── helpers ─────────────────────────────────────────────── */
const fmt = (v: number, d = 2) => v.toFixed(d);
const pct = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(1) + '%';

/* metric columns for comparison table */
const COLS: { key: string; label: string; unit: string; higherBetter: boolean }[] = [
  { key: 'avg_ue_throughput_mbps', label: 'Avg Throughput', unit: 'Mbps', higherBetter: true },
  { key: 'p5_ue_throughput_mbps', label: 'P5 Throughput', unit: 'Mbps', higherBetter: true },
  { key: 'pingpong_rate', label: 'Ping-pong', unit: '%', higherBetter: false },
  { key: 'jain_load_fairness', label: 'Jain Fairness', unit: '', higherBetter: true },
  { key: 'load_std', label: 'Load Std', unit: '', higherBetter: false },
  { key: 'handovers_per_1000_decisions', label: 'HO/1000', unit: '', higherBetter: false },
  { key: 'outage_rate', label: 'Outage', unit: '%', higherBetter: false },
];

const METHOD_LABELS: Record<string, string> = {
  no_handover: 'No Handover',
  random_valid: 'Random Valid',
  strongest_rsrp: 'Strongest RSRP',
  a3_ttt: 'A3 + TTT (3GPP)',
  load_aware: 'Load-Aware',
  gnn_dqn: 'GNN-DQN (Direct)',
  son_gnn_dqn: 'SON-GNN-DQN ★',
  son_gnn_dqn_true_prb: 'SON-GNN-DQN (PRB)',
};

/* ── App ─────────────────────────────────────────────────── */
export default function App() {
  const [topology, setTopology] = useState<any>(null);
  const [training, setTraining] = useState<any>(null);
  const [comparison, setComparison] = useState<any>(null);
  const [simulation, setSimulation] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [selectedScenario, setSelectedScenario] = useState('dense_urban');
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  /* initial load */
  useEffect(() => {
    Promise.all([
      fetchTopology().then(setTopology),
      fetchTraining().then(setTraining),
      fetchComparison('dense_urban').then(setComparison),
      fetchSimulation().then(setSimulation),
      fetchModelInfo().then(setModelInfo),
      fetchScenarios().then(d => setScenarios(d.scenarios)),
    ]).catch(console.error);
  }, []);

  /* scenario change */
  useEffect(() => {
    fetchComparison(selectedScenario).then(setComparison).catch(console.error);
  }, [selectedScenario]);

  /* simulation player */
  useEffect(() => {
    if (!playing || !simulation?.steps?.length) return;
    if (step >= simulation.steps.length - 1) { setPlaying(false); return; }
    const id = setInterval(() => setStep(s => s + 1), 400);
    return () => clearInterval(id);
  }, [playing, step, simulation]);

  if (!topology || !training || !comparison) {
    return <div className="loading"><div className="spinner" /> Loading dashboard…</div>;
  }

  const episodes = training.episodes;
  const methods = comparison.methods || [];
  const simSteps = simulation?.steps || [];
  const curSim = simSteps[step] || null;

  /* find best values per column for highlighting */
  const bestVals: Record<string, number> = {};
  COLS.forEach(col => {
    const vals = methods.map((m: any) => Number(m[col.key]) || 0);
    bestVals[col.key] = col.higherBetter ? Math.max(...vals) : Math.min(...vals);
  });

  return (
    <div id="root">
      {/* ── Header ──────────────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <h1 className="title">SON-GNN-DQN Handover Optimizer</h1>
          <span className="badge badge-accent">Pokhara Valley</span>
          <span className="badge badge-success">{training.source === 'mock' ? 'Demo Data' : training.source}</span>
        </div>
        <div className="header-controls">
          <button className="control-btn" onClick={() => { setStep(0); setPlaying(false); }}><RotateCcw size={16} /></button>
          <button className="control-btn" onClick={() => setPlaying(!playing)}>
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <span className="step-indicator">Step {step}/{Math.max(simSteps.length - 1, 0)}</span>
        </div>
      </header>

      <div className="dashboard">
        {/* ── Stat Cards ─────────────────────────────── */}
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-label">Network Cells</div>
            <div className="stat-value">{topology.nodes?.length || 0}</div>
            <div className="stat-sub">Pokhara Valley LTE</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Training Episodes</div>
            <div className="stat-value">{training.summary?.total_episodes || 0}</div>
            <div className="stat-sub">Best ep. {training.summary?.best_episode}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Best Method</div>
            <div className="stat-value" style={{ fontSize: '1.1rem', color: '#10b981' }}>SON-GNN-DQN</div>
            <div className="stat-sub">Score {fmt(training.summary?.best_score || 0, 1)}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Feature Profile</div>
            <div className="stat-value" style={{ fontSize: '1.1rem' }}>UE-Only</div>
            <div className="stat-sub">RSRP + RSRQ + serving</div>
          </div>
        </div>

        {/* ── Row: Map + Architecture/Info ────────────── */}
        <div className="row-2col">
          {/* Map */}
          <div className="card">
            <div className="card-header">
              <Radio size={16} style={{ color: '#6366f1' }} />
              <h2>Network Topology — {topology.region?.name}</h2>
              <span className="source-badge">OpenCellID</span>
            </div>
            <div className="map-container">
              {topology.region && (
                <MapContainer center={topology.region.center} zoom={13} scrollWheelZoom>
                  <TileLayer
                    attribution='&copy; <a href="https://carto.com">CARTO</a>'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                  />
                  {/* Adjacency edges */}
                  {topology.edges?.map((e: any, i: number) => {
                    const s = topology.nodes.find((n: any) => n.id === e.source);
                    const t = topology.nodes.find((n: any) => n.id === e.target);
                    return s && t ? (
                      <Polyline key={i} positions={[[s.lat, s.lon], [t.lat, t.lon]]}
                        color="#334155" weight={1} opacity={0.4} />
                    ) : null;
                  })}
                  {/* Cell towers */}
                  {topology.nodes?.map((n: any) => (
                    <Circle key={n.id} center={[n.lat, n.lon]} radius={80}
                      pathOptions={{
                        color: curSim?.serving_cell === n.id ? '#6366f1' : '#475569',
                        fillColor: curSim?.serving_cell === n.id ? '#6366f1' : '#1e293b',
                        fillOpacity: 0.7,
                      }}>
                      <Popup>Cell {n.id}<br />Radio: {n.radio}</Popup>
                    </Circle>
                  ))}
                  {/* UE trajectory */}
                  {simSteps.length > 0 && (
                    <Polyline
                      positions={simSteps.slice(0, step + 1).map((s: any) => [s.lat, s.lon])}
                      color="#f59e0b" weight={3} dashArray="6,6"
                    />
                  )}
                  {/* UE marker */}
                  {curSim && (
                    <Marker position={[curSim.lat, curSim.lon]}>
                      <Popup>
                        RSRP: {fmt(curSim.serving_rsrp, 1)} dBm<br />
                        Throughput: {fmt(curSim.throughput)} Mbps<br />
                        SON: {curSim.son_action} | A3: {curSim.a3_action}
                      </Popup>
                    </Marker>
                  )}
                </MapContainer>
              )}
            </div>
          </div>

          {/* Right panel: Architecture + SON + Q-values */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Architecture */}
            <div className="card">
              <div className="card-header">
                <Layers size={16} style={{ color: '#818cf8' }} />
                <h2>Model Architecture</h2>
              </div>
              <div className="card-body">
                <div className="arch-layer arch-input">UE Features (RSRP · RSRQ · Serving) → 11 dim</div>
                <div className="arch-arrow">▼</div>
                <div className="arch-layer arch-gnn">3× GCN Layers + LayerNorm (128 → 128 → 64)</div>
                <div className="arch-arrow">▼</div>
                <div className="arch-layer arch-head">Dueling Head: V(s) + A(s,a) → Q-values</div>
                <div className="arch-arrow">▼</div>
                <div className="arch-layer arch-son">SON Safety Layer: CIO ±6 dB · TTT 2-8 steps</div>
              </div>
            </div>

            {/* Q-values */}
            {curSim && (
              <div className="card" style={{ flex: 1 }}>
                <div className="card-header">
                  <Zap size={16} style={{ color: '#f59e0b' }} />
                  <h2>GNN Q-Values (Step {step})</h2>
                  <span className="source-badge">{curSim.son_action === 'handover' ? '🔄 HO' : '📶 Stay'}</span>
                </div>
                <div className="card-body">
                  {Object.entries(curSim.q_values).map(([id, val]: [string, any]) => {
                    const isServing = curSim.serving_cell === Number(id);
                    const color = isServing ? '#6366f1' : val > 0.6 ? '#10b981' : '#475569';
                    return (
                      <div key={id} className="qval-row">
                        <span className="qval-label" style={isServing ? { color: '#6366f1', fontWeight: 600 } : {}}>
                          {isServing ? '★ ' : ''}C{id.slice(-4)}
                        </span>
                        <div className="qval-bar-bg">
                          <div className="qval-bar" style={{ width: `${val * 100}%`, background: color }} />
                        </div>
                        <span className="qval-num">{(val as number).toFixed(2)}</span>
                      </div>
                    );
                  })}
                  <div style={{ marginTop: 10, fontSize: '0.72rem', color: '#64748b' }}>
                    A3 decision: <b style={{ color: curSim.a3_action === 'handover' ? '#ef4444' : '#94a3b8' }}>{curSim.a3_action}</b>
                    {' · '}
                    SON-GNN-DQN: <b style={{ color: curSim.son_action === 'handover' ? '#10b981' : '#94a3b8' }}>{curSim.son_action}</b>
                    {' · CIO: +'}
                    {curSim.son_cio_db} dB
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Training Progress ──────────────────────── */}
        <div className="card">
          <div className="card-header">
            <TrendingUp size={16} style={{ color: '#10b981' }} />
            <h2>Training Progress</h2>
            <span className="source-badge">{training.source}</span>
          </div>
          <div className="card-body">
            <div className="charts-grid">
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 4 }}>Episode Reward</div>
                <div className="chart-mini">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={episodes}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="episode" stroke="#475569" fontSize={10} />
                      <YAxis stroke="#475569" fontSize={10} />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="episode_reward" stroke="#6366f1" dot={false} strokeWidth={2} name="Reward" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 4 }}>Average Throughput (Mbps)</div>
                <div className="chart-mini">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={episodes}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="episode" stroke="#475569" fontSize={10} />
                      <YAxis stroke="#475569" fontSize={10} />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="avg_ue_throughput_mbps" stroke="#10b981" dot={false} strokeWidth={2} name="Avg Mbps" />
                      <Line type="monotone" dataKey="p5_ue_throughput_mbps" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="P5 Mbps" strokeDasharray="4 4" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 4 }}>Training Loss</div>
                <div className="chart-mini">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={episodes}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="episode" stroke="#475569" fontSize={10} />
                      <YAxis stroke="#475569" fontSize={10} />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="loss" stroke="#ef4444" dot={false} strokeWidth={2} name="Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: 4 }}>Exploration (ε) &amp; Ping-pong Rate</div>
                <div className="chart-mini">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={episodes}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="episode" stroke="#475569" fontSize={10} />
                      <YAxis stroke="#475569" fontSize={10} domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: '0.75rem' }} />
                      <Line type="monotone" dataKey="epsilon" stroke="#818cf8" dot={false} strokeWidth={1.5} name="Epsilon" />
                      <Line type="monotone" dataKey="pingpong_rate" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="Ping-pong" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Policy Comparison ──────────────────────── */}
        <div className="card comparison-section">
          <div className="card-header">
            <GitCompare size={16} style={{ color: '#3b82f6' }} />
            <h2>Policy Comparison — {selectedScenario.replace(/_/g, ' ')}</h2>
            <select className="scenario-select" value={selectedScenario}
              onChange={e => setSelectedScenario(e.target.value)}>
              {scenarios.map(s => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
            </select>
            <span className="source-badge">{comparison.source}</span>
          </div>
          <div className="table-wrapper">
            <table className="comparison-table">
              <thead>
                <tr>
                  <th>Method</th>
                  {COLS.map(c => <th key={c.key}>{c.label}{c.unit ? ` (${c.unit})` : ''}</th>)}
                </tr>
              </thead>
              <tbody>
                {methods.map((m: any) => {
                  const isOurs = m.method === 'son_gnn_dqn';
                  const isA3 = m.method === 'a3_ttt';
                  return (
                    <tr key={m.method} className={isOurs ? 'highlight' : isA3 ? 'baseline' : ''}>
                      <td className="method-name">
                        {METHOD_LABELS[m.method] || m.method}
                      </td>
                      {COLS.map(col => {
                        const val = Number(m[col.key]) || 0;
                        const isBest = Math.abs(val - bestVals[col.key]) < 0.001;
                        const display = col.key.includes('rate') ? fmt(val * 100, 1) : fmt(val, 2);
                        return (
                          <td key={col.key} className={isBest ? 'best-value' : ''}>
                            {display}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {/* Improvement banner */}
          {comparison.improvement && Object.keys(comparison.improvement).length > 0 && (
            <div className="improvement-banner">
              <div className="improvement-item">
                <Activity size={14} />
                <span>vs A3/TTT Avg Throughput:</span>
                <span className="improvement-value">{pct(comparison.improvement.vs_a3_avg_pct)}</span>
              </div>
              <div className="improvement-item">
                <ShieldCheck size={14} />
                <span>P5 (Worst-case):</span>
                <span className="improvement-value">{pct(comparison.improvement.vs_a3_p5_pct)}</span>
              </div>
              <div className="improvement-item">
                <BarChart3 size={14} />
                <span>Load Balance:</span>
                <span className="improvement-value">{pct(comparison.improvement.vs_a3_load_std_reduction_pct)} std reduction</span>
              </div>
            </div>
          )}
        </div>

        {/* ── Throughput Comparison Bar Chart ─────────── */}
        <div className="card">
          <div className="card-header">
            <BarChart3 size={16} style={{ color: '#f59e0b' }} />
            <h2>Throughput Comparison — {selectedScenario.replace(/_/g, ' ')}</h2>
          </div>
          <div className="card-body" style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={methods.map((m: any) => ({
                name: METHOD_LABELS[m.method]?.replace(' ★', '') || m.method,
                avg: Number(m.avg_ue_throughput_mbps) || 0,
                p5: Number(m.p5_ue_throughput_mbps) || 0,
                isOurs: m.method === 'son_gnn_dqn',
                isA3: m.method === 'a3_ttt',
              }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#475569" fontSize={10} angle={-20} textAnchor="end" height={60} />
                <YAxis stroke="#475569" fontSize={10} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', fontSize: '0.75rem' }} />
                <Bar dataKey="avg" name="Avg Throughput">
                  {methods.map((m: any, i: number) => (
                    <RCell key={i} fill={m.method === 'son_gnn_dqn' ? '#10b981' : m.method === 'a3_ttt' ? '#3b82f6' : '#475569'} />
                  ))}
                </Bar>
                <Bar dataKey="p5" name="P5 Throughput">
                  {methods.map((m: any, i: number) => (
                    <RCell key={i} fill={m.method === 'son_gnn_dqn' ? 'rgba(16,185,129,0.4)' : m.method === 'a3_ttt' ? 'rgba(59,130,246,0.4)' : 'rgba(71,85,105,0.4)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
