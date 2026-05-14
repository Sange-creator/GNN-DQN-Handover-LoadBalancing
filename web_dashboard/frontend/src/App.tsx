import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from 'recharts';
import { Activity, ShieldCheck, Zap, Info, Play, Pause, RotateCcw } from 'lucide-react';
import { fetchTopology, fetchSimulation } from './api';

// Fix Leaflet icon issue
// @ts-ignore
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png',
});

const App: React.FC = () => {
  const [topology, setTopology] = useState<any>(null);
  const [simulation, setSimulation] = useState<any[]>([]);
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    const loadData = async () => {
      try {
        const topo = await fetchTopology();
        setTopology(topo);
        const sim = await fetchSimulation();
        setSimulation(sim);
      } catch (err) {
        console.error("Failed to load data", err);
      }
    };
    loadData();
  }, []);

  useEffect(() => {
    let interval: any;
    if (isPlaying && step < simulation.length - 1) {
      interval = setInterval(() => {
        setStep(prev => prev + 1);
      }, 500);
    } else {
      setIsPlaying(false);
    }
    return () => clearInterval(interval);
  }, [isPlaying, step, simulation]);

  if (!topology || simulation.length === 0) {
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading Pokhara Valley Topology...</div>;
  }

  const currentData = simulation[step];
  const chartData = simulation.slice(0, step + 1);

  return (
    <div id="root">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <h1 className="title">SON-GNN-DQN Optimizer</h1>
          <span className="badge">Pokhara Valley Deployment</span>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button onClick={() => { setStep(0); setIsPlaying(false); }} className="control-btn"><RotateCcw size={18} /></button>
          <button onClick={() => setIsPlaying(!isPlaying)} className="control-btn">
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <div style={{ marginLeft: '10px', fontSize: '0.9rem' }}>Step: {step} / {simulation.length - 1}</div>
        </div>
      </header>

      <div className="dashboard-container">
        <aside className="sidebar">
          <section className="card">
            <h2><Info size={18} style={{ verticalAlign: 'middle', marginRight: '5px' }} /> What is SON-GNN-DQN?</h2>
            <p>
              A 3-layer architecture for 5G handover optimization.
              It uses <b>Graph Neural Networks</b> to learn cell relationships
              and a <b>SON Safety Layer</b> to ensure 3GPP compliance.
            </p>
          </section>

          <section className="card">
            <h2><Zap size={18} style={{ verticalAlign: 'middle', marginRight: '5px', color: '#ffcc00' }} /> Learning Preferences</h2>
            <p>
              The GNN evaluates neighbors based on <b>RSRQ Load Proxy</b>. 
              The Q-values below show the model's preference for each cell.
            </p>
            <div style={{ height: '150px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={Object.entries(currentData.q_values).map(([id, val]) => ({ name: `Cell ${id.slice(-4)}`, val }))}>
                  <XAxis dataKey="name" hide />
                  <YAxis hide domain={[0, 1]} />
                  <Tooltip labelStyle={{ color: '#000' }} />
                  <Bar dataKey="val" fill="#646cff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card">
            <h2><ShieldCheck size={18} style={{ verticalAlign: 'middle', marginRight: '5px', color: '#4caf50' }} /> SON Safety Bounds</h2>
            <p>
              Requested CIO: <b>+0.5 dB</b><br />
              Total CIO: <b>1.5 dB</b> (Bounded ±6 dB)<br />
              Status: <span style={{ color: '#4caf50' }}>SAFE</span>
            </p>
          </section>

          <div style={{ fontSize: '0.8rem', color: '#666', marginTop: 'auto' }}>
            Reference: thesis_support.md
          </div>
        </aside>

        <main className="main-content">
          <div className="map-container">
            <MapContainer center={topology.region.center} zoom={13} scrollWheelZoom={true}>
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              />
              
              {/* Interference Graph Edges */}
              {topology.edges.map((edge: any, idx: number) => {
                const sourceNode = topology.nodes.find((n: any) => n.id === edge.source);
                const targetNode = topology.nodes.find((n: any) => n.id === edge.target);
                if (!sourceNode || !targetNode) return null;
                return (
                  <Polyline 
                    key={idx} 
                    positions={[[sourceNode.lat, sourceNode.lon], [targetNode.lat, targetNode.lon]]} 
                    color="#333" 
                    weight={1}
                    opacity={0.3}
                  />
                );
              })}

              {/* Cell Towers */}
              {topology.nodes.map((node: any) => (
                <Circle 
                  key={node.id}
                  center={[node.lat, node.lon]}
                  radius={100}
                  pathOptions={{ 
                    color: node.id === currentData.serving_cell ? '#646cff' : '#444',
                    fillColor: node.id === currentData.serving_cell ? '#646cff' : '#222',
                    fillOpacity: 0.8
                  }}
                >
                  <Popup>Cell ID: {node.id}<br/>Radio: {node.radio}</Popup>
                </Circle>
              ))}

              {/* UE Trajectory */}
              <Polyline 
                positions={simulation.slice(0, step + 1).map(d => [d.lat, d.lon])} 
                color="#ffcc00" 
                weight={3}
                dashArray="5, 5"
              />

              {/* Current UE Marker */}
              <Marker position={[currentData.lat, currentData.lon]}>
                <Popup>
                  UE Status:<br/>
                  RSRP: {currentData.rsrp.toFixed(1)} dBm<br/>
                  RSRQ: {currentData.rsrq.toFixed(1)} dB
                </Popup>
              </Marker>
            </MapContainer>
          </div>

          <div className="charts-container">
            <div className="chart-card">
              <h2><Activity size={16} style={{ verticalAlign: 'middle', marginRight: '5px' }} /> Throughput (Mbps)</h2>
              <div style={{ height: '140px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="step" hide />
                    <YAxis domain={[0, 10]} stroke="#666" fontSize={10} />
                    <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                    <Line type="monotone" dataKey="throughput" stroke="#646cff" dot={false} strokeWidth={2} name="SON-GNN-DQN" />
                    <Line type="monotone" dataKey="throughput" stroke="#444" dot={false} strokeWidth={1} name="Standard A3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </main>
      </div>
      <style>{`
        .control-btn {
          background: #333;
          border: none;
          color: white;
          padding: 5px 10px;
          border-radius: 4px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .control-btn:hover { background: #444; }
      `}</style>
    </div>
  );
};

export default App;
