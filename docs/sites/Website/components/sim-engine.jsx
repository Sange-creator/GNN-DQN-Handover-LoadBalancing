// =========================================================================
// sim-engine.jsx — Stylized canvas simulator. Cells, UEs, handovers,
// per-method behaviour. NOT a real PyTorch run — visualisation that mirrors
// the policy contracts described in scenarios.py / policies.py.
// =========================================================================

class SimEngine {
  constructor(scenarioId, methodId, opts = {}) {
    this.scenarioId = scenarioId;
    this.methodId = methodId;
    this.seed = opts.seed || 42;
    this.rng = mulberry32(this.seed);
    this.kpiOnUpdate = opts.kpiOnUpdate || (() => {});
    this.reset();
  }

  reset() {
    const data = window.getScenarioPositions(this.scenarioId);
    if (!data) return;
    const s = data.scenario;
    this.scenario = s;
    this.bbox = data.bbox;
    this.landmarks = data.landmarks;
    this.isRealMap = data.isReal;
    this.centerLatLon = data.centerLatLon;
    this.boundsLatLon = data.boundsLatLon;

    // Cells
    this.cells = data.positions.map(([x, y], i) => ({
      id: i, x, y,
      lat: data.cellLatLons ? data.cellLatLons[i][0] : null,
      lon: data.cellLatLons ? data.cellLatLons[i][1] : null,
      load: 0,        // 0..1, fraction of capacity in use
      attached: 0,    // count
      cio: 0,         // dB offset (SON-controlled)
    }));

    // Capacity (UEs that can be served comfortably) — coarse model.
    this.capacityPerCell = Math.max(4, Math.round(s.ues / s.cells * 0.95));

    // UEs
    this.ues = [];
    const w = this.bbox.w || 1, h = this.bbox.h || 1;
    for (let i = 0; i < s.ues; i++) {
      const ue = {
        id: i,
        x: this.bbox.minX + this.rng() * w,
        y: this.bbox.minY + this.rng() * h,
        vx: 0, vy: 0,
        speed: lerp(s.speed_min, s.speed_max, this.rng()),
        cell: 0,
        prev_cell: -1,
        last_ho_t: -999,
        ttt_remaining: 0,
        candidate_cell: -1,
        ho_count: 0,
        pingpong: 0,
        trail: [],
      };
      this.assignVelocity(ue, s);
      ue.cell = this.bestCellByRsrp(ue);
      this.ues.push(ue);
    }

    // KPI accumulators
    this.t = 0;
    this.kpi = {
      thr_avg: 0, thr_p5: 0, jain: 0,
      ho_per_min: 0, pingpong_pct: 0, outage_pct: 0,
    };
    this.handoverArcs = [];
    this.eventFlash = []; // recent events for visual flicker
    this.lastKpiComputeT = -999;
    this.recompute();
    this.computeKPIs([]);
  }

  assignVelocity(ue, s) {
    if (s.mobility === "highway") {
      // x-axis dominant motion
      const dir = this.rng() > 0.5 ? 1 : -1;
      const angle = (this.rng() - 0.5) * 0.2; // small drift
      ue.vx = Math.cos(angle) * ue.speed * dir;
      ue.vy = Math.sin(angle) * ue.speed * dir;
    } else if (s.mobility === "event") {
      // Mostly static, gentle wandering
      const a = this.rng() * Math.PI * 2;
      ue.vx = Math.cos(a) * ue.speed * 0.3;
      ue.vy = Math.sin(a) * ue.speed * 0.3;
    } else {
      const a = this.rng() * Math.PI * 2;
      ue.vx = Math.cos(a) * ue.speed;
      ue.vy = Math.sin(a) * ue.speed;
    }
  }

  rsrpAt(ue, cell) {
    const dx = ue.x - cell.x, dy = ue.y - cell.y;
    const dist = Math.max(20, Math.sqrt(dx * dx + dy * dy));
    // Simplified path loss (not calibrated absolute, used as ranking).
    // dBm-like scale: -50 at 10m, drops with log distance.
    const pl = 32.4 + 21 * Math.log10(dist);
    const rsrp = -30 - pl + cell.cio; // higher = better
    return rsrp;
  }

  bestCellByRsrp(ue) {
    let best = 0, bestV = -Infinity;
    for (let i = 0; i < this.cells.length; i++) {
      const v = this.rsrpAt(ue, this.cells[i]);
      if (v > bestV) { bestV = v; best = i; }
    }
    return best;
  }

  // ---- Method behaviours ----
  // Each method returns the desired target cell for a UE given current state,
  // or -1 to keep current attachment.
  decideTarget(ue) {
    const m = this.methodId;
    if (m === "no_handover") return -1;
    if (m === "random_valid") {
      // 1.5% chance per step to randomly switch.
      if (this.rng() < 0.015) {
        const cand = Math.floor(this.rng() * this.cells.length);
        return cand !== ue.cell ? cand : -1;
      }
      return -1;
    }
    if (m === "strongest_rsrp") {
      const best = this.bestCellByRsrp(ue);
      const margin = this.rsrpAt(ue, this.cells[best]) - this.rsrpAt(ue, this.cells[ue.cell]);
      // No hysteresis → switches frequently when close.
      if (best !== ue.cell && margin > 0.5) return best;
      return -1;
    }
    if (m === "a3_ttt") {
      // A3 with hysteresis 2 dB and TTT 320 ms ≈ 4 frames at our rate.
      const best = this.bestCellByRsrp(ue);
      const margin = this.rsrpAt(ue, this.cells[best]) - this.rsrpAt(ue, this.cells[ue.cell]);
      if (best !== ue.cell && margin > 2) {
        if (ue.candidate_cell !== best) {
          ue.candidate_cell = best;
          ue.ttt_remaining = 4;
          return -1;
        }
        ue.ttt_remaining -= 1;
        if (ue.ttt_remaining <= 0) return best;
      } else {
        ue.candidate_cell = -1;
        ue.ttt_remaining = 0;
      }
      return -1;
    }
    if (m === "load_aware") {
      // Strongest RSRP biased by load — but no rate limit → ping-pong.
      let best = ue.cell, bestScore = this.rsrpAt(ue, this.cells[ue.cell]) - this.cells[ue.cell].load * 6;
      for (let i = 0; i < this.cells.length; i++) {
        const score = this.rsrpAt(ue, this.cells[i]) - this.cells[i].load * 6;
        if (score > bestScore + 0.8) { best = i; bestScore = score; }
      }
      return best !== ue.cell ? best : -1;
    }
    if (m === "gnn_dqn") {
      // Learned policy (raw): like load_aware but smarter. Slight inertia,
      // still no SON layer → occasional aggressive moves.
      const dt_since = this.t - ue.last_ho_t;
      if (dt_since < 5) return -1;
      let best = ue.cell, bestScore = this.rsrpAt(ue, this.cells[ue.cell]) - this.cells[ue.cell].load * 4 + 0.5;
      for (let i = 0; i < this.cells.length; i++) {
        const score = this.rsrpAt(ue, this.cells[i]) - this.cells[i].load * 4;
        if (score > bestScore + 1.2) { best = i; bestScore = score; }
      }
      return best !== ue.cell ? best : -1;
    }
    if (m === "son_gnn_dqn") {
      // Learned preferences + SON CIO updates + TTT scaling. The SON
      // controller has already nudged cell.cio toward the right offsets
      // (handled in step()), so we mostly do A3-with-load via CIO.
      const best = this.bestCellByRsrp(ue);
      const dt_since = this.t - ue.last_ho_t;
      if (best !== ue.cell && dt_since > 8) {
        const margin = this.rsrpAt(ue, this.cells[best]) - this.rsrpAt(ue, this.cells[ue.cell]);
        // Higher TTT for fast UEs
        const tttFrames = ue.speed > 14 ? 6 : 3;
        if (margin > 1.5) {
          if (ue.candidate_cell !== best) {
            ue.candidate_cell = best;
            ue.ttt_remaining = tttFrames;
            return -1;
          }
          ue.ttt_remaining -= 1;
          if (ue.ttt_remaining <= 0) return best;
        } else {
          ue.candidate_cell = -1;
        }
      }
      return -1;
    }
    return -1;
  }

  // SON controller updates CIO offsets to balance load. Only active for
  // son_gnn_dqn. Bounded ±6 dB.
  sonStep(dt) {
    if (this.methodId !== "son_gnn_dqn") return;
    if ((this.t * 60) % 30 > 0.5) return; // throttle update rate
    // Compute mean load
    const mean = this.cells.reduce((a, c) => a + c.load, 0) / this.cells.length;
    for (const c of this.cells) {
      // Push CIO down for overloaded cells (so neighbors look better);
      // push CIO up for underloaded cells (so they attract more UEs).
      const target = clamp((mean - c.load) * 9, -6, 6);
      c.cio = lerp(c.cio, target, 0.06);
    }
  }

  step(dt) {
    if (!this.scenario) return;
    this.t += dt;
    const s = this.scenario;
    const w = this.bbox.w || 1, h = this.bbox.h || 1;

    // Move UEs
    for (const ue of this.ues) {
      ue.x += ue.vx * dt * 0.6;
      ue.y += ue.vy * dt * 0.6;
      // Bounce / wrap depending on layout
      if (s.mobility === "highway") {
        // Wrap horizontally
        if (ue.x < this.bbox.minX - 50) ue.x = this.bbox.maxX + 50;
        if (ue.x > this.bbox.maxX + 50) ue.x = this.bbox.minX - 50;
        // Drift vertical bounce
        if (ue.y < this.bbox.minY - h * 0.2 || ue.y > this.bbox.maxY + h * 0.2) ue.vy *= -1;
      } else {
        if (ue.x < this.bbox.minX - w * 0.05) ue.vx = Math.abs(ue.vx);
        if (ue.x > this.bbox.maxX + w * 0.05) ue.vx = -Math.abs(ue.vx);
        if (ue.y < this.bbox.minY - h * 0.05) ue.vy = Math.abs(ue.vy);
        if (ue.y > this.bbox.maxY + h * 0.05) ue.vy = -Math.abs(ue.vy);
      }

      // Trail
      if (this.t * 2.5 % 1 < 0.16) {
        ue.trail.push([ue.x, ue.y]);
        if (ue.trail.length > 10) ue.trail.shift();
      }
    }

    // SON updates CIO before decisions are made.
    this.sonStep(dt);

    // Decisions
    const events = [];
    for (const ue of this.ues) {
      const target = this.decideTarget(ue);
      if (target >= 0 && target !== ue.cell) {
        const isPingpong = target === ue.prev_cell && (this.t - ue.last_ho_t) < 6;
        ue.prev_cell = ue.cell;
        ue.cell = target;
        ue.last_ho_t = this.t;
        ue.ho_count += 1;
        if (isPingpong) ue.pingpong += 1;
        events.push({ ue, isPingpong });
        this.handoverArcs.push({
          from: { x: this.cells[ue.prev_cell].x, y: this.cells[ue.prev_cell].y },
          to:   { x: this.cells[ue.cell].x,      y: this.cells[ue.cell].y },
          t: 0, isPingpong, ueId: ue.id,
        });
      }
    }

    // Recompute attachments + load
    this.recompute();

    // Decay arcs
    for (let i = this.handoverArcs.length - 1; i >= 0; i--) {
      this.handoverArcs[i].t += dt * 1.25;
      if (this.handoverArcs[i].t > 1) this.handoverArcs.splice(i, 1);
    }

    // KPIs are comparatively expensive on dense scenarios. Keep the
    // simulation smooth and refresh the displayed values a few times/second.
    if (this.t - this.lastKpiComputeT > 0.35) {
      this.computeKPIs(events);
      this.lastKpiComputeT = this.t;
      this.kpiOnUpdate(this.kpi);
    }
  }

  recompute() {
    for (const c of this.cells) { c.attached = 0; }
    for (const ue of this.ues) this.cells[ue.cell].attached += 1;
    for (const c of this.cells) c.load = clamp(c.attached / this.capacityPerCell, 0, 1.4);
  }

  computeKPIs(events) {
    // Throughput: simple Shannon-like proxy w/ load contention.
    const throughputs = [];
    for (const ue of this.ues) {
      const c = this.cells[ue.cell];
      const rsrp = this.rsrpAt(ue, c); // rough -50…-110
      // map rsrp to 0..1
      const sig = clamp((rsrp + 110) / 60, 0, 1);
      const share = 1 / Math.max(1, c.attached);
      const cap = 12 * sig * Math.pow(share, 0.6); // Mbps approx
      throughputs.push(cap);
    }
    throughputs.sort((a, b) => a - b);
    const avg = throughputs.reduce((a, b) => a + b, 0) / throughputs.length;
    const p5Index = Math.max(0, Math.floor(throughputs.length * 0.05));
    const p5 = throughputs[p5Index] || 0;
    // Jain
    const sum = throughputs.reduce((a, b) => a + b, 0);
    const sqsum = throughputs.reduce((a, b) => a + b * b, 0);
    const jain = (sum * sum) / (throughputs.length * sqsum || 1);

    // Ping-pong rate: pingpong handovers / total handovers (last 60s window approx)
    let totalHo = 0, totalPP = 0;
    for (const ue of this.ues) {
      totalHo += ue.ho_count;
      totalPP += ue.pingpong;
    }
    const ppPct = totalHo > 0 ? (totalPP / totalHo) * 100 : 0;
    const hoPerMin = totalHo > 0 && this.t > 0 ? (totalHo / Math.max(1, this.t)) * 60 / this.ues.length : 0;

    // Outage: UEs with rsrp below threshold
    let outage = 0;
    for (const ue of this.ues) {
      if (this.rsrpAt(ue, this.cells[ue.cell]) < -100) outage += 1;
    }
    const outagePct = (outage / this.ues.length) * 100;

    this.kpi = {
      thr_avg: avg,
      thr_p5: p5,
      jain: jain,
      pingpong_pct: ppPct,
      ho_per_min: hoPerMin,
      outage_pct: outagePct,
    };
  }
}

// =========================================================================
// Renderer — draws SimEngine state to a canvas.
// =========================================================================

function drawSim(canvas, sim, opts = {}) {
  const ctx = canvas.getContext("2d");
  const map = opts.map || null;
  const dpr = map ? 1 : Math.min(1.5, Math.max(1, window.devicePixelRatio || 1));
  const r = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (!sim || !sim.scenario) return;
  const bb = sim.bbox;
  const pad = 28 * dpr;
  const w = bb.w || 1, h = bb.h || 1;

  // Maintain aspect: fit bbox into canvas with padding.
  const sxAvail = W - pad * 2, syAvail = H - pad * 2;
  let scale = Math.min(sxAvail / w, syAvail / h);
  // Center
  const ox = (W - w * scale) / 2 - bb.minX * scale;
  const oy = (H - h * scale) / 2 - bb.minY * scale;
  const sx = (x) => ox + x * scale;
  const sy = (y) => H - (oy + y * scale); // y-flip
  let mapProject = null;
  if (map && sim.centerLatLon && window.xyToLatLon) {
    const [latBL, lonBL] = window.xyToLatLon(bb.minX, bb.minY, sim.centerLatLon);
    const [latTR, lonTR] = window.xyToLatLon(bb.maxX, bb.maxY, sim.centerLatLon);
    const bl = map.latLngToContainerPoint([latBL, lonBL]);
    const tr = map.latLngToContainerPoint([latTR, lonTR]);
    mapProject = (x, y) => [
      (bl.x + ((x - bb.minX) / w) * (tr.x - bl.x)) * dpr,
      (bl.y + ((y - bb.minY) / h) * (tr.y - bl.y)) * dpr,
    ];
  }
  const project = (x, y) => {
    if (mapProject) return mapProject(x, y);
    if (map && sim.centerLatLon && window.xyToLatLon) {
      const [lat, lon] = window.xyToLatLon(x, y, sim.centerLatLon);
      const p = map.latLngToContainerPoint([lat, lon]);
      return [p.x * dpr, p.y * dpr];
    }
    return [sx(x), sy(y)];
  };
  const metersToPx = (meters, x, y) => {
    if (mapProject) {
      const [x1] = mapProject(x, y);
      const [x2] = mapProject(x + meters, y);
      return Math.max(8 * dpr, Math.abs(x2 - x1));
    }
    if (map && sim.centerLatLon && window.xyToLatLon) {
      const [lat, lon] = window.xyToLatLon(x, y, sim.centerLatLon);
      const [lat2, lon2] = window.xyToLatLon(x + meters, y, sim.centerLatLon);
      const p1 = map.latLngToContainerPoint([lat, lon]);
      const p2 = map.latLngToContainerPoint([lat2, lon2]);
      return Math.max(8 * dpr, Math.abs(p2.x - p1.x) * dpr);
    }
    return meters * scale;
  };

  // Background grid for synthetic layouts. Real-coordinate scenarios use
  // the Leaflet/OpenStreetMap layer beneath this transparent overlay.
  if (!map) {
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    const grid = 80 * dpr;
    for (let x = 0; x < W; x += grid) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += grid) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
  }

  // Landmarks (water + place labels)
  if (sim.landmarks && !map) {
    sim.landmarks.forEach((lm) => {
      const [x, y] = project(lm.x, lm.y);
      if (lm.kind === "water") {
        ctx.fillStyle = "rgba(80,140,220,0.10)";
        ctx.strokeStyle = "rgba(120,170,230,0.30)";
        ctx.beginPath();
        ctx.ellipse(x, y, 110 * dpr, 55 * dpr, 0, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = "rgba(160,200,240,0.55)";
        ctx.font = `${10 * dpr}px IBM Plex Mono`;
        ctx.fillText(lm.name, x + 8 * dpr, y + 4 * dpr);
      } else {
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.font = `${9.5 * dpr}px IBM Plex Mono`;
        ctx.fillText(lm.name, x + 8 * dpr, y + 4 * dpr);
        ctx.beginPath();
        ctx.arc(x, y, 1.5 * dpr, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.fill();
      }
    });
  }

  // Cells: coverage rings and load
  for (const c of sim.cells) {
    const [cx, cy] = project(c.x, c.y);
    const baseR = map
      ? metersToPx(520, c.x, c.y)
      : (sim.scenario.isd_m || 600) * scale * 0.55;
    // Coverage rings: fewer on map-backed panes to keep the street map legible.
    const ringCount = map ? 1 : 2;
    for (let k = ringCount; k > 0; k--) {
      ctx.beginPath();
      ctx.arc(cx, cy, baseR * (k / ringCount), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(120,170,230,${map ? 0.045 : 0.07 / k})`;
      ctx.lineWidth = map ? 0.8 * dpr : 1;
      ctx.stroke();
    }
    // Load fill
    const loadColor = c.load > 0.95 ? "rgba(225,170,70,0.24)"
                    : c.load > 0.75 ? "rgba(210,155,70,0.18)"
                    : "rgba(80,140,230,0.13)";
    ctx.fillStyle = loadColor;
    ctx.beginPath();
    ctx.arc(cx, cy, baseR * 0.5, 0, Math.PI * 2);
    ctx.fill();

    // Tower mast icon.
    ctx.save();
    ctx.translate(cx, cy);
    ctx.fillStyle = "rgba(18,25,34,0.72)";
    ctx.beginPath();
    ctx.arc(0, 0, 6.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(105,175,235,0.82)";
    ctx.lineWidth = 1.2 * dpr;
    ctx.stroke();
    ctx.strokeStyle = "rgba(245,250,255,0.92)";
    ctx.lineWidth = 1.4 * dpr;
    ctx.beginPath();
    ctx.moveTo(0, -6 * dpr);
    ctx.lineTo(0, 5 * dpr);
    ctx.moveTo(-4.5 * dpr, 5 * dpr);
    ctx.lineTo(0, -1 * dpr);
    ctx.lineTo(4.5 * dpr, 5 * dpr);
    ctx.stroke();
    ctx.fillStyle = "rgba(245,250,255,0.96)";
    ctx.fillRect(-3.5 * dpr, -7.5 * dpr, 7 * dpr, 2.2 * dpr);
    ctx.restore();

    // Load bar above tower
    const barW = 28 * dpr;
    const barH = 3 * dpr;
    const bx = cx - barW / 2, by = cy - 18 * dpr;
    ctx.fillStyle = "rgba(255,255,255,0.10)";
    ctx.fillRect(bx, by, barW, barH);
    const barFill = c.load > 0.95 ? "rgb(225,170,70)"
                  : c.load > 0.75 ? "rgb(210,155,70)"
                  : "rgb(110,170,230)";
    ctx.fillStyle = barFill;
    ctx.fillRect(bx, by, barW * Math.min(1, c.load), barH);

    // CIO badge for SON method
    if (sim.methodId === "son_gnn_dqn" && Math.abs(c.cio) > 0.4) {
      const sign = c.cio > 0 ? "+" : "";
      ctx.fillStyle = "rgba(150,200,255,0.85)";
      ctx.font = `${9 * dpr}px IBM Plex Mono`;
      ctx.fillText(`${sign}${c.cio.toFixed(1)}dB`, cx + 6 * dpr, cy + 14 * dpr);
    }

    // Cell ID
    ctx.fillStyle = "rgba(160,180,210,0.55)";
    ctx.font = `${9 * dpr}px IBM Plex Mono`;
    if (!map) ctx.fillText(`#${c.id}`, cx - 14 * dpr, cy + 14 * dpr);
  }

  // Handover arcs
  for (const a of sim.handoverArcs) {
    const [x1, y1] = project(a.from.x, a.from.y);
    const [x2, y2] = project(a.to.x, a.to.y);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - 38 * dpr;
    const alpha = 1 - a.t;
    const arcColor = a.isPingpong ? [115, 210, 190] : [225, 178, 78];
    ctx.strokeStyle = `rgba(${arcColor[0]},${arcColor[1]},${arcColor[2]},${alpha * 0.62})`;
    ctx.lineWidth = (a.isPingpong ? 2 : 1.4) * dpr;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(mx, my, x2, y2);
    ctx.stroke();

    const p = clamp(a.t, 0, 1);
    const qx = (1 - p) * (1 - p) * x1 + 2 * (1 - p) * p * mx + p * p * x2;
    const qy = (1 - p) * (1 - p) * y1 + 2 * (1 - p) * p * my + p * p * y2;
    const tx = 2 * (1 - p) * (mx - x1) + 2 * p * (x2 - mx);
    const ty = 2 * (1 - p) * (my - y1) + 2 * p * (y2 - my);
    ctx.save();
    ctx.translate(qx, qy);
    ctx.rotate(Math.atan2(ty, tx));
    ctx.fillStyle = `rgba(${arcColor[0]},${arcColor[1]},${arcColor[2]},${Math.min(1, alpha + 0.2)})`;
    ctx.beginPath();
    ctx.moveTo(7 * dpr, 0);
    ctx.lineTo(-4 * dpr, -4 * dpr);
    ctx.lineTo(-2 * dpr, 0);
    ctx.lineTo(-4 * dpr, 4 * dpr);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  const maxVisibleUes = sim.scenario.real ? 75 : 120;
  const ueStride = Math.max(1, Math.ceil(sim.ues.length / maxVisibleUes));
  const visibleUes = sim.ues.filter((ue) => ue.id % ueStride === 0 || (sim.t - ue.last_ho_t) < 1.2);

  // UE trails (subtle, sampled in dense scenarios)
  for (const ue of visibleUes) {
    if (ue.id % (ueStride * 3) !== 0 && (sim.t - ue.last_ho_t) > 0.8) continue;
    if (ue.trail.length < 2) continue;
    ctx.strokeStyle = "rgba(105,175,235,0.16)";
    ctx.lineWidth = 0.9 * dpr;
    ctx.beginPath();
    for (let i = 0; i < ue.trail.length; i++) {
      const [x, y] = ue.trail[i];
      const [px, py] = project(x, y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // UEs: clean radio-user marker.
  for (const ue of visibleUes) {
    const [px, py] = project(ue.x, ue.y);
    const c = sim.cells[ue.cell];
    const rsrp = sim.rsrpAt(ue, c);
    const inOutage = rsrp < -100;
    const recentlyMoved = (sim.t - ue.last_ho_t) < 0.6;
    const body = inOutage ? "rgb(160,190,220)"
               : recentlyMoved ? "rgb(225,180,85)"
               : "rgb(84,172,232)";
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.38)";
    ctx.shadowBlur = 3 * dpr;
    ctx.fillStyle = body;
    ctx.beginPath();
    ctx.arc(px, py, (recentlyMoved ? 3.8 : 3.0) * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(245,250,255,0.92)";
    ctx.lineWidth = 1 * dpr;
    ctx.stroke();
    if (recentlyMoved) {
      const pulse = 1 - Math.min(1, sim.t - ue.last_ho_t);
      ctx.strokeStyle = "rgba(225,180,85,0.45)";
      ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath();
      ctx.arc(px, py, 7 * dpr * pulse, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }
}

// =========================================================================
// Helpers
// =========================================================================
function mulberry32(a) {
  return function() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function lerp(a, b, t) { return a + (b - a) * t; }

window.SimEngine = SimEngine;
window.drawSim = drawSim;
