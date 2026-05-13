// =========================================================================
// topology.js — synthetic-realistic OpenCellID-style coordinates around
// known clusters in the Pokhara valley + Kathmandu, plus landmark anchors.
// Coordinates are not exact OpenCellID rows; they are realistic stand-ins
// hand-encoded from the codebase's known cluster centres.
// =========================================================================

// Pokhara valley landmarks (lat, lon) — used as visual anchors only.
const POKHARA_LANDMARKS = [
  { id: "phewa",        name: "Phewa Lake",      lat: 28.2070, lon: 83.9510, kind: "water" },
  { id: "lakeside",     name: "Lakeside",        lat: 28.2150, lon: 83.9590, kind: "place" },
  { id: "mahendrapul",  name: "Mahendrapul",     lat: 28.2335, lon: 83.9785, kind: "place" },
  { id: "chipledhunga", name: "Chipledhunga",    lat: 28.2210, lon: 83.9890, kind: "place" },
  { id: "bagar",        name: "Bagar",           lat: 28.2440, lon: 83.9930, kind: "place" },
  { id: "prithvichowk", name: "Prithvi Chowk",   lat: 28.2245, lon: 83.9870, kind: "place" },
  { id: "newroad",      name: "New Road",        lat: 28.2295, lon: 83.9820, kind: "place" },
  { id: "airport",      name: "Pokhara Airport", lat: 28.2010, lon: 84.0010, kind: "place" },
];

// 20 Pokhara cell positions clustered around Lakeside (south) and
// Mahendrapul/Chipledhunga (north). Realistic-stand-in coordinates.
const POKHARA_CELLS_LATLON = [
  // Lakeside cluster (8)
  [28.2120, 83.9555], [28.2155, 83.9580], [28.2185, 83.9610],
  [28.2110, 83.9620], [28.2160, 83.9640], [28.2090, 83.9590],
  [28.2200, 83.9560], [28.2140, 83.9530],
  // Central Pokhara (4)
  [28.2225, 83.9740], [28.2260, 83.9785], [28.2200, 83.9810],
  [28.2245, 83.9830],
  // Mahendrapul / Bagar cluster (8)
  [28.2310, 83.9760], [28.2335, 83.9800], [28.2370, 83.9810],
  [28.2360, 83.9870], [28.2420, 83.9905], [28.2450, 83.9950],
  [28.2300, 83.9885], [28.2390, 83.9930],
];

// Kathmandu valley landmarks
const KATHMANDU_LANDMARKS = [
  { id: "thamel",       name: "Thamel",        lat: 27.7150, lon: 85.3120, kind: "place" },
  { id: "durbar_marg",  name: "Durbar Marg",   lat: 27.7110, lon: 85.3185, kind: "place" },
  { id: "newroad_ktm",  name: "New Road",      lat: 27.7050, lon: 85.3110, kind: "place" },
  { id: "patan",        name: "Patan",         lat: 27.6730, lon: 85.3250, kind: "place" },
  { id: "boudha",       name: "Boudhanath",    lat: 27.7215, lon: 85.3625, kind: "place" },
  { id: "bagmati",      name: "Bagmati River", lat: 27.6920, lon: 85.3210, kind: "water" },
  { id: "tribhuvan",    name: "TIA Airport",   lat: 27.6985, lon: 85.3590, kind: "place" },
];

// 25 Kathmandu cell positions across a dense central cluster.
const KATHMANDU_CELLS_LATLON = [
  [27.7160, 85.3125], [27.7140, 85.3155], [27.7110, 85.3180], [27.7180, 85.3170],
  [27.7100, 85.3110], [27.7075, 85.3145], [27.7050, 85.3105], [27.7035, 85.3175],
  [27.7195, 85.3105], [27.7220, 85.3140], [27.7165, 85.3070], [27.7130, 85.3090],
  [27.7080, 85.3210], [27.7015, 85.3225], [27.6985, 85.3175], [27.7000, 85.3115],
  [27.7250, 85.3185], [27.7280, 85.3130], [27.7220, 85.3225], [27.7155, 85.3225],
  [27.7095, 85.3270], [27.7045, 85.3275], [27.7170, 85.3265], [27.7100, 85.3050],
  [27.7240, 85.3060],
];

// Dharan synthetic cluster (centre approx 26.81, 87.28).
const DHARAN_CELLS_LATLON = (() => {
  const arr = [];
  const cx = 26.8120, cy = 87.2820;
  // Hand-spread 20 positions across an irregular layout.
  const seeds = [
    [0.000,0.000],[0.012,0.008],[0.005,0.020],[-0.010,0.015],[0.018,-0.004],
    [-0.020,0.002],[0.008,-0.018],[-0.012,-0.016],[0.025,0.012],[-0.018,-0.025],
    [0.030,-0.010],[0.000,0.030],[-0.025,0.022],[0.022,0.025],[-0.030,-0.005],
    [0.012,-0.030],[-0.008,0.035],[0.035,0.005],[-0.005,-0.030],[0.018,0.032],
  ];
  seeds.forEach(([dx, dy]) => arr.push([cx + dx, cy + dy]));
  return arr;
})();

// =========================================================================
// Helpers
// =========================================================================

// Equirectangular projection of lat/lon → meters about a centre.
function latlonToXY(rows, center) {
  const R = 6371000; // earth radius m
  const [cLat, cLon] = center;
  const cosC = Math.cos((cLat * Math.PI) / 180);
  return rows.map(([lat, lon]) => [
    ((lon - cLon) * Math.PI / 180) * R * cosC,
    ((lat - cLat) * Math.PI / 180) * R,
  ]);
}

function bbox(points) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, maxX, minY, maxY, w: maxX - minX, h: maxY - minY };
}

function centerOf(rows) {
  const cx = rows.reduce((a, [x]) => a + x, 0) / rows.length;
  const cy = rows.reduce((a, [, y]) => a + y, 0) / rows.length;
  return [cx, cy];
}

// =========================================================================
// Procedural layouts (mirrors scenarios.py)
// =========================================================================

function hexGrid(numCells, isd) {
  const side = Math.ceil(Math.sqrt(numCells));
  const out = [];
  for (let row = 0; row <= side && out.length < numCells; row++) {
    for (let col = 0; col <= side && out.length < numCells; col++) {
      const x = col * isd + (row % 2) * isd * 0.5;
      const y = row * isd * 0.866;
      out.push([x, y]);
    }
  }
  return out.slice(0, numCells);
}

function linearLayout(numCells, isd) {
  const out = [];
  for (let i = 0; i < numCells; i++) {
    const x = i * isd;
    const y = ((Math.sin(i * 1.7) * 0.08) - 0.04) * isd;
    out.push([x, y]);
  }
  return out;
}

function ringWithHole(numCells, radius) {
  const out = [];
  for (let i = 0; i < numCells; i++) {
    const angle = (i / numCells) * Math.PI * 2;
    out.push([
      radius + radius * Math.cos(angle),
      radius + radius * Math.sin(angle),
    ]);
  }
  return out;
}

// =========================================================================
function xyToLatLon(x, y, center) {
  const R = 6371000;
  const [cLat, cLon] = center;
  const lat = cLat + ((y / R) * 180) / Math.PI;
  const lon = cLon + ((x / (R * Math.cos((cLat * Math.PI) / 180))) * 180) / Math.PI;
  return [lat, lon];
}

function latlonBounds(rows) {
  let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
  for (const [lat, lon] of rows) {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
  }
  return [[minLat, minLon], [maxLat, maxLon]];
}

// Public API: get cell positions for a scenario, normalized to a bbox.
// Returns { positions: [[x,y]...], bbox, landmarks?, isReal }
// =========================================================================

function getScenarioPositions(scenarioId) {
  const s = window.SCENARIO_BY_ID[scenarioId];
  if (!s) return null;

  let raw;
  let landmarks = null;
  let isReal = false;
  let centerLatLon = null;
  let cellLatLons = null;

  switch (scenarioId) {
    case "real_pokhara":
    case "pokhara_dense_peakhour": {
      const center = centerOf(POKHARA_CELLS_LATLON);
      raw = latlonToXY(POKHARA_CELLS_LATLON, center);
      centerLatLon = center;
      cellLatLons = POKHARA_CELLS_LATLON;
      landmarks = POKHARA_LANDMARKS.map((lm) => {
        const [x, y] = latlonToXY([[lm.lat, lm.lon]], center)[0];
        return { ...lm, x, y };
      });
      isReal = true;
      break;
    }
    case "kathmandu_real": {
      const center = centerOf(KATHMANDU_CELLS_LATLON);
      raw = latlonToXY(KATHMANDU_CELLS_LATLON, center);
      centerLatLon = center;
      cellLatLons = KATHMANDU_CELLS_LATLON;
      landmarks = KATHMANDU_LANDMARKS.map((lm) => {
        const [x, y] = latlonToXY([[lm.lat, lm.lon]], center)[0];
        return { ...lm, x, y };
      });
      isReal = true;
      break;
    }
    case "dharan_synthetic": {
      const center = centerOf(DHARAN_CELLS_LATLON);
      raw = latlonToXY(DHARAN_CELLS_LATLON, center);
      break;
    }
    case "highway":
      raw = linearLayout(s.cells, s.isd_m);
      break;
    case "coverage_hole":
      raw = ringWithHole(s.cells, 500);
      break;
    case "unknown_hex_grid":
    case "dense_urban":
    case "suburban":
    case "sparse_rural":
    case "overloaded_event":
    default:
      raw = hexGrid(s.cells, s.isd_m || 400);
      break;
  }

  return {
    positions: raw,
    bbox: bbox(raw),
    landmarks,
    isReal,
    centerLatLon,
    cellLatLons,
    boundsLatLon: cellLatLons ? latlonBounds(cellLatLons) : null,
    scenario: s,
  };
}

window.getScenarioPositions = getScenarioPositions;
window.xyToLatLon = xyToLatLon;
window.POKHARA_LANDMARKS = POKHARA_LANDMARKS;
window.KATHMANDU_LANDMARKS = KATHMANDU_LANDMARKS;
