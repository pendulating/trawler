import React, { useMemo, useEffect, useState, useCallback, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { _GlobeView as GlobeView, COORDINATE_SYSTEM } from '@deck.gl/core';
import { SolidPolygonLayer, GeoJsonLayer, ScatterplotLayer, TextLayer, IconLayer, ArcLayer } from '@deck.gl/layers';
import Overlays from './Overlays.jsx';
import ColorbarHUD from './ColorbarHUD.jsx';

const WORLD_LOCAL_URL = '/world_countries.geojson';
const WORLD_REMOTE_URL = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson';
const AI_URL = '/country_data.geojson';
const ARC_URL = '/cross_country_mentions_by_year.json';

const FIXED_ZOOM = 2.4;
const ROTATE_DEG_PER_SEC = 6; // rotation speed
const SCALE_MAX = 0.15;
const GROWTH_ABS_MAX = 0.10; // 10 percentage points
const GROWTH_PCT_MAX = 1.0;  // 100% relative growth
const VIEW_LATITUDE = 25; // Center around US/Western Europe
const ARC_DEFAULT_MIN_SHARE = 0.02;
const ARC_MAX_SEGMENTS = 600;
const ARC_SHARE_HIGHLIGHT = 0.2;
const ARC_MIN_ALPHA = 60;
const ARC_MAX_ALPHA = 230;
const NO_DATA_COLOR = [60, 65, 85, 160];

function normalizeLongitude(lon) {
  let x = lon;
  while (x > 180) x -= 360;
  while (x < -180) x += 360;
  return x;
}

// Shared 6-stop color scale (0..1)
const COLOR_STOPS = [
  { stop: 0.0,  color: [56, 189, 248] },  // sky
  { stop: 0.2,  color: [102, 126, 234] }, // indigo
  { stop: 0.4,  color: [118, 75, 162] },  // purple
  { stop: 0.6,  color: [236, 72, 153] },  // pink
  { stop: 0.8,  color: [251, 146, 60] },  // orange
  { stop: 1.0,  color: [239, 68, 68] }    // red
];

// Diverging for growth metrics [-max, 0, +max]
const DIVERGING_STOPS = [
  { stop: 0.0, color: [59, 130, 246] },   // blue
  { stop: 0.5, color: [226, 232, 240] },  // light gray
  { stop: 1.0, color: [239, 68, 68] }     // red
];

function interpolateStops(t, stops) {
  const tt = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (tt >= a.stop && tt <= b.stop) {
      const local = (tt - a.stop) / Math.max(1e-6, (b.stop - a.stop));
      return [
        Math.round(a.color[0] + (b.color[0] - a.color[0]) * local),
        Math.round(a.color[1] + (b.color[1] - a.color[1]) * local),
        Math.round(a.color[2] + (b.color[2] - a.color[2]) * local)
      ];
    }
  }
  return stops[stops.length - 1].color.slice(0, 3);
}

export default function App() {
  const [landData, setLandData] = useState(null);
  const [aiData, setAiData] = useState(null);
  const [arcData, setArcData] = useState(null);
  const [periods, setPeriods] = useState([]);
  const [periodIndex, setPeriodIndex] = useState(0);
  const [viewMode, setViewMode] = useState('surface');
  const [arcYears, setArcYears] = useState([]);
  const [arcYearIndex, setArcYearIndex] = useState(0);
  const [arcMinShare, setArcMinShare] = useState(ARC_DEFAULT_MIN_SHARE);
  const [arcMaxVisible, setArcMaxVisible] = useState(400);
  const [autoAdvance, setAutoAdvance] = useState(true);
  const [mode, setMode] = useState('fraction'); // 'fraction' | 'growth_abs' | 'growth_pct'
  const [growthPctMax, setGrowthPctMax] = useState(1.0);
  const deckRef = useRef(null);
  const rotationLonRef = useRef(0);
  const [hoverUi, setHoverUi] = useState(null);
  const lastHoverTsRef = useRef(0);
  const [autoOverlays, setAutoOverlays] = useState([]);
  const [hudMarkers, setHudMarkers] = useState([]);
  const readyRef = useRef(false);
  const [webglReady, setWebglReady] = useState(false);
  const [size, setSize] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }));
  const [deckMounted, setDeckMounted] = useState(false);
  const containerRef = useRef(null);
  const roPatchedRef = useRef(false);
  const labelIndexRef = useRef(new Map());
  const [cameraOverlay, setCameraOverlay] = useState(null);
  const lastCamOverlayTsRef = useRef(0);
  const jitterMapRef = useRef(new Map());
  const [transitionsEnabled, setTransitionsEnabled] = useState(false);
  // Period label animation state (one-way dial)
  const [periodLabelTop, setPeriodLabelTop] = useState('');
  const [nextPeriodLabel, setNextPeriodLabel] = useState('');
  const [isPeriodSliding, setIsPeriodSliding] = useState(false);
  const [disablePeriodTransition, setDisablePeriodTransition] = useState(false);
  const [paused, setPaused] = useState(false);

  const initialViewState = useMemo(() => ({
    longitude: 0,
    latitude: VIEW_LATITUDE,
    zoom: FIXED_ZOOM,
    minZoom: FIXED_ZOOM,
    maxZoom: FIXED_ZOOM,
    pitch: 0,
    bearing: 0
  }), []);

  // Initialize top label when periods first load
  useEffect(() => {
    if (!periods.length) return;
    const current = periods[periodIndex] || '';
    if (!periodLabelTop) setPeriodLabelTop(current);
  }, [periods, periodIndex, periodLabelTop]);

  // Trigger period label slide on change (5s forward, hold) after transitions enabled
  useEffect(() => {
    if (!periods.length || !transitionsEnabled) return;
    const current = periods[periodIndex] || '';
    if (current === periodLabelTop) return;
    setNextPeriodLabel(current);
    setIsPeriodSliding(true);
    const endId = setTimeout(() => {
      // Swap content without reverse animation
      setDisablePeriodTransition(true);
      setIsPeriodSliding(false);
      setPeriodLabelTop(current);
      setNextPeriodLabel('');
      // Re-enable transition on next frame
      const reId = setTimeout(() => setDisablePeriodTransition(false), 0);
      return () => clearTimeout(reId);
    }, 5000);
    return () => clearTimeout(endId);
  }, [periodIndex, periods, periodLabelTop, transitionsEnabled]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        let resp = await fetch(WORLD_LOCAL_URL);
        if (!resp.ok) resp = await fetch(WORLD_REMOTE_URL);
        const json = await resp.json();
        if (!cancelled) { setLandData(json); console.log('[autoOverlay] land loaded', Array.isArray(json.features) ? json.features.length : 'n/a'); }
      } catch (e) {
        console.error('Failed loading world countries', e);
      }
    })();
    (async () => {
      try {
        const resp = await fetch(AI_URL);
        if (resp.ok) {
          const json = await resp.json();
          if (!cancelled) {
            setAiData(json);
            const feats = json.type === 'FeatureCollection' ? json.features : json;
            const sample = feats.find(f => f && f.properties && Object.keys(f.properties).some(k => k.startsWith('ai_fraction_')));
            if (sample) {
              const ps = Object.keys(sample.properties)
                .filter(k => k.startsWith('ai_fraction_'))
                .map(k => k.replace('ai_fraction_', '').replace(/_/g, '-'))
                .sort((a, b) => parseInt(a.split('-')[0]) - parseInt(b.split('-')[0]));
              setPeriods(ps);
              setPeriodIndex(0);
              console.log('[autoOverlay] ai loaded', feats.length, 'periods', ps);
              // Compute dynamic relative growth max across all periods/regions
              let maxAbsGrowthPct = 0;
              for (const f of feats) {
                const p = f.properties || {};
                for (const k of Object.keys(p)) {
                  if (!k.startsWith('ai_growth_pct_')) continue;
                  const v = Math.abs(Number(p[k]));
                  if (Number.isFinite(v)) maxAbsGrowthPct = Math.max(maxAbsGrowthPct, v);
                }
              }
              // Default to 1.0 if data has no growth or all zeros
              setGrowthPctMax(maxAbsGrowthPct > 0 ? Math.min(maxAbsGrowthPct, 5.0) : 1.0);
              console.log('[growth] dynamic relative max =', maxAbsGrowthPct);
            }
            // Start 5s initial buffer before enabling transitions and auto-advance/overlays
            setTransitionsEnabled(false);
            setTimeout(() => { if (!cancelled) setTransitionsEnabled(true); }, 5000);
          }
        }
      } catch (e) {
        console.error('Failed loading AI data', e);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(ARC_URL);
        if (!resp.ok) return;
        const json = await resp.json();
        if (cancelled) return;
        const records = Array.isArray(json) ? json : [];
        setArcData(records);
        const years = Array.from(
          new Set(
            records
              .map(d => Number(d.year))
              .filter(v => Number.isFinite(v))
          )
        ).sort((a, b) => a - b);
        if (years.length) {
          setArcYears(years);
          setArcYearIndex(0);
        }
      } catch (e) {
        console.error('Failed loading arc data', e);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const id = requestAnimationFrame(() => setDeckMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    if (roPatchedRef.current) return;
    if (typeof window === 'undefined' || !('ResizeObserver' in window)) return;
    const OriginalRO = window.ResizeObserver;
    function FilteredRO(cb) {
      const filteredCb = (entries, obs) => {
        const root = containerRef.current;
        if (!root) return cb(entries, obs);
        const filtered = entries.filter(e => {
          const t = e && e.target;
          return !(t && (t === root || root.contains(t)));
        });
        if (filtered.length) cb(filtered, obs);
      };
      return new OriginalRO(filteredCb);
    }
    FilteredRO.prototype = OriginalRO.prototype;
    window.ResizeObserver = FilteredRO;
    roPatchedRef.current = true;
    return () => {
      window.ResizeObserver = OriginalRO;
      roPatchedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const idx = new Map();
    if (landData && landData.features) {
      for (const f of landData.features) {
        const p = f.properties || {};
        const iso = (p.ISO_A2 || p.iso_a2 || p.ISO_A2_EH || '').toUpperCase();
        const lx = p.LABEL_X;
        const ly = p.LABEL_Y;
        if (!iso || typeof lx !== 'number' || typeof ly !== 'number') continue;
        const name = p.ADMIN || p.NAME || 'Unknown';
        idx.set(iso, { lon: lx, lat: ly, name });
      }
    }
    labelIndexRef.current = idx;
  }, [landData]);

  const getMetricValue = useCallback((props) => {
    if (!periods.length) return 0;
    const periodKey = periods[periodIndex].replace('-', '_');
    if (mode === 'fraction') return props[`ai_fraction_${periodKey}`] || 0;
    if (mode === 'growth_abs') return props[`ai_growth_${periodKey}`] || 0;
    if (mode === 'growth_pct') return props[`ai_growth_pct_${periodKey}`] || 0;
    return 0;
  }, [periods, periodIndex, mode]);

  const getAIColor = useCallback((value) => {
    if (value === null || value === undefined || isNaN(value)) return [140, 160, 200, 160];
    if (mode === 'fraction') {
      const t = Math.min(Math.max(value / SCALE_MAX, 0), 1);
      const [r, g, b] = interpolateStops(t, COLOR_STOPS);
      return [r, g, b, 220];
    }
    if (mode === 'growth_abs') {
      const max = GROWTH_ABS_MAX;
      const n = (Math.max(-max, Math.min(max, value)) + max) / (2 * max); // symmetric
      const [r, g, b] = interpolateStops(n, DIVERGING_STOPS);
      return [r, g, b, 220];
    }
    // growth_pct: asymmetric domain [-1, +growthPctMax]
    const posMax = Math.max(0.001, growthPctMax || GROWTH_PCT_MAX);
    const v = Math.max(-1, Math.min(posMax, value));
    let t;
    if (v < 0) {
      // map [-1, 0] -> [0, 0.5]
      t = 0.5 * (v + 1);
    } else {
      // map [0, posMax] -> [0.5, 1]
      t = 0.5 + 0.5 * (v / posMax);
    }
    const [r, g, b] = interpolateStops(t, DIVERGING_STOPS);
    return [r, g, b, 220];
  }, [mode, growthPctMax]);

  const getCountryFill = useCallback((props) => {
    if (!periods.length) return NO_DATA_COLOR;
    const periodKey = periods[periodIndex].replace('-', '_');
    const total = Number(props[`total_${periodKey}`] || 0);
    if (!total) return NO_DATA_COLOR;
    return getAIColor(getMetricValue(props));
  }, [periods, periodIndex, getMetricValue, getAIColor]);

  const background = useMemo(() => new SolidPolygonLayer({
    id: 'earth-background',
    data: [[[-180, 90], [0, 90], [180, 90], [180, -90], [0, -90], [-180, -90], [-180, 90]]],
    getPolygon: d => d,
    stroked: false,
    filled: true,
    pickable: false,
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    getFillColor: [30, 55, 135]
  }), []);

  const land = useMemo(() => {
    if (!landData) return null;
    const data = landData.type === 'FeatureCollection' ? landData.features : landData;
    return new GeoJsonLayer({
      id: 'land',
      data,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      filled: true,
      stroked: true,
      getFillColor: [180, 205, 170, 255],
      getLineColor: [240, 250, 255, 180],
      lineWidthMinPixels: 1,
      pickable: false
    });
  }, [landData]);

  const aiLayer = useMemo(() => {
    if (!aiData) return null;
    const data = aiData.type === 'FeatureCollection' ? aiData.features : aiData;
    return new GeoJsonLayer({
      id: 'ai-countries',
      data,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      filled: true,
      extruded: true,
      stroked: true,
      getFillColor: f => getCountryFill(f.properties),
      getLineColor: [40, 60, 120, 180],
      lineWidthMinPixels: 1,
      getElevation: f => {
        const v = getMetricValue(f.properties);
        if (mode === 'fraction') {
          const capped = Math.min(Math.max(v, 0), SCALE_MAX);
          return (capped / SCALE_MAX) * 1200000;
        }
        if (mode === 'growth_abs') {
          const max = GROWTH_ABS_MAX;
          const mag = Math.min(Math.abs(v), max);
          return (mag / max) * 1200000;
        }
        // growth_pct: asymmetric scaling, neg side capped at 1.0, pos side at growthPctMax
        const posMax = Math.max(0.001, growthPctMax || GROWTH_PCT_MAX);
        const ratio = v < 0 ? Math.min(Math.abs(v), 1.0) / 1.0 : Math.min(Math.max(v, 0), posMax) / posMax;
        return ratio * 1200000;
      },
      updateTriggers: {
        getFillColor: [periodIndex, mode, growthPctMax],
        getElevation: [periodIndex, mode, growthPctMax]
      },
      transitions: {
        getFillColor: { duration: 5000 },
        getElevation: { duration: 5000 }
      },
      pickable: true
    });
  }, [aiData, getMetricValue, getAIColor, periodIndex, mode]);

  const hudLayer = useMemo(() => new ScatterplotLayer({
    id: 'auto-hud',
    data: hudMarkers,
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    getPosition: d => d.position,
    radiusUnits: 'pixels',
    getRadius: d => 8 * (d.opacity ?? 1),
    getFillColor: d => [244, 63, 94, Math.round(255 * (d.opacity ?? 1))],
    parameters: { depthTest: false },
    transitions: {
      getRadius: { duration: 400 },
      getFillColor: { duration: 400 }
    }
  }), [hudMarkers]);

  const hudIconLayer = useMemo(() => new IconLayer({
    id: 'auto-hud-icons',
    data: hudMarkers,
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    getPosition: d => d.position,
    getIcon: d => ({ url: d.iso2 ? `https://flagcdn.com/h20/${d.iso2}.png` : '', width: 26, height: 20, anchorY: 10, anchorX: 13 }),
    sizeUnits: 'pixels',
    sizeScale: 1,
    getSize: d => 26 * (d.opacity ?? 1),
    parameters: { depthTest: false },
    transitions: { getSize: { duration: 400 } }
  }), [hudMarkers]);

  const hudTextLayer = useMemo(() => new TextLayer({
    id: 'auto-hud-text',
    data: hudMarkers,
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    getPosition: d => d.position,
    getText: d => `${d.name}: ${Number(d.total || 0).toLocaleString()}`,
    getColor: d => [255, 255, 255, Math.round(255 * (d.opacity ?? 1))],
    getSize: d => 16 * (d.opacity ?? 1),
    sizeUnits: 'pixels',
    getTextAnchor: 'middle',
    getAlignmentBaseline: 'bottom',
    getPixelOffset: [0, -16],
    parameters: { depthTest: false },
    transitions: {
      getSize: { duration: 400 },
      getColor: { duration: 400 }
    }
  }), [hudMarkers]);

  const debugPoint = useMemo(() => new ScatterplotLayer({
    id: 'debug-points',
    data: [{ position: [0, 0], color: [255, 200, 0] }],
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    getPosition: d => d.position,
    getFillColor: d => d.color,
    filled: true,
    stroked: true,
    getLineColor: [255, 255, 255],
    lineWidthMinPixels: 2,
    radiusUnits: 'pixels',
    getRadius: 1200000,
    radiusMinPixels: 12,
    radiusMaxPixels: 64,
    pickable: false
  }), []);

  const updateCameraOverlay = useCallback(() => {
    const now = performance.now();
    if (now - lastCamOverlayTsRef.current < 100) return; // throttle ~10 fps
    lastCamOverlayTsRef.current = now;
    const deck = deckRef.current && deckRef.current.deck;
    if (!deck) return;
    const viewports = (typeof deck.getViewports === 'function') ? deck.getViewports() : [];
    const vp = Array.isArray(viewports) && viewports.length ? (viewports.find(v => v && v.id === 'globe') || viewports[0]) : null;
    if (!vp || typeof vp.project !== 'function') return;
    const xy = vp.project([rotationLonRef.current, VIEW_LATITUDE]);
    if (!xy || !Number.isFinite(xy[0]) || !Number.isFinite(xy[1])) return;
    const x = Math.max(0, Math.min(size.width - 1, xy[0]));
    const y = Math.max(0, Math.min(size.height - 1, xy[1] - 24));
    setCameraOverlay({ x, y });
  }, [size.width, size.height]);

  const onViewStateChange = useCallback(({ viewState: vs }) => {
    rotationLonRef.current = vs.longitude;
    const deck = deckRef.current && deckRef.current.deck;
    if (deck) {
      deck.setProps({
        viewState: {
          longitude: normalizeLongitude(rotationLonRef.current),
          latitude: VIEW_LATITUDE,
          zoom: FIXED_ZOOM,
          minZoom: FIXED_ZOOM,
          maxZoom: FIXED_ZOOM,
          pitch: 0,
          bearing: 0
        }
      });
      updateCameraOverlay();
    }
  }, [updateCameraOverlay]);

  useEffect(() => {
    let rafId = 0;
    let lastTs = performance.now();
    const tick = (now) => {
      const dt = (now - lastTs) / 1000;
      lastTs = now;
      if (!paused) {
        rotationLonRef.current = normalizeLongitude(rotationLonRef.current + ROTATE_DEG_PER_SEC * dt);
      }
      const deck = deckRef.current && deckRef.current.deck;
      if (readyRef.current && deck) {
        deck.setProps({
          viewState: {
            longitude: rotationLonRef.current,
            latitude: VIEW_LATITUDE,
            zoom: FIXED_ZOOM,
            minZoom: FIXED_ZOOM,
            maxZoom: FIXED_ZOOM,
            pitch: 0,
            bearing: 0
          }
        });
        updateCameraOverlay();
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [updateCameraOverlay, paused]);

  useEffect(() => {
    let pending = false;
    const onResize = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        setSize({ width: window.innerWidth, height: window.innerHeight });
      });
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'a' || e.key === 'A') {
        setViewMode(mode => (mode === 'surface' ? 'arcs' : 'surface'));
        return;
      }
      if (viewMode === 'surface') {
        if (!periods.length) return;
        if (e.key === 'ArrowRight') setPeriodIndex(i => (i + 1) % periods.length);
        if (e.key === 'ArrowLeft') setPeriodIndex(i => (i + periods.length - 1) % periods.length);
      } else if (viewMode === 'arcs') {
        if (!arcYears.length) return;
        if (e.key === 'ArrowRight') setArcYearIndex(i => (i + 1) % arcYears.length);
        if (e.key === 'ArrowLeft') setArcYearIndex(i => (i + arcYears.length - 1) % arcYears.length);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [periods, arcYears, viewMode]);

  useEffect(() => {
    if (viewMode !== 'surface' || !autoAdvance || periods.length <= 1 || !transitionsEnabled || paused) return;
    const next = () => setPeriodIndex(i => (i + 1) % periods.length);
    // First advance exactly at 5s mark (when transitionsEnabled flips) to align HUD and layers
    const firstId = setTimeout(next, 0);
    const intervalId = setInterval(next, 10000);
    return () => { clearTimeout(firstId); clearInterval(intervalId); };
  }, [autoAdvance, periods, transitionsEnabled, paused, viewMode]);

  const isFrontFacing = useCallback((lonDeg, latDeg) => {
    const lon = (lonDeg * Math.PI) / 180;
    const lat = (latDeg * Math.PI) / 180;
    const camLon = (rotationLonRef.current * Math.PI) / 180;
    const camLat = (VIEW_LATITUDE * Math.PI) / 180;
    const dot = Math.sin(lat) * Math.sin(camLat) + Math.cos(lat) * Math.cos(camLat) * Math.cos(lon - camLon);
    return dot > 0;
  }, []);

  useEffect(() => {
    if (viewMode !== 'surface' || !aiData || !webglReady) {
      setHudMarkers([]);
      return;
    }
    const deck = deckRef.current && deckRef.current.deck;
    if (!deck) return;
    let cancelled = false;
    let timeouts = [];
    let intervalId = null;

    if (paused) {
      // On pause, clear any visible HUDs
      setHudMarkers([]);
      return () => {};
    }

    const showOverlays = () => {
      if (cancelled) return;
      const period = periods[periodIndex] || '';
      const periodKey = period.replace('-', '_');
      try {
        const feats = aiData.type === 'FeatureCollection' ? aiData.features : aiData;
        const idx = labelIndexRef.current;
        const candidates = [];
        for (let i = 0; i < feats.length; i++) {
          const p = feats[i].properties || {};
          const iso = (p.country_iso || p.ISO_A2 || p.iso_a2 || p.ISO_A2_EH || '').toUpperCase();
          const label = idx.get(iso);
          if (!label) continue;
          const { lon: lx, lat: ly, name: fallbackName } = label;
          if (!isFrontFacing(lx, ly)) continue;
          const total = Number(p[`total_${periodKey}`] || 0);
          const name = p.country_name || p.ADMIN || p.NAME || fallbackName || 'Unknown';
          candidates.push({ lon: lx, lat: ly, name, total, iso2: iso.toLowerCase() });
          if (candidates.length > 3000) break;
        }
        const picks = [];
        const n = Math.min(10, candidates.length);
        for (let k = 0; k < n; k++) {
          const idxP = (Math.random() * candidates.length) | 0;
          picks.push(candidates[idxP]);
          candidates.splice(idxP, 1);
        }
        const baseMarkers = picks.map(c => ({ position: [c.lon, c.lat + 2], name: c.name, total: c.total, period, opacity: 0, iso2: c.iso2 }));
        setHudMarkers(baseMarkers);
        // fade in immediately
        const tIn = setTimeout(() => { if (!cancelled) setHudMarkers(ms => ms.map(m => ({ ...m, opacity: 1 }))); }, 0);
        // fade out near end of 5s window
        const tOut = setTimeout(() => { if (!cancelled) setHudMarkers(ms => ms.map(m => ({ ...m, opacity: 0 }))); }, 4600);
        // clear at 5s
        const tClear = setTimeout(() => { if (!cancelled) setHudMarkers([]); }, 5000);
        timeouts.push(tIn, tOut, tClear);
      } catch (_) {
        setHudMarkers([]);
      }
    };

    // Delay first overlays by 5s to align with initial buffer
    const startTimeout = setTimeout(() => {
      if (cancelled) return;
      showOverlays();
      intervalId = setInterval(showOverlays, 5000);
    }, 5000);

    return () => {
      cancelled = true;
      clearTimeout(startTimeout);
      if (intervalId) clearInterval(intervalId);
      timeouts.forEach(id => clearTimeout(id));
    };
  }, [aiData, webglReady, periods, periodIndex, isFrontFacing, paused, viewMode]);

  useEffect(() => {
    if (viewMode !== 'surface') {
      setHoverUi(null);
    }
  }, [viewMode]);

  const onHover = useCallback((info) => {
    if (viewMode !== 'surface') {
      setHoverUi(null);
      return;
    }
    const now = performance.now();
    if (now - lastHoverTsRef.current < 80) return;
    lastHoverTsRef.current = now;
    if (!info || !info.object || info.layer?.id !== 'ai-countries') {
      setHoverUi(null);
      return;
    }
    const props = info.object.properties || {};
    const period = periods[periodIndex] || '';
    const periodKey = period.replace('-', '_');
    const name = props.country_name || props.ADMIN || props.NAME || 'Unknown';
    const total = props[`total_${periodKey}`] || 0;
    setHoverUi({
      x: info.x,
      y: info.y,
      name,
      period,
      total: Number(total) || 0
    });
  }, [periods, periodIndex, viewMode]);

  const deckTooltip = useCallback(({ object, layer }) => {
    if (!object) return null;
    if (viewMode === 'arcs' && layer?.id === 'arc-links') {
      const sharePct = ((object.mention_share || 0) * 100).toFixed(2);
      return {
        html: `
          <div class="tooltip-country">${object.sourceName} → ${object.targetName}</div>
          <div class="tooltip-row"><span class="tooltip-label">Year:</span><span class="tooltip-value">${object.year ?? '—'}</span></div>
          <div class="tooltip-row"><span class="tooltip-label">Mentions:</span><span class="tooltip-value">${Number(object.mention_occurrences || 0).toLocaleString()}</span></div>
          <div class="tooltip-row"><span class="tooltip-label">Share:</span><span class="tooltip-value">${sharePct}%</span></div>
        `,
        style: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderRadius: '8px',
          padding: '10px 12px',
          fontSize: '12px',
          color: '#e2e8f0'
        }
      };
    }
    return null;
  }, [viewMode]);

  const colorbarStops = useMemo(() => (mode === 'fraction' ? COLOR_STOPS : DIVERGING_STOPS), [mode]);

  const colorbarDots = useMemo(() => {
    if (viewMode !== 'surface' || !aiData || !periods.length) return [];
    const period = periods[periodIndex] || '';
    const pk = period.replace('-', '_');
    const feats = aiData.type === 'FeatureCollection' ? aiData.features : aiData;
    const out = [];
    for (const f of feats) {
      const p = f.properties || {};
      const total = Number(p[`total_${pk}`] || 0);
      if (!total) continue;
      const value = Number(
        mode === 'fraction' ? (p[`ai_fraction_${pk}`] || 0) :
        mode === 'growth_abs' ? (p[`ai_growth_${pk}`] || 0) :
        (p[`ai_growth_pct_${pk}`] || 0)
      );
      const name = p.country_name || p.ADMIN || p.NAME || '';
      const iso2 = (p.country_iso || p.ISO_A2 || p.iso_a2 || p.ISO_A2_EH || '').toLowerCase();
      const color = getAIColor(value);
      const jitterKey = iso2 || (p.country_iso || p.ISO_A2 || name || '');
      let jitter = jitterMapRef.current.get(jitterKey);
      if (typeof jitter !== 'number') {
        jitter = (Math.random() - 0.5) * 10; // stable per country for session
        jitterMapRef.current.set(jitterKey, jitter);
      }
      out.push({ id: p.country_iso || p.ISO_A2 || name, iso2, name, total, value, color, jitterY: jitter });
    }
    return out;
  }, [aiData, periods, periodIndex, mode, getAIColor, viewMode]);

  const currentArcYear = arcYears.length ? arcYears[arcYearIndex % arcYears.length] : null;

  const arcYearBundle = useMemo(() => {
    if (!arcData || !arcData.length || currentArcYear === null || !landData) {
      return { rows: [], totalAvailable: 0, maxOccurrences: 0 };
    }
    const idx = labelIndexRef.current;
    const rows = [];
    for (const entry of arcData) {
      const entryYear = Number(entry.year);
      if (!Number.isFinite(entryYear) || entryYear !== currentArcYear) continue;
      const sourceIso = (entry.source_country || '').toUpperCase();
      const targetIso = (entry.mentioned_country || '').toUpperCase();
      const sourcePos = idx.get(sourceIso);
      const targetPos = idx.get(targetIso);
      if (!sourcePos || !targetPos) continue;
      const share = Number(entry.mention_share) || 0;
      if (share < arcMinShare) continue;
      rows.push({
        ...entry,
        year: entryYear,
        source_country: sourceIso,
        mentioned_country: targetIso,
        mention_occurrences: Number(entry.mention_occurrences) || 0,
        article_hits: Number(entry.article_hits) || 0,
        mention_share: share,
        sourcePosition: [sourcePos.lon, sourcePos.lat],
        targetPosition: [targetPos.lon, targetPos.lat],
        sourceName: entry.source_country_name || sourcePos.name || sourceIso,
        targetName: entry.mentioned_country_name || targetPos.name || targetIso,
      });
    }
    rows.sort((a, b) => b.mention_occurrences - a.mention_occurrences);
    const totalAvailable = rows.length;
    const limited = rows.slice(0, arcMaxVisible);
    const maxOccurrences = limited.length ? limited[0].mention_occurrences : 0;
    return { rows: limited, totalAvailable, maxOccurrences };
  }, [arcData, arcYears, arcYearIndex, currentArcYear, arcMinShare, arcMaxVisible, landData]);

  const arcNodes = useMemo(() => {
    const nodeMap = new Map();
    const accumulate = (iso, position, name, weight) => {
      const key = iso;
      const existing = nodeMap.get(key);
      if (existing) {
        existing.total += weight;
      } else {
        nodeMap.set(key, { id: key, position, name, iso2: iso.toLowerCase(), total: weight });
      }
    };
    for (const row of arcYearBundle.rows) {
      accumulate(row.source_country, row.sourcePosition, row.sourceName, row.mention_occurrences || 0);
      accumulate(row.mentioned_country, row.targetPosition, row.targetName, row.mention_occurrences || 0);
    }
    return Array.from(nodeMap.values());
  }, [arcYearBundle]);

  const arcSummary = useMemo(() => ({
    year: currentArcYear,
    shown: arcYearBundle.rows.length,
    available: arcYearBundle.totalAvailable
  }), [arcYearBundle, currentArcYear]);

  const arcLayer = useMemo(() => {
    if (viewMode !== 'arcs' || !arcYearBundle.rows.length) return null;
    const maxOcc = Math.max(arcYearBundle.maxOccurrences, 1);
    const alphaFromShare = (share) => {
      const intensity = Math.min(Math.max(share, 0) / ARC_SHARE_HIGHLIGHT, 1);
      return Math.round(ARC_MIN_ALPHA + (ARC_MAX_ALPHA - ARC_MIN_ALPHA) * intensity);
    };
    return new ArcLayer({
      id: 'arc-links',
      data: arcYearBundle.rows,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      greatCircle: true,
      getSourcePosition: d => d.sourcePosition,
      getTargetPosition: d => d.targetPosition,
      getSourceColor: d => [64, 196, 255, alphaFromShare(d.mention_share || 0)],
      getTargetColor: d => [248, 113, 113, alphaFromShare(d.mention_share || 0)],
      getWidth: d => 1 + 8 * Math.sqrt((d.mention_occurrences || 0) / maxOcc),
      pickable: true,
      parameters: { depthTest: true, blend: true }
    });
  }, [arcYearBundle, viewMode]);

  const arcNodeLayer = useMemo(() => {
    if (viewMode !== 'arcs' || !arcNodes.length) return null;
    const maxNodeWeight = arcNodes.reduce((max, node) => Math.max(max, node.total || 0), 0) || 1;
    return new ScatterplotLayer({
      id: 'arc-nodes',
      data: arcNodes,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
      getPosition: d => d.position,
      getFillColor: [15, 23, 42, 230],
      getLineColor: [255, 255, 255, 200],
      getRadius: d => 100000 + 800000 * Math.sqrt((d.total || 0) / maxNodeWeight),
      stroked: true,
      filled: true,
      pickable: false,
      radiusUnits: 'meters',
      parameters: { depthTest: true }
    });
  }, [arcNodes, viewMode]);

  const layers = useMemo(() => {
    if (viewMode === 'surface') {
      return [background, land, aiLayer, hudIconLayer, hudTextLayer].filter(Boolean);
    }
    return [background, land, arcLayer, arcNodeLayer].filter(Boolean);
  }, [background, land, aiLayer, hudIconLayer, hudTextLayer, arcLayer, arcNodeLayer, viewMode]);

  return (
    <div style={{width: '100vw', height: '100vh', position: 'relative'}}>
      <div ref={containerRef} style={{position: 'absolute', inset: 0, contain: 'size layout paint', overflow: 'hidden'}}>
        {deckMounted && (
        <DeckGL
          ref={deckRef}
          views={[new GlobeView({ id: 'globe', resolution: 1, nearZMultiplier: 0.003, farZMultiplier: 30, controller: { dragPan: true, scrollZoom: false, touchZoom: false, doubleClickZoom: false, keyboard: false, inertia: 0 } })]}
          initialViewState={initialViewState}
          onViewStateChange={onViewStateChange}
          layers={layers}
          getTooltip={deckTooltip}
          useDevicePixels={1}
          glOptions={{ powerPreference: 'high-performance', antialias: false, stencil: false, desynchronized: true }}
          parameters={{ clearColor: [0.05, 0.08, 0.16, 1], depthTest: true, depthMask: true }}
          style={{position: 'absolute', inset: 0}}
          width={size.width}
          height={size.height}
          onWebGLInitialized={() => { readyRef.current = true; setWebglReady(true); console.log('[webgl] ready'); }}
        />)}
      </div>
      <Overlays hoverUi={hoverUi} autoOverlays={autoOverlays} cameraOverlay={cameraOverlay} />
      {viewMode === 'surface' && (
        <ColorbarHUD leftPx={240} scaleMax={mode === 'fraction' ? SCALE_MAX : (growthPctMax || 1)} dots={colorbarDots} gradientStops={colorbarStops} mode={mode} transitionsEnabled={transitionsEnabled} />
      )}
      <div style={{position: 'absolute', top: 12, left: 12, background: 'rgba(15,23,42,0.8)', padding: '8px 12px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)', width: 220, pointerEvents: 'auto'}}>
        <div style={{fontSize: 12, color: '#94a3b8'}}>View</div>
        <div style={{marginTop: 6, display: 'flex', gap: 6}}>
          <button onClick={() => setViewMode('surface')} style={{flex: 1, padding: '4px 8px', fontSize: 12, borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: viewMode === 'surface' ? '#0b1026' : '#cbd5e1', background: viewMode === 'surface' ? '#AFC7FF' : 'transparent', cursor: 'pointer'}}>
            AI Coverage
          </button>
          <button onClick={() => setViewMode('arcs')} style={{flex: 1, padding: '4px 8px', fontSize: 12, borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: viewMode === 'arcs' ? '#0b1026' : '#cbd5e1', background: viewMode === 'arcs' ? '#AFC7FF' : 'transparent', cursor: 'pointer'}}>
            Cross-links
          </button>
        </div>

        {viewMode === 'surface' ? (
          <>
            <div style={{marginTop: 10, fontSize: 12, color: '#94a3b8'}}>Period</div>
            <div style={{position: 'relative', width: '100%', height: 22, overflow: 'hidden'}}>
              <div style={{position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', transform: isPeriodSliding ? 'translateY(-100%)' : 'translateY(0%)', transition: disablePeriodTransition ? 'none' : 'transform 5000ms ease-in-out'}}>
                <div style={{position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', display: 'flex', alignItems: 'center', color: '#AFC7FF', fontSize: 14, fontWeight: 600}}>
                  {periodLabelTop || '—'}
                </div>
                <div style={{position: 'absolute', left: 0, top: '100%', width: '100%', height: '100%', display: 'flex', alignItems: 'center', color: '#AFC7FF', fontSize: 14, fontWeight: 600}}>
                  {nextPeriodLabel}
                </div>
              </div>
            </div>
            <div style={{marginTop: 6, fontSize: 11, color: '#94a3b8'}}>Use ← → to change period</div>
            <label style={{display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, fontSize: 12, color: '#cbd5e1', cursor: 'pointer'}}>
              <input type="checkbox" checked={autoAdvance} onChange={e => setAutoAdvance(e.target.checked)} />
              Auto-advance (5s)
            </label>
            <div style={{marginTop: 10, fontSize: 12, color: '#94a3b8'}}>Mode</div>
            <div style={{marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap'}}>
              <button onClick={() => setMode('fraction')} style={{flex: 1, padding: '4px 6px', fontSize: 12, borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: mode === 'fraction' ? '#0b1026' : '#cbd5e1', background: mode === 'fraction' ? '#AFC7FF' : 'transparent', cursor: 'pointer'}}>
                AI Fraction
              </button>
              <button onClick={() => setMode('growth_abs')} style={{flex: 1, padding: '4px 6px', fontSize: 12, borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: mode === 'growth_abs' ? '#0b1026' : '#cbd5e1', background: mode === 'growth_abs' ? '#AFC7FF' : 'transparent', cursor: 'pointer'}}>
                Growth (Δ)
              </button>
              <button onClick={() => setMode('growth_pct')} style={{flex: 1, padding: '4px 6px', fontSize: 12, borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: mode === 'growth_pct' ? '#0b1026' : '#cbd5e1', background: mode === 'growth_pct' ? '#AFC7FF' : 'transparent', cursor: 'pointer'}}>
                Growth (%)
              </button>
            </div>
          </>
        ) : (
          <>
            <div style={{marginTop: 10, fontSize: 12, color: '#94a3b8'}}>Year</div>
            <div style={{marginTop: 6, display: 'flex', alignItems: 'center', gap: 8}}>
              <button disabled={!arcYears.length} onClick={() => arcYears.length && setArcYearIndex(i => (i + arcYears.length - 1) % arcYears.length)} style={{padding: '4px 6px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.15)', background: 'transparent', color: '#cbd5e1', cursor: arcYears.length ? 'pointer' : 'not-allowed'}}>
                ‹
              </button>
              <div style={{flex: 1, textAlign: 'center', color: '#AFC7FF', fontSize: 14, fontWeight: 600}}>
                {arcSummary.year ?? '—'}
              </div>
              <button disabled={!arcYears.length} onClick={() => arcYears.length && setArcYearIndex(i => (i + 1) % arcYears.length)} style={{padding: '4px 6px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.15)', background: 'transparent', color: '#cbd5e1', cursor: arcYears.length ? 'pointer' : 'not-allowed'}}>
                ›
              </button>
            </div>
            <div style={{marginTop: 6, fontSize: 11, color: '#94a3b8'}}>Use ← → to change year</div>
            <div style={{marginTop: 10, fontSize: 12, color: '#94a3b8'}}>Min mention share</div>
            <input type="range" min="0" max="0.2" step="0.005" value={arcMinShare} onChange={e => setArcMinShare(Number(e.target.value))} style={{width: '100%'}} />
            <div style={{fontSize: 11, color: '#cbd5e1'}}>{(arcMinShare * 100).toFixed(1)}%</div>
            <div style={{marginTop: 10, fontSize: 12, color: '#94a3b8'}}>Max arcs</div>
            <input type="range" min="50" max={ARC_MAX_SEGMENTS} step="10" value={arcMaxVisible} onChange={e => setArcMaxVisible(Number(e.target.value))} style={{width: '100%'}} />
            <div style={{fontSize: 11, color: '#cbd5e1'}}>{arcSummary.shown.toLocaleString()} shown · {arcSummary.available.toLocaleString()} available</div>
          </>
        )}

        <button onClick={() => setPaused(p => !p)} style={{marginTop: 12, width: '100%', padding: '4px 8px', fontSize: 12, borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', color: paused ? '#0b1026' : '#cbd5e1', background: paused ? '#AFC7FF' : 'transparent', cursor: 'pointer'}}>
          {paused ? 'Play' : 'Pause'}
        </button>
        <div style={{marginTop: 6, fontSize: 11, color: '#94a3b8'}}>Shortcut: press “A” to toggle view</div>
      </div>
    </div>
  );
}
