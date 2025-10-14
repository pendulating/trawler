import React, { useMemo, useEffect, useState, useCallback, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { _GlobeView as GlobeView, COORDINATE_SYSTEM } from '@deck.gl/core';
import { SolidPolygonLayer, GeoJsonLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import Overlays from './Overlays.jsx';

const WORLD_LOCAL_URL = '/world_countries.geojson';
const WORLD_REMOTE_URL = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson';
const AI_URL = '/country_data.geojson';

const FIXED_ZOOM = 2.4;
const ROTATE_DEG_PER_SEC = 6; // rotation speed

function normalizeLongitude(lon) {
  let x = lon;
  while (x > 180) x -= 360;
  while (x < -180) x += 360;
  return x;
}

export default function App() {
  const [landData, setLandData] = useState(null);
  const [aiData, setAiData] = useState(null);
  const [periods, setPeriods] = useState([]);
  const [periodIndex, setPeriodIndex] = useState(0);
  const [autoAdvance, setAutoAdvance] = useState(true);
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

  const initialViewState = useMemo(() => ({
    longitude: 0,
    latitude: 0,
    zoom: FIXED_ZOOM,
    minZoom: FIXED_ZOOM,
    maxZoom: FIXED_ZOOM,
    pitch: 0,
    bearing: 0
  }), []);

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
            }
          }
        }
      } catch (e) {
        console.error('Failed loading AI data', e);
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

  const getAIFraction = useCallback((props) => {
    if (!periods.length) return 0;
    const key = `ai_fraction_${periods[periodIndex].replace('-', '_')}`;
    return props[key] || 0;
  }, [periods, periodIndex]);

  const getAIColor = useCallback((fraction) => {
    if (!fraction || isNaN(fraction)) return [140, 160, 200, 160];
    const t = Math.min(fraction / 0.15, 1);
    const a = [102,126,234], b = [118,75,162], c = [236,72,153];
    const c0 = t < 0.5 ? a : b;
    const c1 = t < 0.5 ? b : c;
    const local = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
    return [
      Math.round(c0[0] + (c1[0] - c0[0]) * local),
      Math.round(c0[1] + (c1[1] - c0[1]) * local),
      Math.round(c0[2] + (c1[2] - c0[2]) * local),
      220
    ];
  }, []);

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
      getFillColor: f => getAIColor(getAIFraction(f.properties)),
      getLineColor: [40, 60, 120, 180],
      lineWidthMinPixels: 1,
      getElevation: f => {
        const frac = getAIFraction(f.properties);
        const capped = Math.min(frac, 0.15);
        return (capped / 0.15) * 1200000;
      },
      updateTriggers: {
        getFillColor: [periodIndex],
        getElevation: [periodIndex]
      },
      transitions: {
        getFillColor: { duration: 5000 },
        getElevation: { duration: 5000 }
      },
      pickable: true
    });
  }, [aiData, getAIFraction, getAIColor, periodIndex]);

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

  const layers = useMemo(() => [background, land, aiLayer, hudLayer, hudTextLayer].filter(Boolean), [background, land, aiLayer, hudLayer, hudTextLayer]);

  const updateCameraOverlay = useCallback(() => {
    const now = performance.now();
    if (now - lastCamOverlayTsRef.current < 100) return; // throttle ~10 fps
    lastCamOverlayTsRef.current = now;
    const deck = deckRef.current && deckRef.current.deck;
    if (!deck) return;
    const viewports = (typeof deck.getViewports === 'function') ? deck.getViewports() : [];
    const vp = Array.isArray(viewports) && viewports.length ? (viewports.find(v => v && v.id === 'globe') || viewports[0]) : null;
    if (!vp || typeof vp.project !== 'function') return;
    const xy = vp.project([rotationLonRef.current, 0]);
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
          latitude: 0,
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
      rotationLonRef.current = normalizeLongitude(rotationLonRef.current + ROTATE_DEG_PER_SEC * dt);
      const deck = deckRef.current && deckRef.current.deck;
      if (readyRef.current && deck) {
        deck.setProps({
          viewState: {
            longitude: rotationLonRef.current,
            latitude: 0,
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
  }, [updateCameraOverlay]);

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
      if (!periods.length) return;
      if (e.key === 'ArrowRight') setPeriodIndex(i => (i + 1) % periods.length);
      if (e.key === 'ArrowLeft') setPeriodIndex(i => (i + periods.length - 1) % periods.length);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [periods]);

  useEffect(() => {
    if (!autoAdvance || periods.length <= 1) return;
    const next = () => setPeriodIndex(i => (i + 1) % periods.length);
    let intervalId = null;
    const initialTimeout = setTimeout(() => {
      next();
      intervalId = setInterval(next, 10000);
    }, 5000);
    return () => {
      clearTimeout(initialTimeout);
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoAdvance, periods]);

  const isFrontFacing = useCallback((lonDeg, latDeg) => {
    const lon = (lonDeg * Math.PI) / 180;
    const lat = (latDeg * Math.PI) / 180;
    const camLon = (rotationLonRef.current * Math.PI) / 180;
    const dot = Math.cos(lat) * Math.cos(lon - camLon);
    return dot > 0;
  }, []);

  useEffect(() => {
    if (!aiData || !webglReady) return;
    const deck = deckRef.current && deckRef.current.deck;
    if (!deck) return;
    let cleared = false;
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
        candidates.push({ lon: lx, lat: ly, name, total });
        if (candidates.length > 2000) break;
      }
      const picks = [];
      const n = Math.min(5, candidates.length);
      for (let k = 0; k < n; k++) {
        const idxP = (Math.random() * candidates.length) | 0;
        picks.push(candidates[idxP]);
        candidates.splice(idxP, 1);
      }
      const baseMarkers = picks.map(c => ({ position: [c.lon, c.lat + 2], name: c.name, total: c.total, period, opacity: 0 }));
      setHudMarkers(baseMarkers);
      // fade in
      const fadeIn = setTimeout(() => {
        setHudMarkers(ms => ms.map(m => ({ ...m, opacity: 1 })));
      }, 0);
      // stay visible then fade out
      const displayMs = 7000;
      const fadeMs = 400;
      const fadeOut = setTimeout(() => {
        setHudMarkers(ms => ms.map(m => ({ ...m, opacity: 0 })));
      }, displayMs);
      const clearId = setTimeout(() => {
        if (!cleared) setHudMarkers([]);
      }, displayMs + fadeMs + 50);
      return () => { cleared = true; clearTimeout(fadeIn); clearTimeout(fadeOut); clearTimeout(clearId); };
    } catch (_) {
      setHudMarkers([]);
    }
  }, [aiData, webglReady, periods, periodIndex, isFrontFacing]);

  const onHover = useCallback((info) => {
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
  }, [periods, periodIndex]);

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
      <div style={{position: 'absolute', top: 12, left: 12, background: 'rgba(15,23,42,0.8)', padding: '8px 12px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)'}}>
        <div style={{fontSize: 12, color: '#94a3b8'}}>Period</div>
        <div style={{fontSize: 14, fontWeight: 600, color: '#AFC7FF'}}>{periods[periodIndex] || '—'}</div>
        <div style={{marginTop: 6, fontSize: 11, color: '#94a3b8'}}>Use ← → to change period</div>
        <label style={{display: 'flex', alignItems: 'center', gap: 8, marginTop: 8, fontSize: 12, color: '#cbd5e1', cursor: 'pointer'}}>
          <input type="checkbox" checked={autoAdvance} onChange={e => setAutoAdvance(e.target.checked)} />
          Auto-advance (5s)
        </label>
      </div>
    </div>
  );
}
