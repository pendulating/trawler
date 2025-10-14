import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export default function ColorbarHUD({ leftPx, scaleMax = 0.15, dots = [], gradientStops, mode = 'fraction', transitionsEnabled = true }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      setWidth(el.getBoundingClientRect().width);
    });
    ro.observe(el);
    setWidth(el.getBoundingClientRect().width);
    return () => ro.disconnect();
  }, []);

  const ticks = useMemo(() => {
    if (mode === 'fraction') return [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15];
    // asymmetric for growth_pct: negative bound fixed at -1 (i.e., -100%), positives use dynamic scaleMax
    const negMin = -1;
    const posMax = Math.max(scaleMax, 0.001);
    return [negMin, negMin * 0.5, negMin * 0.25, 0, posMax * 0.25, posMax * 0.5, posMax];
  }, [mode, scaleMax]);

  const gradientCSS = useMemo(() => {
    // gradientStops: array of {stop: 0..1, color: [r,g,b]}
    const stops = gradientStops && gradientStops.length
      ? gradientStops
      : [
          { stop: 0.0,  color: [56, 189, 248] },
          { stop: 0.2,  color: [102, 126, 234] },
          { stop: 0.4,  color: [118, 75, 162] },
          { stop: 0.6,  color: [236, 72, 153] },
          { stop: 0.8,  color: [251, 146, 60] },
          { stop: 1.0,  color: [239, 68, 68] }
        ];
    const parts = stops.map(s => `rgb(${s.color[0]}, ${s.color[1]}, ${s.color[2]}) ${Math.round(s.stop * 100)}%`);
    return `linear-gradient(to right, ${parts.join(', ')})`;
  }, [gradientStops]);

  const barHeight = 96;
  const containerHeight = 128;
  const paddingX = 8;
  const innerWidth = Math.max(0, width - paddingX * 2);

  return createPortal(
    <div style={{ position: 'fixed', top: 12, left: Math.max(0, leftPx) + 'px', right: 12, height: containerHeight, pointerEvents: 'none', zIndex: 999998 }}>
      <div ref={containerRef} style={{ position: 'absolute', inset: 0 }}>
        {/* Gradient bar */}
        <div style={{ position: 'absolute', left: paddingX, right: paddingX, top: 0, height: barHeight, backgroundImage: gradientCSS, borderRadius: 8, border: '1px solid rgba(255,255,255,0.15)' }} />

        {/* Ticks */}
        {ticks.map(t => {
          let xn;
          if (mode === 'fraction') {
            xn = Math.min(1, Math.max(0, t / scaleMax));
          } else {
            const negMin = -1;
            const posMax = Math.max(scaleMax, 0.001);
            // piecewise map: [-1,0] -> [0,0.5], [0,posMax] -> [0.5,1]
            xn = t <= 0 ? 0.5 * (t - negMin) / (0 - negMin) : 0.5 + 0.5 * (t / posMax);
          }
          const x = paddingX + xn * innerWidth;
          return (
            <div key={`tick-${t}`} style={{ position: 'absolute', left: (x - 1) + 'px', top: barHeight, width: 2, height: 6, background: 'rgba(255,255,255,0.7)' }} />
          );
        })}
        {ticks.map(t => {
          let xn;
          if (mode === 'fraction') {
            xn = Math.min(1, Math.max(0, t / scaleMax));
          } else {
            const negMin = -1;
            const posMax = Math.max(scaleMax, 0.001);
            xn = t <= 0 ? 0.5 * (t - negMin) / (0 - negMin) : 0.5 + 0.5 * (t / posMax);
          }
          const x = paddingX + xn * innerWidth;
          const label = mode === 'fraction'
            ? (() => { const valPct = t * 100; return Number.isInteger(valPct) ? `${valPct}%` : `${valPct.toFixed(1)}%`; })()
            : (() => { const valPct = t * 100; return `${Math.round(valPct)}%`; })();
          return (
            <div key={`label-${t}`} style={{ position: 'absolute', left: Math.max(0, x - 12) + 'px', top: barHeight + 8, fontSize: 11, color: '#cbd5e1' }}>{label}</div>
          );
        })}

        {/* Dots with flag fill and gradient outline */}
        {dots.map(d => {
          const raw = d.value ?? d.fraction ?? 0;
          let xn;
          if (mode === 'fraction') {
            xn = Math.min(1, Math.max(0, raw / scaleMax));
          } else {
            const negMin = -1;
            const posMax = Math.max(scaleMax, 0.001);
            const v = Math.max(negMin, Math.min(posMax, raw));
            xn = v <= 0 ? 0.5 * (v - negMin) / (0 - negMin) : 0.5 + 0.5 * (v / posMax);
          }
          const x = paddingX + xn * innerWidth;
          // Scale jitter to bar height so dots spread slightly vertically
          const jitterScale = barHeight / 18; // ~2px per unit jitter when barHeight=96
          const y = Math.round(barHeight / 2 + ((d.jitterY || 0) * jitterScale));
          const flagUrl = d.iso2 ? `https://flagcdn.com/h20/${d.iso2}.png` : '';
          return (
            <div key={d.id} style={{ position: 'absolute', left: Math.max(0, x - 16) + 'px', top: Math.max(0, y - 16) + 'px', width: 32, height: 32, borderRadius: 9999, padding: 2,
              backgroundImage: 'linear-gradient(90deg, rgba(102,126,234,0.9), rgba(236,72,153,0.9))',
              transition: transitionsEnabled ? 'left 5000ms ease-in-out' : 'none' }} title={`${d.name || ''}: ${Number(d.total || 0).toLocaleString()}`}>
              <div style={{ width: '100%', height: '100%', borderRadius: 9999, overflow: 'hidden', backgroundColor: '#111' }}>
                {flagUrl ? (
                  <img src={flagUrl} alt={d.iso2} width={32} height={32} style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <div style={{ width: '100%', height: '100%', background: `rgb(${d.color?.[0] ?? 244}, ${d.color?.[1] ?? 63}, ${d.color?.[2] ?? 94})` }} />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>,
    document.body
  );
}


