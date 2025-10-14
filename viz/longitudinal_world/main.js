import { Deck, COORDINATE_SYSTEM, _GlobeView as GlobeView } from '@deck.gl/core';
import { ScatterplotLayer, SolidPolygonLayer, GeoJsonLayer } from '@deck.gl/layers';

// Barebones demo + land outlines
let landData = null;
const WORLD_LOCAL_URL = '/world_countries.geojson';
const WORLD_REMOTE_URL = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson';
let aiData = null;
const AI_URL = '/country_data.geojson';
let TIME_PERIODS = ['2015-2017', '2018-2020', '2021-2023'];
let currentPeriodIndex = 0;
const FIXED_ZOOM = 0.3;
let controlledViewState = {
    globe: {
        longitude: 0,
        latitude: 0,
        zoom: FIXED_ZOOM,
        minZoom: FIXED_ZOOM,
        maxZoom: FIXED_ZOOM,
        pitch: 0,
        bearing: 0
    }
};

// Color scale from low to high AI fraction (continuous ramp)
// No color scaling needed in barebones demo

// Get elevation (height) based on AI fraction
// No elevation needed

// No AI fraction in barebones demo

// Create deck.gl layer
// Minimal debug layer: show equator and prime meridian crosshair
function createDebugPoints() {
    return new ScatterplotLayer({
        id: 'debug-points',
        data: [
            { position: [0, 0], color: [255, 200, 0] }
        ],
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        getPosition: d => d.position,
        getFillColor: d => d.color,
        filled: true,
        stroked: true,
        getLineColor: [255, 255, 255],
        lineWidthMinPixels: 2,
        radiusUnits: 'meters',
        getRadius: 1200000,
        radiusMinPixels: 12,
        radiusMaxPixels: 64,
        pickable: false
    });
}

// Create a globe background so the sphere is visible
function createBackgroundLayer() {
    return new SolidPolygonLayer({
        id: 'earth-background',
        data: [
            [[-180, 90], [0, 90], [180, 90], [180, -90], [0, -90], [-180, -90], [-180, 90]]
        ],
        getPolygon: d => d,
        stroked: false,
        filled: true,
        pickable: false,
        opacity: 1,
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        parameters: { depthTest: true, cull: false },
        getFillColor: [40, 70, 160]
    });
}

// Render land polygons (continents) for visual reference
function createLandLayer() {
    if (!landData) return null;
    const data = landData.type === 'FeatureCollection' ? landData.features : landData;
    return new GeoJsonLayer({
        id: 'land',
        data,
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        filled: true,
        stroked: true,
        getFillColor: [180, 205, 170, 255], // light greenish land
        getLineColor: [240, 250, 255, 180], // subtle outline
        lineWidthMinPixels: 1,
        pickable: false
    });
}

// Color scale for AI fraction
function getAIColor(fraction) {
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
}

function getAIFractionForCurrentPeriod(props) {
    const periodKey = TIME_PERIODS[currentPeriodIndex].replace('-', '_');
    return props[`ai_fraction_${periodKey}`] || 0;
}

function createAILayer() {
    if (!aiData) return null;
    const data = aiData.type === 'FeatureCollection' ? aiData.features : aiData;
    return new GeoJsonLayer({
        id: 'ai-countries',
        data,
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        filled: true,
        extruded: true,
        stroked: true,
        getFillColor: f => getAIColor(getAIFractionForCurrentPeriod(f.properties)),
        getLineColor: [40, 60, 120, 180],
        lineWidthMinPixels: 1,
        getElevation: f => {
            const frac = getAIFractionForCurrentPeriod(f.properties);
            const capped = Math.min(frac, 0.15);
            return (capped / 0.15) * 1200000;
        },
        updateTriggers: {
            getFillColor: [currentPeriodIndex],
            getElevation: [currentPeriodIndex]
        },
        pickable: true
    });
}

// Deck.gl instance (will be initialized after DOM is ready)
let deck = null;

// Initialize deck.gl
function initDeck() {
    const mapContainer = document.getElementById('map');

    deck = new Deck({
        parent: mapContainer,
        width: '100%',
        height: '100%',
        initialViewState: {
            longitude: controlledViewState.globe.longitude,
            latitude: controlledViewState.globe.latitude,
            zoom: controlledViewState.globe.zoom,
            minZoom: controlledViewState.globe.minZoom,
            maxZoom: controlledViewState.globe.maxZoom,
            pitch: controlledViewState.globe.pitch,
            bearing: controlledViewState.globe.bearing
        },
        viewState: controlledViewState,
        useDevicePixels: window.devicePixelRatio || 1,
        
        // Styling
        parameters: {
            clearColor: [0.05, 0.08, 0.16, 1],
            depthTest: true,
            depthMask: true
        },
        
        // Map view
        views: [
            new GlobeView({
                id: 'globe',
                resolution: 5,
                controller: {
                    dragPan: true, // use pan gestures to change longitude
                    scrollZoom: false,
                    touchZoom: false,
                    doubleClickZoom: false,
                    keyboard: false,
                    inertia: 0
                }
            })
        ],
        onViewStateChange: ({ viewState }) => {
            const next = viewState.globe ? viewState.globe : viewState;
            controlledViewState = {
                globe: {
                    longitude: next.longitude,
                    latitude: 0, // lock lat
                    zoom: FIXED_ZOOM,
                    minZoom: FIXED_ZOOM,
                    maxZoom: FIXED_ZOOM,
                    pitch: 0,
                    bearing: 0
                }
            };
            deck.setProps({ viewState: controlledViewState });
        },
        layers: [createBackgroundLayer(), createLandLayer(), createAILayer(), createDebugPoints()].filter(Boolean),
        onAfterRender: () => {
            const canvas = mapContainer.querySelector('canvas');
            if (canvas && !canvas.__logged) {
                console.info('Deck canvas size', canvas.width, canvas.height);
                canvas.__logged = true;
            }
        },

        // Tooltip
        getTooltip: ({ object, layer }) => {
            if (!object || layer?.id !== 'ai-countries') return null;
            const props = object.properties || {};
            const period = TIME_PERIODS[currentPeriodIndex] || '';
            const periodKey = period.replace('-', '_');
            const name = props.country_name || props.ADMIN || props.NAME || 'Unknown';
            const total = props[`total_${periodKey}`] || 0;
            return {
                html: `
                    <div class="tooltip-country">${name}</div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">Period:</span>
                        <span class="tooltip-value">${period}</span>
                    </div>
                    <div class="tooltip-row">
                        <span class="tooltip-label">Articles:</span>
                        <span class="tooltip-value">${Number(total).toLocaleString()}</span>
                    </div>
                `,
                style: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderRadius: '8px',
                    padding: '10px 12px',
                    fontSize: '12px'
                }
            };
        }
    });

    // Expose for debugging
    try { window.__deck = deck; } catch (_) {}

    // Ensure globe background is visible immediately before data loads
    deck.setProps({ layers: [createBackgroundLayer(), createLandLayer(), createAILayer(), createDebugPoints()].filter(Boolean) });
}

// Update visualization
function updateVisualization() {
    if (!deck) return;
    deck.setProps({ layers: [createBackgroundLayer(), createLandLayer(), createAILayer(), createDebugPoints()].filter(Boolean) });
}

// Load data and initialize
async function init() {
    // Initialize deck.gl immediately
    initDeck();
    // Fetch land polygons: prefer local, fallback to remote
    try {
        let response = await fetch(WORLD_LOCAL_URL);
        if (!response.ok) {
            response = await fetch(WORLD_REMOTE_URL);
        }
        landData = await response.json();
        updateVisualization();
    } catch (e) {
        console.error('Failed to load land data', e);
    }

    // Load AI data
    try {
        const aiResp = await fetch(AI_URL);
        if (aiResp.ok) {
            aiData = await aiResp.json();
            // Derive available periods dynamically from properties
            try {
                const feats = aiData.type === 'FeatureCollection' ? aiData.features : aiData;
                const sample = feats.find(f => f && f.properties && Object.keys(f.properties).some(k => k.startsWith('ai_fraction_')));
                if (sample) {
                    TIME_PERIODS = Object.keys(sample.properties)
                        .filter(k => k.startsWith('ai_fraction_'))
                        .map(k => k.replace('ai_fraction_', '').replace(/_/g, '-'))
                        .sort((a, b) => parseInt(a.split('-')[0]) - parseInt(b.split('-')[0]));
                    currentPeriodIndex = 0;
                }
            } catch (_) {}
            updateVisualization();
        }
    } catch (e) {
        console.error('Failed to load AI data', e);
    }

    // Minimal keyboard controls for period
    window.addEventListener('keydown', (e) => {
        if (!TIME_PERIODS || TIME_PERIODS.length === 0) return;
        if (e.key === 'ArrowRight') {
            currentPeriodIndex = (currentPeriodIndex + 1) % TIME_PERIODS.length;
            updateVisualization();
        } else if (e.key === 'ArrowLeft') {
            currentPeriodIndex = (currentPeriodIndex + TIME_PERIODS.length - 1) % TIME_PERIODS.length;
            updateVisualization();
        }
    });
}

// Time slider event listener
// No controls in barebones demo

// Handle window resize
window.addEventListener('resize', () => {
    if (!deck) return;
    deck.setProps({ width: '100%', height: '100%' });
    deck.redraw(true);
});

// Start the app
init();

