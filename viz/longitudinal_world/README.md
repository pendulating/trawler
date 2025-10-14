# Longitudinal AI News Coverage - 3D World Visualization

An interactive 3D globe visualization showing the fraction of AI-relevant news articles by country over time using deck.gl.

## Overview

This visualization displays countries as 3D extruded polygons on a globe, where:
- **Height** represents the fraction of AI-relevant articles
- **Color** represents the same metric (gradient)
- **Time slider** allows exploration across different time periods (3-year bins)

## Structure

```
longitudinal_world/
├── prepare_geo_data.py    # Python script to prepare GeoJSON data
├── country_data.geojson   # Generated GeoJSON file (not committed)
├── package.json           # Node.js dependencies
├── index.html            # Main HTML file
├── main.js               # Deck.gl visualization code
└── README.md             # This file
```

## Setup

### Step 1: Prepare Data (Python)

```bash
# Install required Python packages
pip install pandas geopandas requests

# Run the data preparation script
python prepare_geo_data.py
```

This will:
1. Load article classification data
2. Calculate AI-relevant fractions per country per time period
3. Download country geometries from Natural Earth
4. Merge and export to `country_data.geojson`

### Step 2: Run Visualization (Node.js)

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

The visualization will be available at `http://localhost:8924`

## Data Processing

- **Minimum articles**: Countries must have at least 100 articles to be included
- **Time bins**: Articles are grouped into 3-year periods (2015-2017, 2018-2020, etc.)
- **Calculation**: For each country and time period, fraction = (AI-relevant articles) / (total articles)

## Dependencies

### Python
- pandas
- requests

### JavaScript
- vite (build tool)
- deck.gl (3D visualization)
- @deck.gl/layers
- @deck.gl/geo-layers
- @loaders.gl/json

## Usage

Once running:
1. Use mouse to rotate the globe
2. Hover over countries to see detailed statistics
3. Use the time slider to see changes over different periods
4. Zoom in/out with scroll wheel

## Notes

- The visualization uses Natural Earth's low-resolution country boundaries for performance
- Country codes follow ISO 3166-1 alpha-2 standard
- Only countries with sufficient article data are included

