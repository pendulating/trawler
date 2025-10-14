#!/usr/bin/env python3
"""
Prepare geospatial data for deck.gl visualization.

This script:
1. Loads article classification data
2. Calculates AI-relevant article fractions per country per time period
3. Merges with country GeoJSON geometries
4. Exports to GeoJSON for deck.gl visualization
"""

import pandas as pd
import json
import requests
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Paths
DATA_PATH = PROJECT_ROOT / "outputs/classify/oct14_classify_all.parquet"
OUTPUT_PATH = Path(__file__).parent / "public" / "country_data.geojson"
WORLD_OUTPUT_PATH = Path(__file__).parent / "public" / "world_countries.geojson"

# ISO 3166-1 alpha-2 country code mapping
COUNTRY_CODE_MAP = {
    'ar': 'AR', 'au': 'AU', 'at': 'AT', 'bd': 'BD', 'be': 'BE', 'bg': 'BG',
    'br': 'BR', 'ca': 'CA', 'ch': 'CH', 'cl': 'CL', 'cn': 'CN', 'co': 'CO',
    'cz': 'CZ', 'dk': 'DK', 'ee': 'EE', 'eg': 'EG', 'et': 'ET', 'fi': 'FI',
    'fr': 'FR', 'de': 'DE', 'gh': 'GH', 'gr': 'GR', 'hk': 'HK', 'hr': 'HR',
    'hu': 'HU', 'id': 'ID', 'ie': 'IE', 'il': 'IL', 'in': 'IN', 'ir': 'IR',
    'it': 'IT', 'jp': 'JP', 'ke': 'KE', 'kr': 'KR', 'lt': 'LT', 'lu': 'LU',
    'lv': 'LV', 'ma': 'MA', 'mt': 'MT', 'mx': 'MX', 'my': 'MY', 'nl': 'NL',
    'ng': 'NG', 'no': 'NO', 'nz': 'NZ', 'pe': 'PE', 'ph': 'PH', 'pk': 'PK',
    'pl': 'PL', 'pt': 'PT', 'ro': 'RO', 'rs': 'RS', 'ru': 'RU', 'sa': 'SA',
    'se': 'SE', 'sg': 'SG', 'si': 'SI', 'sv': 'SV', 'es': 'ES', 'th': 'TH',
    'tr': 'TR', 'tt': 'TT', 'tw': 'TW', 'tz': 'TZ', 'ae': 'AE', 'gb': 'GB',
    'us': 'US', 'uy': 'UY', 've': 'VE', 'vn': 'VN', 'za': 'ZA'
}


def create_year_bin(year):
    """Create 3-year bins starting from 2015."""
    year = int(year)
    start_year = ((year - 2015) // 3) * 3 + 2015
    end_year = start_year + 2
    return f"{start_year}-{end_year}"


def calculate_country_stats(df, min_articles=100):
    """Calculate AI-relevant article fractions per country per time period."""
    print("Calculating country statistics...")
    
    # Add year bins
    df['year_bin'] = df['year'].apply(create_year_bin)
    
    # Filter countries with minimum article count
    country_counts = df['country'].value_counts()
    valid_countries = country_counts[country_counts >= min_articles].index.tolist()
    df = df[df['country'].isin(valid_countries)]
    
    print(f"Processing {len(valid_countries)} countries with at least {min_articles} articles")
    
    # Get unique year bins
    year_bins = sorted(df['year_bin'].unique())
    
    # Calculate statistics per country and year bin
    stats = []
    for country in valid_countries:
        country_data = df[df['country'] == country]
        country_iso = COUNTRY_CODE_MAP.get(country.lower())
        
        if not country_iso:
            print(f"Warning: No ISO code mapping for {country}, skipping...")
            continue
        
        stat_entry = {
            'country_code': country.lower(),
            'country_iso': country_iso,
            'country_name': country.upper(),
            'total_articles': len(country_data)
        }
        
        # Calculate fraction for each year bin
        for year_bin in year_bins:
            bin_data = country_data[country_data['year_bin'] == year_bin]
            if len(bin_data) > 0:
                total = len(bin_data)
                relevant = len(bin_data[bin_data['is_relevant'] == True])
                fraction = relevant / total
                # Create safe property name
                bin_key = year_bin.replace('-', '_')
                stat_entry[f'ai_fraction_{bin_key}'] = round(fraction, 4)
                stat_entry[f'total_{bin_key}'] = total
                stat_entry[f'relevant_{bin_key}'] = relevant
        
        stats.append(stat_entry)
    
    return pd.DataFrame(stats), year_bins


def load_country_geometries():
    """Load country geometries from Natural Earth GeoJSON."""
    print("Loading country geometries from Natural Earth...")
    
    # Use Natural Earth's low resolution country boundaries
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error loading country geometries: {e}")
        print("Attempting to load from backup source...")
        
        # Backup source
        url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()


def merge_data_with_geometries(stats_df, geo_data):
    """Merge statistics with country geometries."""
    print("Merging statistics with geometries...")
    
    features = []
    matched_countries = 0
    
    for feature in geo_data['features']:
        properties = feature['properties']
        
        # Try to match by ISO code (most reliable)
        iso_a2 = properties.get('ISO_A2') or properties.get('iso_a2') or properties.get('ISO_A2_EH')
        
        if iso_a2 and iso_a2 != '-99':
            # Find matching country in stats
            matching_stats = stats_df[stats_df['country_iso'] == iso_a2]
            
            if not matching_stats.empty:
                stat = matching_stats.iloc[0].to_dict()
                
                # Merge properties
                new_properties = {
                    **properties,
                    **stat
                }
                
                features.append({
                    'type': 'Feature',
                    'properties': new_properties,
                    'geometry': feature['geometry']
                })
                matched_countries += 1
    
    print(f"Successfully matched {matched_countries} countries with geometries")
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def main():
    """Main execution function."""
    print("=" * 60)
    print("Preparing Geospatial Data for Deck.gl Visualization")
    print("=" * 60)
    
    # Load article data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} articles")
    
    # Calculate statistics
    stats_df, year_bins = calculate_country_stats(df)
    print(f"Calculated statistics for {len(stats_df)} countries across {len(year_bins)} time periods")
    print(f"Time periods: {', '.join(year_bins)}")
    
    # Load geometries
    geo_data = load_country_geometries()
    print(f"Loaded {len(geo_data['features'])} country geometries")
    
    # Merge data
    merged_data = merge_data_with_geometries(stats_df, geo_data)
    
    # Export to GeoJSON
    print(f"\nExporting to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(merged_data, f, indent=2)
    # Also export full world countries baseline for visualization completeness
    try:
        print(f"Exporting world countries to {WORLD_OUTPUT_PATH}...")
        with open(WORLD_OUTPUT_PATH, 'w') as wf:
            json.dump(geo_data, wf)
    except Exception as e:
        print(f"Warning: failed to export world countries baseline: {e}")
    
    print(f"✓ Successfully exported {len(merged_data['features'])} countries to GeoJSON")
    print(f"✓ File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Countries with data: {len(merged_data['features'])}")
    print(f"Time periods: {len(year_bins)}")
    
    # Sample data for first country
    if merged_data['features']:
        sample = merged_data['features'][0]['properties']
        print(f"\nSample data structure (first country: {sample.get('country_name', 'Unknown')}):")
        for key in sorted(sample.keys()):
            if key.startswith('ai_fraction_'):
                print(f"  {key}: {sample[key]}")
    
    print("\n✓ Data preparation complete!")


if __name__ == "__main__":
    main()

