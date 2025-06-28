# Frontend-Ready Data Instructions

## Overview

This document describes the data structures and formats available for the Coffee Flavor Analytics frontend interface. The analytics processing system provides comprehensive, multi-method analysis results that can be consumed by Streamlit components for interactive visualization and exploration.

## Main Data Access Function

Use the integrated analysis system to get all processed results:

```python
from analytics.processing.integrated_analysis import get_integrated_analysis

# This returns all analysis results with caching
all_results = get_integrated_analysis()
```

## Data Structure Overview

The `get_integrated_analysis()` function returns a dictionary with the following structure:

```python
{
    'data': {...},           # Raw and aggregated data from extraction
    'statistical': {...},    # Statistical significance analysis results
    'tfidf': {...},         # TF-IDF distinctiveness analysis results  
    'hierarchical': {...},  # Hierarchical pattern analysis results
    'consensus': {...},     # Cross-method validation and consensus findings
    'summary': {...}        # Executive summary and key insights
}
```

## Frontend Data Formats by Tab

### 📊 Overview Tab Data

**Key Metrics Display**
```python
summary = all_results['summary']

# Dataset overview metrics
dataset_stats = summary['dataset_overview']
# Contains: total_coffees, countries_analyzed, regions_analyzed, sellers_analyzed, flavor_parse_rate

# Top distinctive units for ranking
top_units = summary['top_distinctive_units']  # List of dicts with unit, type, score, coffee_count

# Key findings for insight cards
key_findings = summary['key_findings']  # List of finding summaries

# Unique discoveries
unique_discoveries = summary['unique_discoveries']  # List of unique flavor patterns
```

**Geographic Distribution Data**
```python
# Country-level aggregation for treemaps/bar charts
country_data = all_results['data']['country_aggregated']

# Create display-ready DataFrames
country_stats = []
for country, data in country_data.items():
    country_stats.append({
        'country': country,
        'total_coffees': data['metadata']['total_coffees'],
        'flavor_families': len(data['metadata']['unique_flavor_families']),
        'sellers': len(data['metadata']['unique_sellers']),
        'regions': len(data['metadata']['unique_subregions'])
    })

country_df = pd.DataFrame(country_stats)
```

### 🔍 Explore Tab Data

**Unit Selection and Details**
```python
# Get comprehensive profile for a selected unit
from analytics.processing.integrated_analysis import IntegratedFlavorAnalyzer

analyzer = all_results.get('analyzer') or IntegratedFlavorAnalyzer()
unit_profile = analyzer.get_unit_comprehensive_profile(unit_name, unit_type)

# Profile structure:
{
    'unit_name': str,
    'unit_type': str,
    'overview': {
        'total_coffees': int,
        'total_flavor_instances': int,
        'unique_flavor_families': List[str],
        # ... other metadata
    },
    'statistical_findings': {
        'family': [...],  # Statistically significant family-level flavors
        'genus': [...],   # Genus-level findings
        'species': [...]  # Species-level findings  
    },
    'tfidf_findings': {
        'family': {'top_flavors': [...], 'total_unique_flavors': int},
        # ... other levels
    },
    'hierarchical_findings': {
        'cascade_patterns': {...},
        'distinctiveness_types': {...},
        'summary_metrics': {...}
    },
    'consensus_findings': {
        'strong': [...],    # Flavors significant in all methods
        'moderate': [...]   # Flavors significant in 2/3 methods
    },
    'recommendations': [...]  # Actionable insights
}
```

**Statistical Results Display**
```python
# Get statistical significance results for a unit
stat_results = unit_profile['statistical_findings']

# Format for st.dataframe display
for level in ['family', 'genus', 'species']:
    if level in stat_results:
        df = pd.DataFrame(stat_results[level])
        # Contains: flavor, p_value_corrected, odds_ratio, cramers_v, interpretation
        st.dataframe(df[['flavor', 'odds_ratio', 'p_value_corrected', 'interpretation']])
```

### 🫘 By Flavor Tab Data

**Flavor Hierarchy and Selection**
```python
# Get all available flavors at each level
raw_data = all_results['data']['raw_merged_df']

# Extract flavor hierarchies
all_flavors = []
for _, row in raw_data.iterrows():
    if row['has_flavors']:
        all_flavors.extend(row['flavors_parsed'])

# Create hierarchy mappings
flavor_hierarchy = {}
families = set()
genera_by_family = defaultdict(set)
species_by_genus = defaultdict(set)

for flavor in all_flavors:
    family = flavor.get('family', '')
    genus = flavor.get('genus', '')
    species = flavor.get('species', '')
    
    if family:
        families.add(family)
        if genus:
            genera_by_family[family].add(genus)
            if species:
                species_by_genus[genus].add(species)

# Convert to lists for selectbox options
family_list = sorted(list(families))
genus_dict = {f: sorted(list(g)) for f, g in genera_by_family.items()}
species_dict = {g: sorted(list(s)) for g, s in species_by_genus.items()}
```

**Distinctive Regions for Flavor**
```python
# Get regions distinctive for a selected flavor using TF-IDF scores
def get_regions_for_flavor(flavor_name, taxonomy_level):
    level_key = f'{taxonomy_level}_regions_scores'
    if level_key in all_results['tfidf']:
        tfidf_df = all_results['tfidf'][level_key]
        flavor_df = tfidf_df[tfidf_df['flavor'] == flavor_name]
        return flavor_df.nlargest(15, 'tfidf_score')
    return pd.DataFrame()
```

### ⚖️ Compare Tab Data

**Multi-Unit Comparison**
```python
# Compare multiple units
def build_comparison_data(selected_entities, entity_type):
    comparison_data = []
    
    for entity in selected_entities:
        profile = analyzer.get_unit_comprehensive_profile(entity, entity_type)
        
        # Extract key metrics
        entity_metrics = {
            'entity': entity,
            'total_coffees': profile['overview'].get('total_coffees', 0),
            'distinctive_flavors': len(profile['consensus_findings']['strong']),
            'top_family': None,
            'tfidf_score': 0.0
        }
        
        # Get top distinctive flavor
        if profile['tfidf_findings'].get('family', {}).get('top_flavors'):
            top = profile['tfidf_findings']['family']['top_flavors'][0]
            entity_metrics['top_family'] = top['flavor']
            entity_metrics['tfidf_score'] = top['tfidf_score']
        
        comparison_data.append(entity_metrics)
    
    return pd.DataFrame(comparison_data)
```

**Similarity Analysis**
```python
# Get similarity between units using TF-IDF cosine similarity
tfidf_analyzer = all_results['tfidf']['analyzer']
similarity_matrix = tfidf_analyzer.calculate_similarity_matrix('family', entity_type)

# Convert to plotly heatmap format
fig = px.imshow(
    similarity_matrix,
    title=f"Flavor Similarity Between {entity_type.title()}s",
    color_continuous_scale='viridis'
)
```

### 🏆 Rankings Tab Data

**Leaderboard Data**
```python
# Get rankings by different criteria
def get_rankings(ranking_category, taxonomy_level, min_coffees):
    if ranking_category == "Most Distinctive Overall":
        # Use distinctiveness scores from summary
        rankings = all_results['summary']['top_distinctive_units']
        
    elif ranking_category == "Most Specialized":
        # Use hierarchical concentration scores
        rankings = []
        for unit_name, profile in all_results['hierarchical']['profiles'].items():
            rankings.append({
                'entity_name': unit_name,
                'unit_type': profile['unit_type'], 
                'score': profile['summary_metrics']['concentration_index'],
                'total_coffees': profile['total_coffees']
            })
        
    elif ranking_category == "Most Diverse":
        # Use flavor diversity scores
        rankings = []
        for unit_name, profile in all_results['hierarchical']['profiles'].items():
            rankings.append({
                'entity_name': unit_name,
                'unit_type': profile['unit_type'],
                'score': profile['summary_metrics']['flavor_diversity'],
                'total_coffees': profile['total_coffees']
            })
    
    # Filter by minimum coffees and sort
    rankings_df = pd.DataFrame(rankings)
    filtered = rankings_df[rankings_df['total_coffees'] >= min_coffees]
    return filtered.sort_values('score', ascending=False)
```

## Data Export and Download Functionality

**CSV Export Preparation**
```python
# Prepare data for CSV download
def prepare_export_data(unit_name, unit_type):
    profile = analyzer.get_unit_comprehensive_profile(unit_name, unit_type)
    
    # Flatten statistical findings
    export_rows = []
    for level, findings in profile['statistical_findings'].items():
        for finding in findings:
            export_rows.append({
                'taxonomy_level': level,
                'flavor': finding['flavor'],
                'method': 'statistical',
                'p_value': finding['p_value_corrected'],
                'odds_ratio': finding['odds_ratio'],
                'effect_size': finding['cramers_v']
            })
    
    # Add TF-IDF findings
    for level, findings in profile['tfidf_findings'].items():
        if 'top_flavors' in findings:
            for flavor_data in findings['top_flavors']:
                export_rows.append({
                    'taxonomy_level': level,
                    'flavor': flavor_data['flavor'],
                    'method': 'tfidf',
                    'tfidf_score': flavor_data['tfidf_score'],
                    'tf_score': flavor_data['tf_score']
                })
    
    return pd.DataFrame(export_rows)

# Convert to CSV for download
csv = export_df.to_csv(index=False)
st.download_button(
    label="Download Analysis Results",
    data=csv,
    file_name=f"{unit_name}_flavor_analysis.csv",
    mime="text/csv"
)
```

## Performance Considerations

1. **Caching**: All major analysis functions use `@st.cache_data(ttl=3600)` for 1-hour caching
2. **Lazy Loading**: Only compute detailed unit profiles when requested
3. **Data Filtering**: Pre-filter data by minimum occurrence thresholds to reduce computation
4. **Progressive Disclosure**: Show summary data first, detailed analysis on demand

## Error Handling

```python
# Always check for data availability
if not all_results or 'data' not in all_results:
    st.error("Analysis data not available. Please check database connection.")
    return

# Handle missing units gracefully  
if unit_name not in available_units:
    st.warning(f"Unit '{unit_name}' not found in analysis results.")
    return

# Handle empty results
if results_df.empty:
    st.info("No significant results found for the selected criteria.")
```

## Visualization Data Formats

**For Plotly Charts**
- Bar charts: Use DataFrames with x/y columns
- Heatmaps: Use similarity matrices or pivot tables
- Treemaps: Use hierarchical dictionaries with 'labels', 'parents', 'values'
- Scatter plots: Use DataFrames with x/y/color/size columns

**For Streamlit Native Components**
- Metrics: Single values with delta calculations
- DataFrames: Use column_config for formatting
- Progress bars: Normalized 0-1 values
- Maps: Requires lat/lon coordinates (not currently available)

This frontend data structure provides comprehensive access to all analysis results while maintaining good performance through caching and progressive data loading.