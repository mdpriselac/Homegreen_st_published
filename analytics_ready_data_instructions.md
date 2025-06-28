# Analytics-Ready Data Instructions

## Overview

This document describes the data structures and formats output by the Coffee Data Extraction system (`analytics/db_access/coffee_data_extractor.py`). The system extracts coffee flavor data from Supabase and prepares it in multiple formats optimized for different types of analytics.

## Data Extraction Process

### 1. Raw Data Sources
The system pulls data from two main database sources:
- **coffee_attributes table**: Filtered for `is_cleaned = true`, providing clean country/region names and categorized flavor data
- **coffees + sellers tables**: Providing coffee names and seller information

### 2. Key Data Statistics (as of latest extraction)
- Total coffee records: ~1,700
- Successful flavor parsing rate: 99.6%
- Geographic coverage: 34 countries, 161 regions
- Seller coverage: 19 unique sellers
- Missing data: <3% missing countries, <1% missing flavors

## Output Data Structures

The `get_analytics_data()` function returns a dictionary containing all prepared data formats:

```python
{
    'raw_merged_df': pd.DataFrame,           # Raw merged dataset
    'country_aggregated': Dict,              # Country-level aggregations
    'region_aggregated': Dict,               # Region-level aggregations
    'seller_aggregated': Dict,               # Seller-level aggregations
    'contingency_format': pd.DataFrame,      # Statistical analysis format
    'tfidf_format': Dict,                    # TF-IDF analysis format
    'hierarchical_format': Dict,             # Hierarchical analysis format
    'extraction_timestamp': str              # ISO timestamp of extraction
}
```

### 1. Raw Merged DataFrame (`raw_merged_df`)

A pandas DataFrame with one row per coffee containing:
- `coffee_id`: Unique identifier
- `coffee_name`: Coffee name
- `country_final`: Clean country name
- `subregion_final`: Clean subregion name
- `seller_id`: Seller identifier
- `seller_name`: Seller name
- `categorized_flavors`: Original JSON flavor data
- `flavors_parsed`: Parsed list of flavor dictionaries
- `has_flavors`: Boolean indicating if flavors exist
- `flavor_count`: Number of flavor descriptors
- `region_key`: Combined "Country_Subregion" key

### 2. Country Aggregated Data (`country_aggregated`)

Dictionary structure:
```python
{
    "Ethiopia": {
        "coffees": [
            {
                "coffee_id": 123,
                "coffee_name": "Yirgacheffe Natural",
                "seller_name": "Blue Bottle Coffee",
                "subregion": "Sidamo",
                "flavors": [
                    {"family": "Fruity", "genus": "Berry", "species": "Blueberry"},
                    {"family": "Floral", "genus": "Floral", "species": "Jasmine"}
                ]
            }
            # ... more coffees
        ],
        "all_flavors": [  # All flavor instances from all coffees
            {"family": "Fruity", "genus": "Berry", "species": "Blueberry"},
            # ... includes duplicates
        ],
        "metadata": {
            "total_coffees": 45,
            "unique_subregions": ["Sidamo", "Harrar", "Kaffa"],
            "unique_sellers": ["Blue Bottle", "Intelligentsia"],
            "total_flavor_instances": 127,
            "unique_flavor_families": ["Fruity", "Floral", "Nutty/Cocoa"],
            "unique_flavor_genera": ["Berry", "Citrus Fruit", "Floral"],
            "unique_flavor_species": ["Blueberry", "Orange", "Jasmine"]
        }
    }
    # ... other countries
}
```

### 3. Region Aggregated Data (`region_aggregated`)

Similar structure to country data, but keyed by "Country_Subregion":
```python
{
    "Ethiopia_Sidamo": {
        "country": "Ethiopia",
        "subregion": "Sidamo",
        "coffees": [...],
        "all_flavors": [...],
        "metadata": {...}
    }
    # ... other regions
}
```

### 4. Seller Aggregated Data (`seller_aggregated`)

Similar structure, but includes geographic distribution:
```python
{
    "Blue Bottle Coffee": {
        "seller_id": 5,
        "coffees": [...],  # Includes country/subregion for each coffee
        "all_flavors": [...],
        "metadata": {
            "total_coffees": 23,
            "unique_countries": ["Ethiopia", "Colombia", "Guatemala"],
            "unique_subregions": ["Sidamo", "Huila", "Antigua"],
            # ... flavor statistics
        }
    }
    # ... other sellers
}
```

### 5. Contingency Format DataFrame (`contingency_format`)

A pandas DataFrame optimized for statistical analysis with columns:
- `unit`: Name of the geographic unit or seller
- `unit_type`: One of "country", "region", or "seller"
- `total_coffees`: Number of coffees in this unit
- `total_flavor_instances`: Total flavor descriptors across all coffees

For each unique flavor at each taxonomy level:
- `has_{level}_{flavor}`: Boolean presence indicator
- `count_{level}_{flavor}`: Occurrence count
- `freq_{level}_{flavor}`: Relative frequency (count/total_instances)

Example columns:
- `has_family_Fruity`, `count_family_Fruity`, `freq_family_Fruity`
- `has_genus_Berry`, `count_genus_Berry`, `freq_genus_Berry`
- `has_species_Blueberry`, `count_species_Blueberry`, `freq_species_Blueberry`

### 6. TF-IDF Format (`tfidf_format`)

Document-term structure for TF-IDF analysis:
```python
{
    "family_level": {
        "countries": {
            "Ethiopia": ["Fruity", "Floral", "Fruity", "Nutty/Cocoa", ...],
            "Colombia": ["Nutty/Cocoa", "Sweet", "Fruity", ...]
        },
        "regions": {
            "Ethiopia_Sidamo": ["Fruity", "Floral", ...],
            "Colombia_Huila": ["Nutty/Cocoa", "Sweet", ...]
        },
        "sellers": {
            "Blue Bottle Coffee": ["Fruity", "Nutty/Cocoa", ...],
            "Intelligentsia": ["Floral", "Fruity", ...]
        }
    },
    "genus_level": {...},  # Same structure
    "species_level": {...}  # Same structure
}
```

### 7. Hierarchical Format (`hierarchical_format`)

Detailed hierarchical analysis structure:
```python
{
    "Ethiopia": {
        "unit_type": "country",
        "family_level": {
            "Fruity": {"count": 45, "frequency": 0.35},
            "Nutty/Cocoa": {"count": 23, "frequency": 0.18}
        },
        "genus_level": {
            "Berry": {
                "count": 25, 
                "frequency": 0.20, 
                "parent_family": "Fruity"
            },
            # ... other genera
        },
        "species_level": {
            "Blueberry": {
                "count": 15,
                "frequency": 0.12,
                "parent_genus": "Berry",
                "parent_family": "Fruity"
            },
            # ... other species
        },
        "hierarchy_tree": {
            "Fruity": {
                "Berry": ["Blueberry", "Strawberry", "Raspberry"],
                "Citrus Fruit": ["Orange", "Lemon", "Bergamot"],
                "Other Fruit": ["Peach", "Apple"]
            },
            "Nutty/Cocoa": {
                "Chocolate": ["Dark Chocolate", "Milk Chocolate"],
                "Nutty": ["Generic Nutty", "Almond"]
            }
        },
        "total_coffees": 45,
        "total_flavor_instances": 127
    }
    # ... other units
}
```

## Usage Examples

### 1. Basic Data Access
```python
from analytics.db_access.coffee_data_extractor import get_analytics_data

# Load all data (cached for 1 hour)
data = get_analytics_data()

# Access specific formats
country_data = data['country_aggregated']
contingency_df = data['contingency_format']
```

### 2. Finding Top Flavors for a Country
```python
ethiopia_data = data['country_aggregated']['Ethiopia']
flavor_counts = {}
for flavor in ethiopia_data['all_flavors']:
    family = flavor.get('family', '')
    flavor_counts[family] = flavor_counts.get(family, 0) + 1

# Sort by count
top_families = sorted(flavor_counts.items(), key=lambda x: x[1], reverse=True)
```

### 3. Statistical Analysis Setup
```python
# Get contingency data for countries only
country_contingency = data['contingency_format'][
    data['contingency_format']['unit_type'] == 'country'
]

# Get flavor columns for analysis
flavor_cols = [col for col in country_contingency.columns 
               if col.startswith('has_') or col.startswith('freq_')]
```

### 4. TF-IDF Document Preparation
```python
# Get country-level family terms
country_docs = data['tfidf_format']['family_level']['countries']

# Convert to document-term matrix format
for country, terms in country_docs.items():
    # terms is a list of all family occurrences for that country
    term_counts = Counter(terms)
```

## Data Quality Notes

1. **Flavor Taxonomy Consistency**: The system preserves the exact family→genus→species relationships as stored in the database
2. **Missing Data Handling**: 
   - Countries/regions with null values are excluded from aggregations
   - Coffees without flavors are included but contribute 0 to flavor counts
3. **Duplicate Handling**: The `all_flavors` lists intentionally include duplicates to preserve frequency information
4. **Memory Efficiency**: Data is cached for 1 hour using Streamlit's caching mechanism

## Next Steps

The analytics processing system should:
1. Use the contingency format for statistical significance testing
2. Use the TF-IDF format for distinctiveness analysis
3. Use the hierarchical format for multi-level analysis
4. Combine results from all three approaches for comprehensive insights