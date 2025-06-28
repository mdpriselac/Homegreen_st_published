# Coffee Flavor Analysis - Database Access Instructions (Updated)

## Database Structure Overview

### Key Tables and Relationships
```
sellers (id, name, homepage, directory_name)
    ↓ (one-to-many via seller_id)
coffees (id, seller_id, name, url, url_hash)
    ↓ (one-to-many via coffee_id)
coffee_attributes (coffee_id, country_final, subregion_final, categorized_flavors, is_cleaned)
```

### Primary Data Source
- **Main table**: `coffee_attributes`
- **Filter**: Only records where `is_cleaned = true`
- **Geographic data**: 
  - `country_final` (clean country names)
  - `subregion_final` (clean subregion names within countries)
- **Flavor data**: `categorized_flavors` (JSON field containing list of flavor dictionaries)

### Seller Information
- Pull seller names from: `coffees.seller_id → sellers.name`
- This requires joining: `coffee_attributes → coffees → sellers`

## Raw Data Extraction Strategy

### Step 1: Pull Core Dataset
Pull all records from `coffee_attributes` where `is_cleaned = true` with these fields:
- `coffee_id`
- `country_final` 
- `subregion_final`
- `categorized_flavors` (JSON field)

### Step 2: Pull Coffee-Seller Mapping
Pull from `coffees` table joined with `sellers`:
- `coffees.id` (to match with coffee_id)
- `coffees.name` (coffee name)
- `coffees.seller_id`
- `sellers.name` (seller name)

### Step 3: Merge and Aggregate
Combine the datasets and aggregate by analysis units (countries, regions, sellers).

## Expected Flavor Data Format

### Input: `categorized_flavors` Field
The `categorized_flavors` field is a JSON list of dictionaries, each containing exactly three keys:

**Example structure**:
```json
[
  {"family": "Floral", "genus": "Floral", "species": "Floral"},
  {"family": "Fruity", "genus": "Other Fruit", "species": "Peach"},
  {"family": "Fruity", "genus": "Citrus Fruit", "species": "Orange"},
  {"family": "Nutty/Cocoa", "genus": "Nutty", "species": "Generic Nutty"}
]
```

### Flavor Processing Requirements
- **No parsing needed** - data is already structured
- **Direct access** to family, genus, species taxonomy levels
- **Handle empty/null values** gracefully (some records may have empty `categorized_flavors`)
- **Create composite keys** for analysis (e.g., "Fruity|Citrus Fruit|Orange")

## Desired Output Data Structures

### 1. Country-Level Aggregation
**Format**: Dictionary with country names as keys
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
      // ... more coffees
    ],
    "all_flavors": [
      {"family": "Fruity", "genus": "Berry", "species": "Blueberry"},
      {"family": "Floral", "genus": "Floral", "species": "Jasmine"},
      {"family": "Fruity", "genus": "Berry", "species": "Blueberry"},  // duplicates from multiple coffees
      // ... all flavor instances from all coffees in this country
    ],
    "metadata": {
      "total_coffees": 45,
      "unique_subregions": ["Sidamo", "Harrar", "Kaffa"],
      "unique_sellers": ["Blue Bottle", "Intelligentsia", "Counter Culture"],
      "total_flavor_instances": 127,
      "unique_flavor_families": ["Fruity", "Floral", "Nutty/Cocoa"],
      "unique_flavor_genera": ["Berry", "Citrus Fruit", "Floral", "Nutty"],
      "unique_flavor_species": ["Blueberry", "Orange", "Jasmine", "Generic Nutty"]
    }
  }
  // ... other countries
}
```

### 2. Region-Level Aggregation  
**Format**: Dictionary with region keys (country_subregion)
```python
{
  "Ethiopia_Sidamo": {
    "country": "Ethiopia",
    "subregion": "Sidamo", 
    "coffees": [...],  // Same structure as country level
    "all_flavors": [...],
    "metadata": {
      "total_coffees": 15,
      "unique_sellers": ["Blue Bottle", "Intelligentsia"],
      "total_flavor_instances": 42,
      "unique_flavor_families": ["Fruity", "Floral"],
      "unique_flavor_genera": ["Berry", "Floral"],
      "unique_flavor_species": ["Blueberry", "Jasmine"]
    }
  }
  // ... other regions
}
```

### 3. Seller-Level Aggregation
**Format**: Dictionary with seller names as keys
```python
{
  "Blue Bottle Coffee": {
    "seller_id": 5,
    "coffees": [...],  // Same structure, but with country/subregion info
    "all_flavors": [...],
    "metadata": {
      "total_coffees": 23,
      "unique_countries": ["Ethiopia", "Colombia", "Guatemala"],
      "unique_subregions": ["Sidamo", "Huila", "Antigua"],
      "total_flavor_instances": 78,
      "unique_flavor_families": ["Fruity", "Nutty/Cocoa", "Sweet"],
      "unique_flavor_genera": ["Berry", "Chocolate", "Brown Sugar"],
      "unique_flavor_species": ["Blueberry", "Dark Chocolate", "Caramel"]
    }
  }
  // ... other sellers
}
```

## Analysis-Ready Data Formats

### For Frequency & Association Analysis
**Contingency Table Format**: DataFrame with columns:
- `unit` (country/region/seller name)
- `unit_type` ("country", "region", or "seller")
- `total_coffees` (number of coffees in this unit)
- `total_flavor_instances` (total flavor instances across all coffees)

**Flavor presence/frequency columns for each taxonomy level**:
- `has_family_{family_name}` (boolean: unit has this family)
- `count_family_{family_name}` (count: how many times this family appears)
- `freq_family_{family_name}` (frequency: count/total_flavor_instances)
- `has_genus_{genus_name}` (boolean: unit has this genus)
- `count_genus_{genus_name}` (count: how many times this genus appears)
- `freq_genus_{genus_name}` (frequency: count/total_flavor_instances)
- `has_species_{species_name}` (boolean: unit has this species)
- `count_species_{species_name}` (count: how many times this species appears)
- `freq_species_{species_name}` (frequency: count/total_flavor_instances)

### For TF-IDF Analysis
**Document-Term Format**: Dictionary structure with separate levels:
```python
{
  "family_level": {
    "countries": {
      "Ethiopia": ["Fruity", "Floral", "Fruity", "Nutty/Cocoa", ...],  # all family instances
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
  "genus_level": {
    "countries": {
      "Ethiopia": ["Berry", "Floral", "Citrus Fruit", ...],
      "Colombia": ["Chocolate", "Brown Sugar", ...]
    }
    // ... same structure for regions and sellers
  },
  "species_level": {
    "countries": {
      "Ethiopia": ["Blueberry", "Jasmine", "Orange", ...],
      "Colombia": ["Dark Chocolate", "Caramel", ...]
    }
    // ... same structure for regions and sellers
  }
}
```

### For Hierarchical Analysis
**Multi-Level Structure**: Dictionary with taxonomy levels and hierarchical relationships:
```python
{
  "Ethiopia": {
    "family_level": {
      "Fruity": {"count": 45, "frequency": 0.35},
      "Nutty/Cocoa": {"count": 23, "frequency": 0.18},
      "Floral": {"count": 15, "frequency": 0.12}
    },
    "genus_level": {
      "Berry": {"count": 25, "frequency": 0.20, "parent_family": "Fruity"},
      "Citrus Fruit": {"count": 20, "frequency": 0.16, "parent_family": "Fruity"},
      "Chocolate": {"count": 18, "frequency": 0.14, "parent_family": "Nutty/Cocoa"},
      "Floral": {"count": 15, "frequency": 0.12, "parent_family": "Floral"}
    },
    "species_level": {
      "Blueberry": {"count": 15, "frequency": 0.12, "parent_genus": "Berry", "parent_family": "Fruity"},
      "Orange": {"count": 12, "frequency": 0.09, "parent_genus": "Citrus Fruit", "parent_family": "Fruity"},
      "Dark Chocolate": {"count": 10, "frequency": 0.08, "parent_genus": "Chocolate", "parent_family": "Nutty/Cocoa"}
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
      },
      "Floral": {
        "Floral": ["Jasmine", "Rose", "Floral"]
      }
    },
    "total_coffees": 45,
    "total_flavor_instances": 127
  }
}
```

## Performance Considerations

### Efficient Data Pulling Strategy
1. **Single Query for Attributes**: Pull all `coffee_attributes` where `is_cleaned = true`
2. **Single Query for Coffee-Seller Mapping**: Pull all needed coffee and seller relationships
3. **JSON Processing**: Handle `categorized_flavors` as native JSON/list - no string parsing needed
4. **In-Memory Aggregation**: Group and aggregate in Python after data retrieval

### Data Quality Checks
- **Validate JSON structure**: Ensure `categorized_flavors` is properly formatted list of dictionaries
- **Check required keys**: Verify each flavor dict has 'family', 'genus', 'species' keys
- **Handle missing data**: Gracefully handle null/empty `categorized_flavors`
- **Validate taxonomy consistency**: Check that genus-species relationships are consistent across dataset

### Memory Management
- **Pre-filter data**: Only pull cleaned records to reduce memory usage
- **Lazy processing**: Process flavors on-demand rather than pre-expanding all combinations
- **Cache taxonomy mappings**: Store unique family→genus→species relationships for validation

This updated structure leverages the already well-structured `categorized_flavors` JSON data to provide clean, hierarchical flavor analysis without complex text parsing.