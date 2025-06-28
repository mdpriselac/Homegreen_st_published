# Coffee Flavor Distinctiveness Analysis - Development Plan

## Data Processing Requirements

### Input Data Structure Expected
The system should process data with the following structure:
- **Coffee records** with fields: `coffee_id`, `country`, `subregion`, `seller_id`
- **Flavor records** with hierarchical taxonomy: `flavor_family`, `flavor_genus`, `flavor_species`
- **Coffee-Flavor associations** linking each coffee to its flavor profile

### Data Preprocessing Steps
1. **Data Validation & Cleaning**
   - Standardize country/region names (handle variations, misspellings)
   - Normalize seller identifiers
   - Validate flavor taxonomy completeness (ensure family→genus→species hierarchy is intact)
   - Remove duplicate coffee-flavor associations

2. **Aggregation Level Creation**
   - Create analysis groupings: `country`, `subregion`, `seller`
   - Generate flavor vectors for each geographic/seller unit
   - Build taxonomy level datasets (family-only, genus-only, species-level)

## Development Goals

### Phase 1: Flavor Frequency & Association Analysis

**Goal 1.1: Build Contingency Tables**
```
For each combination of:
- Analysis level: [country, subregion, seller]  
- Taxonomy level: [family, genus, species]
- Geographic/seller unit: [each unique unit]
- Flavor: [each unique flavor at that taxonomy level]

Create 2x2 contingency table:
                Has Flavor    No Flavor
In Unit            a            b
Outside Unit       c            d
```

**Goal 1.2: Statistical Testing Engine**
- Implement Chi-square test with Yates' continuity correction
- Implement Fisher's exact test for small samples (expected frequency < 5)
- Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg procedure
- Calculate odds ratios with 95% confidence intervals
- Compute Cramér's V for effect size

**Goal 1.3: Significance Filtering**
- Flag results with p < 0.05 (after FDR correction)
- Filter for meaningful effect sizes (Cramér's V > 0.1)
- Rank by combination of statistical significance and effect size

**Deliverable 1**: CSV files with columns:
`analysis_level, unit_name, taxonomy_level, flavor, p_value, p_value_corrected, odds_ratio, odds_ratio_ci_lower, odds_ratio_ci_upper, cramers_v, interpretation`

### Phase 2: TF-IDF Analysis

**Goal 2.1: Document-Term Matrix Construction**
```
Documents = Geographic/Seller Units
Terms = Flavors (at each taxonomy level)
Values = Term Frequency (proportion of coffees in unit with that flavor)
```

**Goal 2.2: TF-IDF Calculation**
- **Term Frequency (TF)**: `count(flavor in unit) / count(total coffees in unit)`
- **Document Frequency (DF)**: `count(units containing flavor)`
- **Inverse Document Frequency (IDF)**: `log(total_units / DF)`
- **TF-IDF Score**: `TF × IDF`

**Goal 2.3: Distinctiveness Ranking**
- For each unit, rank all flavors by TF-IDF score
- Create "flavor fingerprints" showing top N distinctive flavors
- Generate similarity matrices between units based on TF-IDF vectors

**Deliverable 2**: 
- CSV: `analysis_level, unit_name, taxonomy_level, flavor, tf_score, idf_score, tfidf_score, rank_within_unit`
- JSON: Unit flavor fingerprints with top 10 distinctive flavors per unit

### Phase 3: Multi-level Distinctive Features

**Goal 3.1: Hierarchical Analysis Pipeline**
- Start with family-level analysis using methods from Phase 1 or Phase 2
- For each significantly distinctive family, drill down to genus level
- For each significantly distinctive genus, drill down to species level
- Create parent-child relationship tracking

**Goal 3.2: Cascade Validation**
- Verify that child-level significance supports parent-level findings
- Identify "broad distinctiveness" (significant at family, not species) vs "narrow distinctiveness" (significant only at species)
- Flag contradictory patterns for manual review

**Goal 3.3: Hierarchical Profile Generation**
- Create tree structures showing distinctiveness inheritance
- Calculate "specificity scores" measuring how focused distinctiveness is
- Generate hierarchical confidence scores

**Deliverable 3**: 
- JSON: Hierarchical trees per unit with statistical confidence at each level
- CSV: `analysis_level, unit_name, flavor_family, flavor_genus, flavor_species, hierarchy_level, significance_method, confidence_score, distinctiveness_type`

## Integration & Final Output

### Goal 4.1: Cross-Analysis Validation
- Compare results across all three methods
- Identify consensus findings (flavors distinctive across multiple approaches)
- Flag discrepancies for investigation

### Goal 4.2: Visualization Data Preparation
- Generate aggregated summary tables
- Create comparative ranking across regions/countries/sellers
- Prepare data for potential geographic mapping

### Goal 4.3: Executive Summary Generation
- Automatically generate top findings per analysis level
- Create "most distinctive" rankings
- Identify surprising or counterintuitive results

## Final Deliverables

### 1. Data Files
- **Raw Results**: Individual CSV files from each analysis phase
- **Integrated Dataset**: Combined results with cross-method validation flags
- **Summary Tables**: Top findings per geographic unit and analysis method

### 2. Metadata & Documentation
- **Data Dictionary**: Column definitions and calculation methods
- **Analysis Log**: Parameters used, sample sizes, statistical thresholds
- **Quality Report**: Data completeness, potential biases, confidence levels

### 3. Ready-to-Use Outputs
- **Flavor Fingerprints**: JSON files with distinctive flavor profiles
- **Ranking Tables**: Most distinctive regions/countries/sellers by flavor category
- **Hierarchical Profiles**: Multi-level distinctiveness trees per unit

## Technical Requirements

### Performance Considerations
- Implement batch processing for large datasets
- Use vectorized operations for statistical calculations
- Cache intermediate results for multi-level analysis

### Validation Steps
- Cross-validate statistical test selection (Chi-square vs Fisher's exact)
- Verify TF-IDF calculations against sample manual calculations
- Test hierarchical cascade logic with known data patterns

### Error Handling
- Handle units with insufficient sample sizes
- Manage missing taxonomy levels gracefully
- Provide meaningful warnings for edge cases

This plan creates a systematic approach to identifying what makes each region's, country's, and seller's coffee flavors truly distinctive in the global landscape.