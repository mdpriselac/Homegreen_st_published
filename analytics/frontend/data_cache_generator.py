#!/usr/bin/env python3
"""
Frontend Data Cache Generator

Pre-computes and caches all analysis results for fast frontend loading.
Generates comprehensive unit profiles, flavor hierarchies, rankings, and summary data.
"""

import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict
import math

from analytics.db_access.coffee_data_extractor import get_analytics_data
from analytics.processing.distinctiveness_cache import (
    build_distinctiveness_components, prepare_distinctiveness_input)
from analytics.processing.data_hygiene import headline_counts, is_placeholder
from analytics.processing.turnover_analysis import SellerTurnoverAnalyzer
from analytics.processing.price_analysis import PriceAnalyzer
from analytics.processing.cooccurrence_analysis import FlavorCooccurrenceAnalyzer
from analytics.processing.cross_feature_analysis import CrossFeatureAnalyzer
from analytics.processing.interaction_analysis import InteractionAnalyzer


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrontendDataCacheGenerator:
    """Generate and cache all frontend-ready data"""
    
    def __init__(self, cache_dir: str = "analytics/data/frontend_cache"):
        # Resolve path relative to project root
        if not os.path.isabs(cache_dir):
            # Get the project root (where this script is running from)
            project_root = Path.cwd()
            self.cache_dir = project_root / cache_dir
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.all_results = None
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert objects to native JSON-safe types.

        numpy bool/int/float become Python bool/int/float (so json writes real
        booleans/numbers rather than str() fallbacks); NaN/Inf/NA become None.
        """
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, pd.DataFrame):
            return self._sanitize_for_json(obj.to_dict('records'))
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return [self._sanitize_for_json(item) for item in obj.tolist()]
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            obj = float(obj)
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif obj is None:
            return None
        elif pd.isna(obj):
            return None
        else:
            return obj
        
    def generate_full_cache(self):
        """Generate complete frontend data cache"""
        logger.info("Starting frontend data cache generation...")
        
        # Load the per-coffee data (single source for every analysis)
        logger.info("Loading analytics data...")
        self.all_results = {'data': get_analytics_data()}

        if not self.all_results['data']:
            raise ValueError("Analytics data not available")

        # Per-coffee distinctiveness analysis
        self.distinctiveness = self._compute_distinctiveness()
        # Survival (lifespan) analysis: also feeds the lifespan rankings
        self._turnover = self._generate_turnover_cache()

        # Generate all cache components
        cache_data = {
            'metadata': self._generate_metadata(),
            'overview_data': self._generate_overview_cache(),
            'flavor_hierarchies': self._generate_flavor_hierarchies_cache(),
            'rankings_data': self._generate_rankings_cache(),
            'distinctiveness_meta': self.distinctiveness['meta'],
            'distinctiveness_profiles': self.distinctiveness['profiles'],
            'distinctiveness_by_flavor': self.distinctiveness['by_flavor'],
            'turnover_data': self._turnover,
            'price_analysis_data': self._generate_price_analysis_cache(),
            'cooccurrence_data': self._generate_cooccurrence_cache(),
            'cross_feature_data': self._generate_cross_feature_cache(),
            'interaction_data': self._generate_interaction_cache(),
            'data_completeness': self._generate_completeness_cache(),
        }
        
        # Save to cache files
        self._save_cache_data(cache_data)
        
        logger.info("Frontend data cache generation completed successfully!")
        return cache_data
    
    def _compute_distinctiveness(self) -> Dict[str, Any]:
        """Per-coffee distinctiveness components (see distinctiveness_cache)"""
        logger.info("Computing per-coffee distinctiveness...")
        cross_feature_df = self.all_results['data'].get('cross_feature_df')
        if cross_feature_df is None or cross_feature_df.empty:
            raise ValueError("No per-coffee data available for distinctiveness analysis")
        return build_distinctiveness_components(prepare_distinctiveness_input(cross_feature_df))

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate cache metadata"""
        return {
            'generated_at': datetime.now().isoformat(),
            'version': '2.0',
            'analysis_components': ['distinctiveness', 'price', 'cooccurrence',
                                    'cross_feature', 'interaction', 'turnover'],
            'total_units': {
                t: self.distinctiveness['meta']['by_unit_type'][t]['n_units']['genus']
                for t in ('country', 'region', 'seller')
            },
        }
    
    def _generate_overview_cache(self) -> Dict[str, Any]:
        """Generate overview tab cache data"""
        logger.info("Generating overview cache...")
        
        overview_cache = {}
        
        # Basic dataset statistics
        overview_cache['dataset_stats'] = self._extract_dataset_stats()
        
        # Geographic distribution data
        overview_cache['geographic_data'] = self._extract_geographic_data()
        
        # Key findings (per-coffee distinctiveness; also in distinctiveness_meta)
        overview_cache['key_findings'] = self.distinctiveness['meta']['key_findings']
        
        return overview_cache
    
    def _generate_flavor_hierarchies_cache(self) -> Dict[str, Any]:
        """Generate flavor hierarchy data for By Flavor tab"""
        logger.info("Generating flavor hierarchies cache...")
        
        hierarchies = {
            'families': set(),
            'genera_by_family': defaultdict(set),
            'species_by_genus': defaultdict(set),
            'all_genera': set(),
            'all_species': set()
        }
        
        # Extract from raw data
        if 'raw_merged_df' in self.all_results.get('data', {}):
            df = self.all_results['data']['raw_merged_df']
            
            for _, row in df.iterrows():
                if row.get('has_flavors', False) and row.get('flavors_parsed'):
                    flavors = row['flavors_parsed']
                    
                    for flavor in flavors:
                        family = flavor.get('family', '')
                        genus = flavor.get('genus', '')
                        species = flavor.get('species', '')
                        
                        if family:
                            hierarchies['families'].add(family)
                            if genus:
                                hierarchies['genera_by_family'][family].add(genus)
                                hierarchies['all_genera'].add(genus)
                                if species:
                                    hierarchies['species_by_genus'][genus].add(species)
                                    hierarchies['all_species'].add(species)
        
        # Convert sets to sorted lists for JSON serialization
        return {
            'families': sorted(list(hierarchies['families'])),
            'genera_by_family': {f: sorted(list(g)) for f, g in hierarchies['genera_by_family'].items()},
            'species_by_genus': {g: sorted(list(s)) for g, s in hierarchies['species_by_genus'].items()},
            'all_genera': sorted(list(hierarchies['all_genera'])),
            'all_species': sorted(list(hierarchies['all_species']))
        }
    
    def _generate_rankings_cache(self) -> Dict[str, Any]:
        """Generate rankings data for Rankings tab"""
        logger.info("Generating rankings cache...")

        rankings = {}

        # Best Value origins (lowest median price with enough data)
        rankings['best_value'] = []
        # Highest priced origins
        rankings['highest_priced'] = []
        cross_feature_df = self.all_results['data'].get('cross_feature_df')
        if cross_feature_df is not None and not cross_feature_df.empty:
            priced = cross_feature_df[
                cross_feature_df['price_per_lb'].notna() &
                (cross_feature_df['price_per_lb'] > 0) &
                cross_feature_df['country'].notna()
            ]
            if not priced.empty:
                country_price = priced.groupby('country')['price_per_lb'].agg(
                    ['median', 'mean', 'count']
                ).reset_index()
                country_price = country_price[country_price['count'] >= 5]
                # Best value = lowest median price
                best_val = country_price.sort_values('median', ascending=True)
                for _, row in best_val.iterrows():
                    rankings['best_value'].append({
                        'entity_name': row['country'],
                        'unit_type': 'country',
                        'score': float(row['median']),
                        'mean_price': float(row['mean']),
                        'total_coffees': int(row['count']),
                    })
                # Highest priced
                highest = country_price.sort_values('median', ascending=False)
                for _, row in highest.iterrows():
                    rankings['highest_priced'].append({
                        'entity_name': row['country'],
                        'unit_type': 'country',
                        'score': float(row['median']),
                        'mean_price': float(row['mean']),
                        'total_coffees': int(row['count']),
                    })

        # Fastest Moving / Longest Lasting: Kaplan-Meier median lifespans (coffees still
        # listed count as "at least this long"), per country and per seller. Groups whose
        # median is not reached (fewer than half have left) cannot be ranked.
        rankings['fastest_moving'] = []
        rankings['longest_lasting'] = []
        turnover = getattr(self, '_turnover', None) or {}
        entries = []
        for unit_type, section, rows_key, name_key in (
                ('country', 'by_origin', 'countries', 'country'),
                ('seller', 'by_seller', 'sellers', 'seller')):
            for row in (turnover.get(section) or {}).get(rows_key, []):
                if row.get('median_lifespan') is None:
                    continue
                entries.append({
                    'entity_name': row[name_key],
                    'unit_type': unit_type,
                    'score': float(row['median_lifespan']),
                    'q25_lifespan': row.get('q25_lifespan'),
                    'q75_lifespan': row.get('q75_lifespan'),
                    'total_coffees': int(row.get('count', 0)),
                    'events': row.get('events'),
                    'censored': row.get('censored'),
                })
        rankings['fastest_moving'] = sorted(entries, key=lambda e: e['score'])
        rankings['longest_lasting'] = sorted(entries, key=lambda e: e['score'], reverse=True)

        # New per-unit-type rankings from the per-coffee distinctiveness analysis
        rankings['distinctive_profile'] = self.distinctiveness['rankings']['distinctive_profile']
        rankings['varied_profile'] = self.distinctiveness['rankings']['varied_profile']

        return self._sanitize_for_json(rankings)
    
    def _generate_turnover_cache(self) -> Dict[str, Any]:
        """Generate turnover/lifespan analysis cache"""
        logger.info("Generating turnover analysis cache...")
        try:
            cross_feature_df = self.all_results['data'].get('cross_feature_df')
            if cross_feature_df is None or cross_feature_df.empty:
                logger.warning("No cross-feature data available for turnover analysis")
                return {'has_data': False}

            analyzer = SellerTurnoverAnalyzer(cross_feature_df)
            result = analyzer.run_full_analysis()
            return self._sanitize_for_json(result)
        except Exception as e:
            logger.error(f"Failed to generate turnover cache: {e}")
            return {'has_data': False, 'error': str(e)}

    def _generate_price_analysis_cache(self) -> Dict[str, Any]:
        """Generate price analysis cache"""
        logger.info("Generating price analysis cache...")
        try:
            cross_feature_df = self.all_results['data'].get('cross_feature_df')
            if cross_feature_df is None or cross_feature_df.empty:
                logger.warning("No cross-feature data available for price analysis")
                return {'has_data': False}

            analyzer = PriceAnalyzer(cross_feature_df)
            result = analyzer.run_full_analysis()
            return self._sanitize_for_json(result)
        except Exception as e:
            logger.error(f"Failed to generate price analysis cache: {e}")
            return {'has_data': False, 'error': str(e)}

    def _generate_cooccurrence_cache(self) -> Dict[str, Any]:
        """Generate flavor co-occurrence analysis cache"""
        logger.info("Generating co-occurrence analysis cache...")
        try:
            cross_feature_df = self.all_results['data'].get('cross_feature_df')
            if cross_feature_df is None or cross_feature_df.empty:
                logger.warning("No cross-feature data available for co-occurrence analysis")
                return {'has_data': False}

            analyzer = FlavorCooccurrenceAnalyzer(cross_feature_df)
            result = analyzer.run_full_analysis()
            return self._sanitize_for_json(result)
        except Exception as e:
            logger.error(f"Failed to generate co-occurrence cache: {e}")
            return {'has_data': False, 'error': str(e)}

    def _generate_cross_feature_cache(self) -> Dict[str, Any]:
        """Generate cross-feature analysis cache"""
        logger.info("Generating cross-feature analysis cache...")
        try:
            cross_feature_df = self.all_results['data'].get('cross_feature_df')
            if cross_feature_df is None or cross_feature_df.empty:
                logger.warning("No cross-feature data available for cross-feature analysis")
                return {'has_data': False}

            analyzer = CrossFeatureAnalyzer(cross_feature_df)
            result = analyzer.run_full_analysis()
            return self._sanitize_for_json(result)
        except Exception as e:
            logger.error(f"Failed to generate cross-feature cache: {e}")
            return {'has_data': False, 'error': str(e)}

    def _generate_interaction_cache(self) -> Dict[str, Any]:
        """Generate multi-way interaction analysis cache"""
        logger.info("Generating interaction analysis cache...")
        try:
            cross_feature_df = self.all_results['data'].get('cross_feature_df')
            if cross_feature_df is None or cross_feature_df.empty:
                logger.warning("No cross-feature data available for interaction analysis")
                return {'has_data': False}

            analyzer = InteractionAnalyzer(cross_feature_df)
            result = analyzer.run_full_analysis()
            return self._sanitize_for_json(result)
        except Exception as e:
            logger.error(f"Failed to generate interaction cache: {e}")
            return {'has_data': False, 'error': str(e)}

    def _generate_completeness_cache(self) -> Dict[str, Any]:
        """Generate data completeness/coverage rates cache"""
        logger.info("Generating data completeness cache...")
        try:
            cross_feature_df = self.all_results['data'].get('cross_feature_df')
            if cross_feature_df is None or cross_feature_df.empty:
                return {'has_data': False}

            total = len(cross_feature_df)
            if total == 0:
                return {'has_data': False}

            def present(series):
                return ~series.map(is_placeholder)

            completeness = {
                'has_data': True,
                'total_coffees': total,
                'fields': {
                    'country': {
                        'count': int(present(cross_feature_df['country']).sum()),
                        'rate': float(present(cross_feature_df['country']).mean()),
                    },
                    # region = country+subregion known ("UNKNOWN" counts as missing)
                    'region': {
                        'count': int(present(cross_feature_df['region']).sum()),
                        'rate': float(present(cross_feature_df['region']).mean()),
                    },
                    'process_type': {
                        'count': int(present(cross_feature_df['process_type']).sum()),
                        'rate': float(present(cross_feature_df['process_type']).mean()),
                    },
                    'varietal': {
                        'count': int(cross_feature_df['has_varietal'].sum()),
                        'rate': float(cross_feature_df['has_varietal'].mean()),
                    },
                    'price': {
                        'count': int(cross_feature_df['has_price'].sum()),
                        'rate': float(cross_feature_df['has_price'].mean()),
                    },
                    'flavor': {
                        'count': int(cross_feature_df['has_flavors'].sum()),
                        'rate': float(cross_feature_df['has_flavors'].mean()),
                    },
                    'dates': {
                        'count': int(cross_feature_df['first_observed'].notna().sum()),
                        'rate': float(cross_feature_df['first_observed'].notna().mean()),
                    },
                },
            }
            return completeness
        except Exception as e:
            logger.error(f"Failed to generate completeness cache: {e}")
            return {'has_data': False, 'error': str(e)}

    def _extract_dataset_stats(self) -> Dict[str, Any]:
        """Extract basic dataset statistics"""
        stats = {}
        
        if 'data' in self.all_results and 'raw_merged_df' in self.all_results['data']:
            df = self.all_results['data']['raw_merged_df']
            
            # One definition of the headline numbers: regions are country+subregion
            # keys (what the sidebar lists); missing values never count.
            counts = headline_counts(df)
            stats = {k: counts[k] for k in (
                'total_coffees', 'countries_analyzed', 'regions_analyzed', 'sellers_analyzed')}
            stats['unique_flavor_families'] = counts['unique_flavor_families']
        
        return stats
    
    def _extract_geographic_data(self) -> List[Dict[str, Any]]:
        """Per-country coffee counts, from the per-coffee frame"""
        cf = self.all_results.get('data', {}).get('cross_feature_df')
        if cf is None or cf.empty:
            return []

        def present(series):
            return ~series.map(is_placeholder)

        geo_data = []
        for country, g in cf[present(cf['country'])].groupby('country'):
            families = {f for fl in g['flavor_families'] if isinstance(fl, list) for f in fl}
            geo_data.append({
                'country': country,
                'total_coffees': int(len(g)),
                'flavor_families': len(families),
                'sellers': int(g.loc[present(g['seller']), 'seller'].nunique()),
                'regions': int(g.loc[present(g['region']), 'region'].nunique()),
            })
        return geo_data
    
    def _save_cache_data(self, cache_data: Dict[str, Any]):
        """Save cache data to files"""
        logger.info("Saving cache data to files...")
        # Ensure numpy bool/int/float become real JSON types (not str() fallbacks)
        cache_data = self._sanitize_for_json(cache_data)
        
        # The loader reads only frontend_cache.json (compact JSON: it is parsed, not read)
        main_cache_file = self.cache_dir / "frontend_cache.json"
        with open(main_cache_file, 'w') as f:
            json.dump(cache_data, f, separators=(',', ':'), default=str)

        logger.info(f"Cache data saved to {self.cache_dir}")


def main():
    """Main function to generate frontend cache"""
    generator = FrontendDataCacheGenerator()
    cache_data = generator.generate_full_cache()
    
    print("\n✅ Frontend data cache generation completed!")
    print(f"📁 Cache saved to: {generator.cache_dir}")
    print(f"📊 Generated profiles for {len(cache_data['distinctiveness_profiles'])} units")
    print(f"🫘 Cached {len(cache_data['flavor_hierarchies']['families'])} flavor families")
    print(f"🏆 Generated {len(cache_data['rankings_data'])} ranking categories")


if __name__ == "__main__":
    main()