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

from analytics.processing.integrated_analysis import get_integrated_analysis, IntegratedFlavorAnalyzer
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
        self.analyzer = None
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively sanitize objects for JSON serialization, handling Infinity and NaN"""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None  # or "Infinity" / "-Infinity" if you want to preserve the sign
            else:
                return obj
        elif pd.isna(obj):
            return None
        else:
            return obj
        
    def generate_full_cache(self):
        """Generate complete frontend data cache"""
        logger.info("Starting frontend data cache generation...")
        
        # Load analysis results
        logger.info("Loading integrated analysis results...")
        self.all_results = get_integrated_analysis()
        self.analyzer = IntegratedFlavorAnalyzer()
        
        if not self.all_results or 'data' not in self.all_results:
            raise ValueError("Analysis results not available")
        
        # Generate all cache components
        cache_data = {
            'metadata': self._generate_metadata(),
            'overview_data': self._generate_overview_cache(),
            'unit_profiles': self._generate_unit_profiles_cache(),
            'flavor_hierarchies': self._generate_flavor_hierarchies_cache(),
            'rankings_data': self._generate_rankings_cache(),
            'comparison_matrices': self._generate_comparison_cache(),
            'export_ready_data': self._generate_export_cache(),
            'turnover_data': self._generate_turnover_cache(),
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
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate cache metadata"""
        return {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
            'analysis_components': list(self.all_results.keys()),
            'total_units': {
                'countries': len(self.all_results.get('data', {}).get('country_aggregated', {})),
                'regions': len(self.all_results.get('data', {}).get('region_aggregated', {})),
                'sellers': len(self.all_results.get('data', {}).get('seller_aggregated', {}))
            }
        }
    
    def _generate_overview_cache(self) -> Dict[str, Any]:
        """Generate overview tab cache data"""
        logger.info("Generating overview cache...")
        
        overview_cache = {}
        
        # Basic dataset statistics
        overview_cache['dataset_stats'] = self._extract_dataset_stats()
        
        # Geographic distribution data
        overview_cache['geographic_data'] = self._extract_geographic_data()
        
        # Key findings (from summary if available)
        if 'summary' in self.all_results:
            overview_cache['key_findings'] = self.all_results['summary'].get('key_findings', [])
            overview_cache['top_units'] = self.all_results['summary'].get('top_distinctive_units', [])
        else:
            overview_cache['key_findings'] = []
            overview_cache['top_units'] = []
        
        # Analysis progress indicators
        overview_cache['analysis_progress'] = {
            'statistical_complete': 'statistical' in self.all_results,
            'tfidf_complete': 'tfidf' in self.all_results,
            'hierarchical_complete': 'hierarchical' in self.all_results,
            'consensus_complete': 'consensus' in self.all_results,
            'summary_complete': 'summary' in self.all_results
        }
        
        return overview_cache
    
    def _generate_unit_profiles_cache(self) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive unit profiles for all units"""
        logger.info("Generating unit profiles cache...")
        
        unit_profiles = {}
        
        # Get all available units
        all_units = []
        
        if 'country_aggregated' in self.all_results.get('data', {}):
            for country in self.all_results['data']['country_aggregated'].keys():
                all_units.append((country, 'country'))
        
        if 'region_aggregated' in self.all_results.get('data', {}):
            for region in self.all_results['data']['region_aggregated'].keys():
                all_units.append((region, 'region'))
        
        if 'seller_aggregated' in self.all_results.get('data', {}):
            for seller in self.all_results['data']['seller_aggregated'].keys():
                all_units.append((seller, 'seller'))
        
        logger.info(f"Processing {len(all_units)} units...")
        
        # Generate profile for each unit
        for i, (unit_name, unit_type) in enumerate(all_units):
            if i % 10 == 0:
                logger.info(f"Processing unit {i+1}/{len(all_units)}: {unit_name}")
            
            try:
                profile = self._generate_unit_profile(unit_name, unit_type)
                unit_profiles[f"{unit_type}_{unit_name}"] = profile
            except Exception as e:
                logger.warning(f"Failed to generate profile for {unit_name} ({unit_type}): {e}")
                # Create minimal profile
                unit_profiles[f"{unit_type}_{unit_name}"] = {
                    'unit_name': unit_name,
                    'unit_type': unit_type,
                    'overview': {'total_coffees': 0, 'error': str(e)},
                    'statistical_findings': {},
                    'tfidf_findings': {},
                    'hierarchical_findings': {},
                    'consensus_findings': {'strong': [], 'moderate': []},
                    'recommendations': []
                }
        
        return unit_profiles
    
    def _generate_unit_profile(self, unit_name: str, unit_type: str) -> Dict[str, Any]:
        """Generate comprehensive profile for a single unit"""
        profile = {
            'unit_name': unit_name,
            'unit_type': unit_type,
            'overview': {},
            'statistical_findings': {},
            'tfidf_findings': {},
            'hierarchical_findings': {},
            'consensus_findings': {'strong': [], 'moderate': []},
            'recommendations': []
        }
        
        # Extract overview data from aggregated data
        aggregated_key = f"{unit_type}_aggregated"
        if aggregated_key in self.all_results.get('data', {}):
            unit_data = self.all_results['data'][aggregated_key].get(unit_name, {})
            metadata = unit_data.get('metadata', {})
            
            profile['overview'] = {
                'total_coffees': metadata.get('total_coffees', 0),
                'total_flavor_instances': metadata.get('total_flavor_instances', 0),
                'unique_flavor_families': list(metadata.get('unique_flavor_families', [])),
                'unique_sellers': list(metadata.get('unique_sellers', [])),
                'unique_subregions': list(metadata.get('unique_subregions', [])),
                'flavor_parse_rate': metadata.get('flavor_parse_rate', 0.0)
            }
        
        # Extract statistical findings
        if 'statistical' in self.all_results:
            profile['statistical_findings'] = self._extract_statistical_findings(unit_name, unit_type)
        
        # Extract TF-IDF findings  
        if 'tfidf' in self.all_results:
            profile['tfidf_findings'] = self._extract_tfidf_findings(unit_name, unit_type)
        
        # Extract hierarchical findings
        if 'hierarchical' in self.all_results:
            profile['hierarchical_findings'] = self._extract_hierarchical_findings(unit_name, unit_type)
        
        # Extract consensus findings
        if 'consensus' in self.all_results:
            profile['consensus_findings'] = self._extract_consensus_findings(unit_name, unit_type)
        
        # Generate recommendations
        profile['recommendations'] = self._generate_recommendations(profile)
        
        return profile
    
    def _extract_statistical_findings(self, unit_name: str, unit_type: str) -> Dict[str, List]:
        """Extract statistical significance findings for a unit"""
        findings = {'family': [], 'genus': [], 'species': []}
        
        stat_results = self.all_results.get('statistical', {})
        
        for taxonomy_level in ['family', 'genus', 'species']:
            key = f"{unit_type}_{taxonomy_level}_significant"
            if key in stat_results:
                df = stat_results[key]
                if not df.empty:
                    unit_df = df[df['unit_name'] == unit_name]
                    if not unit_df.empty:
                        findings[taxonomy_level] = self._sanitize_for_json(unit_df.to_dict('records'))
        
        return findings
    
    def _extract_tfidf_findings(self, unit_name: str, unit_type: str) -> Dict[str, Dict]:
        """Extract TF-IDF distinctiveness findings for a unit"""
        findings = {'family': {}, 'genus': {}, 'species': {}}
        
        tfidf_results = self.all_results.get('tfidf', {})
        
        for taxonomy_level in ['family', 'genus', 'species']:
            # Map unit types to the correct plural forms used in TF-IDF results
            unit_type_plurals = {'country': 'countries', 'region': 'regions', 'seller': 'sellers'}
            plural_type = unit_type_plurals.get(unit_type, f"{unit_type}s")
            key = f"{taxonomy_level}_{plural_type}_scores"
            if key in tfidf_results:
                df = tfidf_results[key]
                if not df.empty:
                    unit_df = df[df['unit_name'] == unit_name]
                    if not unit_df.empty:
                        top_flavors = self._sanitize_for_json(unit_df.nlargest(10, 'tfidf_score').to_dict('records'))
                        findings[taxonomy_level] = {
                            'top_flavors': top_flavors,
                            'total_unique_flavors': len(unit_df)
                        }
        
        return findings
    
    def _extract_hierarchical_findings(self, unit_name: str, unit_type: str) -> Dict[str, Any]:
        """Extract hierarchical analysis findings for a unit"""
        findings = {
            'cascade_patterns': {},
            'distinctiveness_types': {},
            'summary_metrics': {}
        }
        
        hier_results = self.all_results.get('hierarchical', {})
        if 'profiles' in hier_results and unit_name in hier_results['profiles']:
            unit_profile = hier_results['profiles'][unit_name]
            findings.update(unit_profile)
        
        return findings
    
    def _extract_consensus_findings(self, unit_name: str, unit_type: str) -> Dict[str, List]:
        """Extract cross-method consensus findings for a unit"""
        findings = {'strong': [], 'moderate': []}
        
        consensus_results = self.all_results.get('consensus', {})
        
        # Look for consensus findings for this unit
        for consensus_type in ['strong_consensus', 'moderate_consensus']:
            if consensus_type in consensus_results:
                for key, unit_findings in consensus_results[consensus_type].items():
                    if unit_type in key:
                        unit_consensus = [f for f in unit_findings if f['unit'] == unit_name]
                        if consensus_type == 'strong_consensus':
                            findings['strong'].extend(unit_consensus)
                        else:
                            findings['moderate'].extend(unit_consensus)
        
        return findings
    
    def _generate_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on profile"""
        recommendations = []
        
        overview = profile.get('overview', {})
        total_coffees = overview.get('total_coffees', 0)
        
        # Data quality recommendations
        if total_coffees < 5:
            recommendations.append("⚠️ Low sample size - results may not be statistically reliable")
        elif total_coffees < 20:
            recommendations.append("⚠️ Moderate sample size - interpret results with caution")
        
        # Statistical significance recommendations
        stat_findings = profile.get('statistical_findings', {})
        significant_count = sum(len(findings) for findings in stat_findings.values())
        
        if significant_count == 0:
            recommendations.append("ℹ️ No statistically significant flavor patterns found")
        elif significant_count > 10:
            recommendations.append("✨ Rich flavor profile with many distinctive characteristics")
        
        # TF-IDF distinctiveness recommendations
        tfidf_findings = profile.get('tfidf_findings', {})
        distinctive_count = sum(len(findings.get('top_flavors', [])) for findings in tfidf_findings.values())
        
        if distinctive_count > 15:
            recommendations.append("🎯 Highly distinctive flavor profile - great for specialty marketing")
        
        return recommendations
    
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
    
    def _get_top_flavors_for_entity(self, entity_name: str, unit_type: str, n: int = 3) -> List[str]:
        """Look up top N TF-IDF family-level flavors for an entity"""
        tfidf_results = self.all_results.get('tfidf', {})
        unit_type_plurals = {'country': 'countries', 'region': 'regions', 'seller': 'sellers'}
        plural = unit_type_plurals.get(unit_type, f"{unit_type}s")
        key = f"family_{plural}_scores"
        df = tfidf_results.get(key)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []
        unit_df = df[df['unit_name'] == entity_name]
        if unit_df.empty:
            return []
        top = unit_df.nlargest(n, 'tfidf_score')
        return top['flavor'].tolist()

    def _generate_rankings_cache(self) -> Dict[str, Any]:
        """Generate rankings data for Rankings tab"""
        logger.info("Generating rankings cache...")

        rankings = {}

        # Most Distinctive Overall
        if 'summary' in self.all_results and 'top_distinctive_units' in self.all_results['summary']:
            rankings['most_distinctive'] = self.all_results['summary']['top_distinctive_units']
        else:
            rankings['most_distinctive'] = []

        # Enrich most_distinctive with top flavors
        for entry in rankings['most_distinctive']:
            name = entry.get('unit') or entry.get('entity_name', '')
            utype = entry.get('type') or entry.get('unit_type', 'country')
            entry['top_flavors'] = self._get_top_flavors_for_entity(name, utype)

        # Most Specialized (from hierarchical analysis)
        rankings['most_specialized'] = []
        if 'hierarchical' in self.all_results and 'profiles' in self.all_results['hierarchical']:
            for unit_name, profile in self.all_results['hierarchical']['profiles'].items():
                top_flavors = self._get_top_flavors_for_entity(
                    unit_name, profile.get('unit_type', 'unknown')
                )
                rankings['most_specialized'].append({
                    'entity_name': unit_name,
                    'unit_type': profile.get('unit_type', 'unknown'),
                    'score': profile.get('summary_metrics', {}).get('concentration_index', 0),
                    'total_coffees': profile.get('total_coffees', 0),
                    'top_flavors': top_flavors,
                })

        # Most Diverse (from hierarchical analysis)
        rankings['most_diverse'] = []
        if 'hierarchical' in self.all_results and 'profiles' in self.all_results['hierarchical']:
            for unit_name, profile in self.all_results['hierarchical']['profiles'].items():
                top_flavors = self._get_top_flavors_for_entity(
                    unit_name, profile.get('unit_type', 'unknown')
                )
                rankings['most_diverse'].append({
                    'entity_name': unit_name,
                    'unit_type': profile.get('unit_type', 'unknown'),
                    'score': profile.get('summary_metrics', {}).get('flavor_diversity', 0),
                    'total_coffees': profile.get('total_coffees', 0),
                    'top_flavors': top_flavors,
                })

        # Best Value origins (lowest median price with enough data)
        rankings['best_value'] = []
        # Highest priced origins
        rankings['highest_priced'] = []
        cross_feature_df = self.all_results['data'].get('cross_feature_df')
        if cross_feature_df is not None and not cross_feature_df.empty:
            priced = cross_feature_df[
                cross_feature_df['avg_price'].notna() &
                (cross_feature_df['avg_price'] > 0) &
                cross_feature_df['country'].notna()
            ]
            if not priced.empty:
                country_price = priced.groupby('country')['avg_price'].agg(
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

        # Fastest Moving origins (shortest median lifespan for expired coffees)
        rankings['fastest_moving'] = []
        rankings['longest_lasting'] = []
        if cross_feature_df is not None and not cross_feature_df.empty:
            expired = cross_feature_df[
                (cross_feature_df['is_active'] == False) &
                cross_feature_df['lifespan_days'].notna() &
                cross_feature_df['country'].notna()
            ]
            if not expired.empty:
                country_life = expired.groupby('country')['lifespan_days'].agg(
                    ['median', 'mean', 'count']
                ).reset_index()
                country_life = country_life[country_life['count'] >= 5]
                # Fastest moving = shortest median lifespan
                fastest = country_life.sort_values('median', ascending=True)
                for _, row in fastest.iterrows():
                    rankings['fastest_moving'].append({
                        'entity_name': row['country'],
                        'unit_type': 'country',
                        'score': float(row['median']),
                        'mean_lifespan': float(row['mean']),
                        'total_coffees': int(row['count']),
                    })
                # Longest lasting
                longest = country_life.sort_values('median', ascending=False)
                for _, row in longest.iterrows():
                    rankings['longest_lasting'].append({
                        'entity_name': row['country'],
                        'unit_type': 'country',
                        'score': float(row['median']),
                        'mean_lifespan': float(row['mean']),
                        'total_coffees': int(row['count']),
                    })

        return self._sanitize_for_json(rankings)
    
    def _generate_comparison_cache(self) -> Dict[str, Any]:
        """Generate comparison matrices for Compare tab"""
        logger.info("Generating comparison cache...")
        
        comparison_data = {}
        
        # Generate similarity matrices if TF-IDF analyzer is available
        if 'tfidf' in self.all_results and 'analyzer' in self.all_results['tfidf']:
            try:
                analyzer = self.all_results['tfidf']['analyzer']
                
                # Use plural forms as expected by the TF-IDF analyzer
                entity_type_mapping = {'country': 'countries', 'region': 'regions', 'seller': 'sellers'}
                
                for entity_type in ['country', 'region', 'seller']:
                    for taxonomy_level in ['family', 'genus', 'species']:
                        key = f"{taxonomy_level}_{entity_type}_similarity"
                        # Use the plural form for the analyzer
                        analyzer_entity_type = entity_type_mapping[entity_type]
                        similarity_matrix = analyzer.calculate_similarity_matrix(taxonomy_level, analyzer_entity_type)
                        
                        if similarity_matrix is not None:
                            # Handle both DataFrame and numpy array cases
                            if hasattr(similarity_matrix, 'values'):
                                # It's a DataFrame
                                matrix_values = similarity_matrix.values.tolist()
                                labels = list(similarity_matrix.index)
                            else:
                                # It's a numpy array
                                matrix_values = similarity_matrix.tolist()
                                labels = []
                            
                            comparison_data[key] = {
                                'matrix': matrix_values,
                                'labels': labels,
                                'description': f"Flavor similarity between {entity_type}s at {taxonomy_level} level"
                            }
            except Exception as e:
                logger.warning(f"Failed to generate similarity matrices: {e}")
        
        return comparison_data
    
    def _generate_export_cache(self) -> Dict[str, Any]:
        """Generate export-ready data formats"""
        logger.info("Generating export cache...")
        
        export_data = {
            'statistical_summary': [],
            'tfidf_summary': [],
            'consensus_summary': []
        }
        
        # Statistical findings summary
        if 'statistical' in self.all_results:
            for key, df in self.all_results['statistical'].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    summary = self._sanitize_for_json(df.to_dict('records'))
                    export_data['statistical_summary'].extend(summary)
        
        # TF-IDF findings summary
        if 'tfidf' in self.all_results:
            for key, df in self.all_results['tfidf'].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    summary = self._sanitize_for_json(df.to_dict('records'))
                    export_data['tfidf_summary'].extend(summary)
        
        return export_data
    
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

            completeness = {
                'has_data': True,
                'total_coffees': total,
                'fields': {
                    'country': {
                        'count': int(cross_feature_df['country'].notna().sum()),
                        'rate': float(cross_feature_df['country'].notna().mean()),
                    },
                    'region': {
                        'count': int(cross_feature_df['region'].notna().sum()),
                        'rate': float(cross_feature_df['region'].notna().mean()),
                    },
                    'process_type': {
                        'count': int(cross_feature_df['has_process'].sum()),
                        'rate': float(cross_feature_df['has_process'].mean()),
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
            
            stats = {
                'total_coffees': len(df),
                'countries_analyzed': df['country_final'].nunique() if 'country_final' in df.columns else 0,
                'regions_analyzed': df['subregion_final'].nunique() if 'subregion_final' in df.columns else 0,
                'sellers_analyzed': df['seller_name'].nunique() if 'seller_name' in df.columns else 0,
                'flavor_parse_rate': df['has_flavors'].mean() if 'has_flavors' in df.columns else 0.0
            }
        
        return stats
    
    def _extract_geographic_data(self) -> List[Dict[str, Any]]:
        """Extract geographic distribution data"""
        geo_data = []
        
        if 'country_aggregated' in self.all_results.get('data', {}):
            country_data = self.all_results['data']['country_aggregated']
            
            for country, data in country_data.items():
                metadata = data.get('metadata', {})
                geo_data.append({
                    'country': country,
                    'total_coffees': metadata.get('total_coffees', 0),
                    'flavor_families': len(metadata.get('unique_flavor_families', [])),
                    'sellers': len(metadata.get('unique_sellers', [])),
                    'regions': len(metadata.get('unique_subregions', []))
                })
        
        return geo_data
    
    def _save_cache_data(self, cache_data: Dict[str, Any]):
        """Save cache data to files"""
        logger.info("Saving cache data to files...")
        
        # Save main cache file
        main_cache_file = self.cache_dir / "frontend_cache.json"
        with open(main_cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        # Save individual components for partial loading
        for component_name, component_data in cache_data.items():
            component_file = self.cache_dir / f"{component_name}.json"
            with open(component_file, 'w') as f:
                json.dump(component_data, f, indent=2, default=str)
        
        logger.info(f"Cache data saved to {self.cache_dir}")


def main():
    """Main function to generate frontend cache"""
    generator = FrontendDataCacheGenerator()
    cache_data = generator.generate_full_cache()
    
    print("\n✅ Frontend data cache generation completed!")
    print(f"📁 Cache saved to: {generator.cache_dir}")
    print(f"📊 Generated profiles for {len(cache_data['unit_profiles'])} units")
    print(f"🫘 Cached {len(cache_data['flavor_hierarchies']['families'])} flavor families")
    print(f"🏆 Generated {len(cache_data['rankings_data'])} ranking categories")


if __name__ == "__main__":
    main()