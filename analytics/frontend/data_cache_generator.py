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

from analytics.processing.integrated_analysis import get_integrated_analysis, IntegratedFlavorAnalyzer


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
            'export_ready_data': self._generate_export_cache()
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
                        findings[taxonomy_level] = unit_df.to_dict('records')
        
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
                        top_flavors = unit_df.nlargest(10, 'tfidf_score').to_dict('records')
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
    
    def _generate_rankings_cache(self) -> Dict[str, Any]:
        """Generate rankings data for Rankings tab"""
        logger.info("Generating rankings cache...")
        
        rankings = {}
        
        # Most Distinctive Overall
        if 'summary' in self.all_results and 'top_distinctive_units' in self.all_results['summary']:
            rankings['most_distinctive'] = self.all_results['summary']['top_distinctive_units']
        else:
            rankings['most_distinctive'] = []
        
        # Most Specialized (from hierarchical analysis)
        rankings['most_specialized'] = []
        if 'hierarchical' in self.all_results and 'profiles' in self.all_results['hierarchical']:
            for unit_name, profile in self.all_results['hierarchical']['profiles'].items():
                rankings['most_specialized'].append({
                    'entity_name': unit_name,
                    'unit_type': profile.get('unit_type', 'unknown'),
                    'score': profile.get('summary_metrics', {}).get('concentration_index', 0),
                    'total_coffees': profile.get('total_coffees', 0)
                })
        
        # Most Diverse (from hierarchical analysis)
        rankings['most_diverse'] = []
        if 'hierarchical' in self.all_results and 'profiles' in self.all_results['hierarchical']:
            for unit_name, profile in self.all_results['hierarchical']['profiles'].items():
                rankings['most_diverse'].append({
                    'entity_name': unit_name,
                    'unit_type': profile.get('unit_type', 'unknown'),
                    'score': profile.get('summary_metrics', {}).get('flavor_diversity', 0),
                    'total_coffees': profile.get('total_coffees', 0)
                })
        
        return rankings
    
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
                    summary = df.to_dict('records')
                    export_data['statistical_summary'].extend(summary)
        
        # TF-IDF findings summary
        if 'tfidf' in self.all_results:
            for key, df in self.all_results['tfidf'].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    summary = df.to_dict('records')
                    export_data['tfidf_summary'].extend(summary)
        
        return export_data
    
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