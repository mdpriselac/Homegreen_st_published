"""
Integrated analysis combining statistical, TF-IDF, and hierarchical approaches

This module integrates results from all three analysis methods to provide
comprehensive insights and cross-validated findings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
from collections import defaultdict

from analytics.db_access.coffee_data_extractor import get_analytics_data
from analytics.processing.statistical_analysis import run_full_statistical_analysis
from analytics.processing.tfidf_analysis import run_full_tfidf_analysis
from analytics.processing.hierarchical_analysis import run_full_hierarchical_analysis


class IntegratedFlavorAnalyzer:
    """Integrate and cross-validate results from multiple analysis methods"""
    
    def __init__(self):
        """Initialize the integrated analyzer"""
        self.data = None
        self.statistical_results = None
        self.tfidf_results = None
        self.hierarchical_results = None
        self.consensus_findings = None
    
    @st.cache_data(ttl=3600)
    def run_all_analyses(_self) -> Dict[str, Any]:
        """
        Run all analysis methods and integrate results
        
        Returns:
            Dictionary with all analysis results
        """
        # Extract data
        with st.spinner("Extracting coffee data..."):
            _self.data = get_analytics_data()
        
        # Run statistical analysis
        with st.spinner("Running statistical significance analysis..."):
            _self.statistical_results = run_full_statistical_analysis(
                _self.data['contingency_format']
            )
        
        # Run TF-IDF analysis
        with st.spinner("Running TF-IDF distinctiveness analysis..."):
            _self.tfidf_results = run_full_tfidf_analysis(
                _self.data['tfidf_format']
            )
        
        # Run hierarchical analysis
        with st.spinner("Running hierarchical pattern analysis..."):
            _self.hierarchical_results = run_full_hierarchical_analysis(
                _self.data['hierarchical_format'],
                _self.statistical_results,
                _self.tfidf_results
            )
        
        # Find consensus findings
        with st.spinner("Integrating results across methods..."):
            _self.consensus_findings = _self._find_consensus_findings()
        
        return {
            'data': _self.data,
            'statistical': _self.statistical_results,
            'tfidf': _self.tfidf_results,
            'hierarchical': _self.hierarchical_results,
            'consensus': _self.consensus_findings,
            'summary': _self._generate_executive_summary()
        }
    
    def _find_consensus_findings(self) -> Dict[str, Any]:
        """Find flavors that are distinctive across multiple methods"""
        consensus = {
            'strong_consensus': defaultdict(list),  # Significant in all methods
            'moderate_consensus': defaultdict(list),  # Significant in 2/3 methods
            'method_specific': defaultdict(list),    # Significant in only 1 method
            'contradictions': []                     # Conflicting results
        }
        
        # Process each unit type and taxonomy level
        for unit_type in ['country', 'region', 'seller']:
            for taxonomy_level in ['family', 'genus', 'species']:
                # Get results from each method
                stat_key = f'{unit_type}_{taxonomy_level}_significant'
                tfidf_key = f'{taxonomy_level}_{unit_type}s_scores'
                
                if stat_key not in self.statistical_results:
                    continue
                
                stat_df = self.statistical_results[stat_key]
                tfidf_df = self.tfidf_results.get(tfidf_key, pd.DataFrame())
                
                # Get unique units
                units = set()
                if not stat_df.empty:
                    units.update(stat_df['unit_name'].unique())
                if not tfidf_df.empty:
                    units.update(tfidf_df['unit_name'].unique())
                
                for unit in units:
                    # Get top findings from each method
                    stat_flavors = self._get_statistical_top_flavors(
                        stat_df, unit, top_n=10
                    )
                    tfidf_flavors = self._get_tfidf_top_flavors(
                        tfidf_df, unit, top_n=10
                    )
                    hier_flavors = self._get_hierarchical_top_flavors(
                        unit, taxonomy_level, top_n=10
                    )
                    
                    # Find overlaps
                    all_methods = set(stat_flavors) & set(tfidf_flavors) & set(hier_flavors)
                    two_methods = ((set(stat_flavors) & set(tfidf_flavors)) |
                                 (set(stat_flavors) & set(hier_flavors)) |
                                 (set(tfidf_flavors) & set(hier_flavors))) - all_methods
                    
                    # Record consensus findings
                    key = f'{unit_type}_{taxonomy_level}'
                    
                    for flavor in all_methods:
                        consensus['strong_consensus'][key].append({
                            'unit': unit,
                            'flavor': flavor,
                            'methods': ['statistical', 'tfidf', 'hierarchical']
                        })
                    
                    for flavor in two_methods:
                        methods = []
                        if flavor in stat_flavors:
                            methods.append('statistical')
                        if flavor in tfidf_flavors:
                            methods.append('tfidf')
                        if flavor in hier_flavors:
                            methods.append('hierarchical')
                        
                        consensus['moderate_consensus'][key].append({
                            'unit': unit,
                            'flavor': flavor,
                            'methods': methods
                        })
        
        return dict(consensus)
    
    def _get_statistical_top_flavors(self, df: pd.DataFrame, unit: str, 
                                   top_n: int = 10) -> List[str]:
        """Get top statistically significant flavors for a unit"""
        if df.empty:
            return []
        
        unit_df = df[df['unit_name'] == unit]
        if unit_df.empty:
            return []
        
        # Sort by combined score (p-value and effect size)
        unit_df = unit_df.copy()
        unit_df['score'] = -np.log10(unit_df['p_value_corrected'] + 1e-10) * unit_df['cramers_v']
        top_df = unit_df.nlargest(top_n, 'score')
        
        return top_df['flavor'].tolist()
    
    def _get_tfidf_top_flavors(self, df: pd.DataFrame, unit: str, 
                             top_n: int = 10) -> List[str]:
        """Get top TF-IDF distinctive flavors for a unit"""
        if df.empty:
            return []
        
        unit_df = df[df['unit_name'] == unit]
        if unit_df.empty:
            return []
        
        top_df = unit_df.nlargest(top_n, 'tfidf_score')
        return top_df['flavor'].tolist()
    
    def _get_hierarchical_top_flavors(self, unit: str, taxonomy_level: str,
                                    top_n: int = 10) -> List[str]:
        """Get top flavors from hierarchical analysis"""
        if unit not in self.hierarchical_results['profiles']:
            return []
        
        profile = self.hierarchical_results['profiles'][unit]
        level_data = profile.get('cascade_patterns', {}).get('patterns', [])
        
        # Extract flavors at the requested level
        flavors = []
        if taxonomy_level == 'family':
            flavors = [(p['family'], p['family_frequency']) for p in level_data]
        elif taxonomy_level == 'genus':
            for pattern in level_data:
                for genus in pattern.get('genera', []):
                    flavors.append((genus['genus'], genus['genus_frequency']))
        elif taxonomy_level == 'species':
            for pattern in level_data:
                for genus in pattern.get('genera', []):
                    for species in genus.get('species', []):
                        flavors.append((species['species'], species['species_frequency']))
        
        # Sort by frequency and get top N
        flavors.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in flavors[:top_n]]
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate high-level summary of findings"""
        summary = {
            'dataset_overview': {
                'total_coffees': len(self.data['raw_merged_df']),
                'countries_analyzed': len(self.data['country_aggregated']),
                'regions_analyzed': len(self.data['region_aggregated']),
                'sellers_analyzed': len(self.data['seller_aggregated']),
                'flavor_parse_rate': (self.data['raw_merged_df']['has_flavors'].sum() / 
                                     len(self.data['raw_merged_df']) * 100)
            },
            'key_findings': [],
            'top_distinctive_units': [],
            'unique_discoveries': []
        }
        
        # Find most distinctive countries
        country_scores = []
        for country in self.data['country_aggregated'].keys():
            score = self._calculate_distinctiveness_score(country, 'country')
            country_scores.append({
                'unit': country,
                'type': 'country',
                'score': score,
                'coffee_count': self.data['country_aggregated'][country]['metadata']['total_coffees']
            })
        
        country_scores.sort(key=lambda x: x['score'], reverse=True)
        summary['top_distinctive_units'] = country_scores[:10]
        
        # Extract key findings from consensus
        strong_consensus = self.consensus_findings.get('strong_consensus', {})
        for key, findings in strong_consensus.items():
            if findings:
                unit_type, taxonomy = key.split('_')
                summary['key_findings'].append({
                    'finding': f"Strong consensus on {len(findings)} distinctive {taxonomy}-level flavors for {unit_type} units",
                    'examples': findings[:3]
                })
        
        # Find unique flavors
        if 'unique_flavors' in self.tfidf_results:
            for level, unit_types in self.tfidf_results['unique_flavors'].items():
                for unit_type, uniques in unit_types.items():
                    if uniques:
                        summary['unique_discoveries'].append({
                            'level': level,
                            'unit_type': unit_type,
                            'count': len(uniques),
                            'examples': uniques[:3]
                        })
        
        return summary
    
    def _calculate_distinctiveness_score(self, unit: str, unit_type: str) -> float:
        """Calculate overall distinctiveness score for a unit"""
        score = 0.0
        
        # Statistical significance contribution
        for level in ['family', 'genus', 'species']:
            stat_key = f'{unit_type}_{level}_significant'
            if stat_key in self.statistical_results:
                unit_sigs = self.statistical_results[stat_key][
                    self.statistical_results[stat_key]['unit_name'] == unit
                ]
                score += len(unit_sigs) * 2
        
        # TF-IDF contribution
        tfidf_fingerprints = self.tfidf_results.get('family_fingerprints', {})
        if unit_type + 's' in tfidf_fingerprints:
            unit_fp = tfidf_fingerprints[unit_type + 's'].get(unit, {})
            if 'top_flavors' in unit_fp:
                avg_tfidf = np.mean([f['tfidf_score'] for f in unit_fp['top_flavors']])
                score += avg_tfidf * 10
        
        # Hierarchical contribution
        if unit in self.hierarchical_results['profiles']:
            hier_score = self.hierarchical_results['profiles'][unit]['summary_metrics']['distinctiveness_score']
            score += hier_score
        
        return score
    
    def get_unit_comprehensive_profile(self, unit_name: str, unit_type: str) -> Dict[str, Any]:
        """Get comprehensive profile combining all analysis methods"""
        profile = {
            'unit_name': unit_name,
            'unit_type': unit_type,
            'overview': {},
            'statistical_findings': {},
            'tfidf_findings': {},
            'hierarchical_findings': {},
            'consensus_findings': {},
            'recommendations': []
        }
        
        # Get basic overview
        if unit_type == 'country':
            unit_data = self.data['country_aggregated'].get(unit_name, {})
        elif unit_type == 'region':
            unit_data = self.data['region_aggregated'].get(unit_name, {})
        else:
            unit_data = self.data['seller_aggregated'].get(unit_name, {})
        
        if unit_data:
            profile['overview'] = unit_data.get('metadata', {})
        
        # Get findings from each method
        for level in ['family', 'genus', 'species']:
            # Statistical
            stat_key = f'{unit_type}_{level}_significant'
            if stat_key in self.statistical_results:
                unit_stats = self.statistical_results[stat_key][
                    self.statistical_results[stat_key]['unit_name'] == unit_name
                ]
                if not unit_stats.empty:
                    profile['statistical_findings'][level] = unit_stats.to_dict('records')
            
            # TF-IDF
            tfidf_key = f'{level}_fingerprints'
            if tfidf_key in self.tfidf_results:
                fp = self.tfidf_results[tfidf_key].get(unit_type + 's', {}).get(unit_name, {})
                if fp:
                    profile['tfidf_findings'][level] = fp
        
        # Hierarchical
        if unit_name in self.hierarchical_results['profiles']:
            profile['hierarchical_findings'] = self.hierarchical_results['profiles'][unit_name]
        
        # Consensus
        profile['consensus_findings'] = self._get_unit_consensus(unit_name, unit_type)
        
        # Generate recommendations
        profile['recommendations'] = self._generate_recommendations(profile)
        
        return profile
    
    def _get_unit_consensus(self, unit_name: str, unit_type: str) -> Dict[str, List]:
        """Get consensus findings for a specific unit"""
        consensus = {'strong': [], 'moderate': []}
        
        for level in ['family', 'genus', 'species']:
            key = f'{unit_type}_{level}'
            
            # Strong consensus
            strong = self.consensus_findings.get('strong_consensus', {}).get(key, [])
            unit_strong = [f for f in strong if f['unit'] == unit_name]
            consensus['strong'].extend(unit_strong)
            
            # Moderate consensus
            moderate = self.consensus_findings.get('moderate_consensus', {}).get(key, [])
            unit_moderate = [f for f in moderate if f['unit'] == unit_name]
            consensus['moderate'].extend(unit_moderate)
        
        return consensus
    
    def _generate_recommendations(self, profile: Dict) -> List[str]:
        """Generate actionable recommendations based on profile"""
        recommendations = []
        
        # Check for strong distinctive flavors
        strong_consensus = profile['consensus_findings'].get('strong', [])
        if strong_consensus:
            flavors = [f['flavor'] for f in strong_consensus[:3]]
            recommendations.append(
                f"Focus marketing on distinctive flavors: {', '.join(flavors)}"
            )
        
        # Check for unique flavors
        tfidf_family = profile['tfidf_findings'].get('family', {})
        if tfidf_family and 'top_flavors' in tfidf_family:
            top_flavor = tfidf_family['top_flavors'][0]['flavor']
            recommendations.append(
                f"Highlight '{top_flavor}' as a signature characteristic"
            )
        
        # Check hierarchical patterns
        hier = profile.get('hierarchical_findings', {})
        if hier:
            types = hier.get('distinctiveness_types', {})
            if types.get('concentrated_distinctive'):
                recommendations.append(
                    "Consider developing specialty lines focusing on concentrated distinctive flavors"
                )
            elif types.get('broad_distinctive'):
                recommendations.append(
                    "Leverage broad flavor diversity for varied product offerings"
                )
        
        return recommendations


# Convenience function
@st.cache_data(ttl=3600)
def get_integrated_analysis():
    """Get complete integrated analysis results"""
    analyzer = IntegratedFlavorAnalyzer()
    return analyzer.run_all_analyses()