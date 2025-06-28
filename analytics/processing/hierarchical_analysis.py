"""
Hierarchical multi-level distinctive features analysis

This module analyzes flavor distinctiveness across the taxonomic hierarchy
(family -> genus -> species) to identify patterns at different granularity levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class HierarchicalFlavorAnalyzer:
    """Analyze flavor distinctiveness across taxonomic hierarchy"""
    
    def __init__(self, hierarchical_data: Dict[str, Dict[str, Any]],
                 statistical_results: Dict[str, pd.DataFrame] = None,
                 tfidf_results: Dict[str, Any] = None):
        """
        Initialize with hierarchical format data and optional analysis results
        
        Args:
            hierarchical_data: Dictionary from coffee_data_extractor.prepare_hierarchical_format()
            statistical_results: Optional results from statistical analysis
            tfidf_results: Optional results from TF-IDF analysis
        """
        self.data = hierarchical_data
        self.statistical_results = statistical_results or {}
        self.tfidf_results = tfidf_results or {}
    
    def analyze_cascade_patterns(self, unit_name: str) -> Dict[str, Any]:
        """
        Analyze how distinctiveness cascades through taxonomy levels
        
        Args:
            unit_name: Name of the unit to analyze
            
        Returns:
            Dictionary with cascade analysis results
        """
        if unit_name not in self.data:
            return {'error': f'Unit {unit_name} not found'}
        
        unit_data = self.data[unit_name]
        cascade_patterns = {
            'unit_name': unit_name,
            'unit_type': unit_data['unit_type'],
            'patterns': []
        }
        
        # Analyze each family
        for family, family_data in unit_data['family_level'].items():
            family_pattern = {
                'family': family,
                'family_frequency': family_data['frequency'],
                'family_count': family_data['count'],
                'genera': [],
                'specificity_score': 0.0
            }
            
            # Find genera under this family
            family_genera = []
            for genus, genus_data in unit_data['genus_level'].items():
                if genus_data.get('parent_family') == family:
                    genus_info = {
                        'genus': genus,
                        'genus_frequency': genus_data['frequency'],
                        'genus_count': genus_data['count'],
                        'species': []
                    }
                    
                    # Find species under this genus
                    for species, species_data in unit_data['species_level'].items():
                        if species_data.get('parent_genus') == genus:
                            genus_info['species'].append({
                                'species': species,
                                'species_frequency': species_data['frequency'],
                                'species_count': species_data['count']
                            })
                    
                    family_genera.append(genus_info)
            
            family_pattern['genera'] = family_genera
            
            # Calculate specificity score
            family_pattern['specificity_score'] = self._calculate_specificity_score(
                family_pattern
            )
            
            cascade_patterns['patterns'].append(family_pattern)
        
        # Sort by family frequency
        cascade_patterns['patterns'].sort(
            key=lambda x: x['family_frequency'], 
            reverse=True
        )
        
        return cascade_patterns
    
    def _calculate_specificity_score(self, family_pattern: Dict) -> float:
        """
        Calculate how specific/focused the distinctiveness is
        
        Higher score = more specific (concentrated in few species)
        Lower score = more broad (spread across many species)
        """
        if not family_pattern['genera']:
            return 0.0
        
        # Count total species and their distribution
        species_counts = []
        for genus in family_pattern['genera']:
            for species in genus['species']:
                species_counts.append(species['species_count'])
        
        if not species_counts:
            return 0.0
        
        # Calculate concentration using Gini coefficient
        species_counts = sorted(species_counts)
        n = len(species_counts)
        cumsum = np.cumsum(species_counts)
        
        # Gini coefficient
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return gini
    
    def identify_distinctiveness_types(self, unit_name: str, 
                                     min_frequency: float = 0.05) -> Dict[str, List[Dict]]:
        """
        Classify distinctiveness patterns into types
        
        Args:
            unit_name: Unit to analyze
            min_frequency: Minimum frequency to consider
            
        Returns:
            Dictionary of distinctiveness types
        """
        if unit_name not in self.data:
            return {}
        
        unit_data = self.data[unit_name]
        types = {
            'broad_distinctive': [],      # High at family, spread across species
            'narrow_distinctive': [],     # Low at family, high at specific species
            'concentrated_distinctive': [],  # High at all levels but few species
            'hierarchical_distinctive': []   # Distinctive pattern through hierarchy
        }
        
        # Analyze each family
        for family, family_data in unit_data['family_level'].items():
            if family_data['frequency'] < min_frequency:
                continue
            
            # Get hierarchy tree for this family
            family_tree = unit_data['hierarchy_tree'].get(family, {})
            
            # Count genera and species
            n_genera = len(family_tree)
            n_species = sum(len(species_list) for species_list in family_tree.values())
            
            # Calculate average frequencies at each level
            genus_freqs = []
            species_freqs = []
            
            for genus, genus_data in unit_data['genus_level'].items():
                if genus_data.get('parent_family') == family:
                    genus_freqs.append(genus_data['frequency'])
                    
                    for species, species_data in unit_data['species_level'].items():
                        if species_data.get('parent_genus') == genus:
                            species_freqs.append(species_data['frequency'])
            
            avg_genus_freq = np.mean(genus_freqs) if genus_freqs else 0
            avg_species_freq = np.mean(species_freqs) if species_freqs else 0
            max_species_freq = max(species_freqs) if species_freqs else 0
            
            # Classify based on patterns
            pattern = {
                'family': family,
                'family_frequency': family_data['frequency'],
                'n_genera': n_genera,
                'n_species': n_species,
                'avg_genus_frequency': avg_genus_freq,
                'avg_species_frequency': avg_species_freq,
                'max_species_frequency': max_species_freq
            }
            
            # Broad distinctive: high family freq, many species, low individual species freq
            if (family_data['frequency'] > 0.15 and n_species > 5 and 
                max_species_freq < family_data['frequency'] / 3):
                types['broad_distinctive'].append(pattern)
            
            # Narrow distinctive: specific species much higher than family average
            elif max_species_freq > avg_species_freq * 3:
                types['narrow_distinctive'].append(pattern)
            
            # Concentrated: high frequency but few species
            elif family_data['frequency'] > 0.1 and n_species <= 3:
                types['concentrated_distinctive'].append(pattern)
            
            # Hierarchical: consistent high frequency through levels
            elif (family_data['frequency'] > 0.1 and 
                  avg_genus_freq > 0.08 and 
                  avg_species_freq > 0.05):
                types['hierarchical_distinctive'].append(pattern)
        
        return types
    
    def compare_with_statistical_significance(self, unit_name: str, 
                                            unit_type: str) -> Dict[str, Any]:
        """
        Compare hierarchical patterns with statistical significance results
        
        Args:
            unit_name: Unit to analyze
            unit_type: Type of unit
            
        Returns:
            Comparison results
        """
        hierarchical_cascade = self.analyze_cascade_patterns(unit_name)
        
        # Get statistical results if available
        sig_results = {}
        for level in ['family', 'genus', 'species']:
            key = f'{unit_type}_{level}_significant'
            if key in self.statistical_results:
                unit_sigs = self.statistical_results[key][
                    self.statistical_results[key]['unit_name'] == unit_name
                ]
                sig_results[level] = unit_sigs
        
        # Compare patterns
        validated_patterns = []
        
        for pattern in hierarchical_cascade.get('patterns', []):
            family = pattern['family']
            
            # Check if family is statistically significant
            family_sig = None
            if 'family' in sig_results and not sig_results['family'].empty:
                family_matches = sig_results['family'][
                    sig_results['family']['flavor'] == family
                ]
                if not family_matches.empty:
                    family_sig = family_matches.iloc[0]
            
            validated = {
                'family': family,
                'hierarchical_frequency': pattern['family_frequency'],
                'statistically_significant': family_sig is not None,
                'statistical_details': {}
            }
            
            if family_sig is not None:
                validated['statistical_details'] = {
                    'p_value': family_sig['p_value_corrected'],
                    'odds_ratio': family_sig['odds_ratio'],
                    'effect_size': family_sig['cramers_v']
                }
            
            # Check child levels
            validated['genus_validation'] = []
            validated['species_validation'] = []
            
            for genus_info in pattern['genera']:
                genus = genus_info['genus']
                genus_sig = None
                
                if 'genus' in sig_results and not sig_results['genus'].empty:
                    genus_matches = sig_results['genus'][
                        sig_results['genus']['flavor'] == genus
                    ]
                    if not genus_matches.empty:
                        genus_sig = genus_matches.iloc[0]
                
                validated['genus_validation'].append({
                    'genus': genus,
                    'significant': genus_sig is not None,
                    'supports_parent': genus_sig is not None and family_sig is not None
                })
            
            validated_patterns.append(validated)
        
        return {
            'unit_name': unit_name,
            'validated_patterns': validated_patterns,
            'summary': self._summarize_validation(validated_patterns)
        }
    
    def _summarize_validation(self, validated_patterns: List[Dict]) -> Dict[str, int]:
        """Summarize validation results"""
        summary = {
            'total_patterns': len(validated_patterns),
            'statistically_significant': sum(
                1 for p in validated_patterns if p['statistically_significant']
            ),
            'hierarchically_consistent': 0,
            'broad_support': 0,
            'narrow_support': 0
        }
        
        for pattern in validated_patterns:
            if pattern['statistically_significant']:
                # Check if children support parent
                supporting_genera = sum(
                    1 for g in pattern['genus_validation'] if g['supports_parent']
                )
                if supporting_genera > 0:
                    summary['hierarchically_consistent'] += 1
                    if supporting_genera >= 3:
                        summary['broad_support'] += 1
                    else:
                        summary['narrow_support'] += 1
        
        return summary
    
    def generate_hierarchical_profile(self, unit_name: str, 
                                    include_statistics: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive hierarchical profile for a unit
        
        Args:
            unit_name: Unit to profile
            include_statistics: Whether to include statistical validation
            
        Returns:
            Complete hierarchical profile
        """
        profile = {
            'unit_name': unit_name,
            'unit_type': self.data[unit_name]['unit_type'],
            'total_coffees': self.data[unit_name]['total_coffees'],
            'total_flavor_instances': self.data[unit_name]['total_flavor_instances']
        }
        
        # Get cascade patterns
        profile['cascade_patterns'] = self.analyze_cascade_patterns(unit_name)
        
        # Get distinctiveness types
        profile['distinctiveness_types'] = self.identify_distinctiveness_types(unit_name)
        
        # Add statistical validation if requested
        if include_statistics and self.statistical_results:
            profile['statistical_validation'] = self.compare_with_statistical_significance(
                unit_name, 
                self.data[unit_name]['unit_type']
            )
        
        # Calculate profile summary metrics
        profile['summary_metrics'] = self._calculate_profile_metrics(profile)
        
        return profile
    
    def _calculate_profile_metrics(self, profile: Dict) -> Dict[str, float]:
        """Calculate summary metrics for a profile"""
        metrics = {
            'flavor_diversity': 0.0,
            'hierarchical_depth': 0.0,
            'concentration_index': 0.0,
            'distinctiveness_score': 0.0
        }
        
        # Flavor diversity (Shannon entropy at family level)
        family_freqs = [
            f['family_frequency'] 
            for f in profile['cascade_patterns']['patterns']
        ]
        if family_freqs:
            family_freqs = np.array(family_freqs)
            family_freqs = family_freqs[family_freqs > 0]
            metrics['flavor_diversity'] = -np.sum(
                family_freqs * np.log(family_freqs)
            )
        
        # Hierarchical depth (average species per family)
        total_species = 0
        families_with_species = 0
        for pattern in profile['cascade_patterns']['patterns']:
            species_count = sum(
                len(g['species']) for g in pattern['genera']
            )
            if species_count > 0:
                total_species += species_count
                families_with_species += 1
        
        if families_with_species > 0:
            metrics['hierarchical_depth'] = total_species / families_with_species
        
        # Concentration index (how concentrated in top families)
        if family_freqs.size > 0:
            top_3_freq = sum(sorted(family_freqs, reverse=True)[:3])
            metrics['concentration_index'] = top_3_freq
        
        # Overall distinctiveness score
        dist_types = profile['distinctiveness_types']
        metrics['distinctiveness_score'] = (
            len(dist_types.get('hierarchical_distinctive', [])) * 3 +
            len(dist_types.get('concentrated_distinctive', [])) * 2 +
            len(dist_types.get('broad_distinctive', [])) * 1 +
            len(dist_types.get('narrow_distinctive', [])) * 1.5
        )
        
        return metrics


def run_full_hierarchical_analysis(hierarchical_data: Dict[str, Dict[str, Any]],
                                 statistical_results: Dict[str, pd.DataFrame] = None,
                                 tfidf_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run complete hierarchical analysis
    
    Args:
        hierarchical_data: Hierarchical format data from extraction
        statistical_results: Optional statistical analysis results
        tfidf_results: Optional TF-IDF analysis results
        
    Returns:
        Dictionary of analysis results
    """
    analyzer = HierarchicalFlavorAnalyzer(
        hierarchical_data, 
        statistical_results, 
        tfidf_results
    )
    
    results = {
        'profiles': {},
        'distinctiveness_summary': defaultdict(list),
        'top_hierarchical_units': []
    }
    
    # Generate profiles for all units
    for unit_name in hierarchical_data.keys():
        profile = analyzer.generate_hierarchical_profile(
            unit_name, 
            include_statistics=bool(statistical_results)
        )
        results['profiles'][unit_name] = profile
        
        # Collect distinctiveness patterns
        for dist_type, patterns in profile['distinctiveness_types'].items():
            if patterns:
                results['distinctiveness_summary'][dist_type].append({
                    'unit_name': unit_name,
                    'unit_type': profile['unit_type'],
                    'pattern_count': len(patterns),
                    'examples': patterns[:3]  # Top 3 examples
                })
    
    # Rank units by distinctiveness score
    unit_scores = [
        {
            'unit_name': name,
            'unit_type': profile['unit_type'],
            'distinctiveness_score': profile['summary_metrics']['distinctiveness_score'],
            'diversity': profile['summary_metrics']['flavor_diversity'],
            'depth': profile['summary_metrics']['hierarchical_depth']
        }
        for name, profile in results['profiles'].items()
    ]
    
    unit_scores.sort(key=lambda x: x['distinctiveness_score'], reverse=True)
    results['top_hierarchical_units'] = unit_scores[:20]
    
    # Store analyzer
    results['analyzer'] = analyzer
    
    return results