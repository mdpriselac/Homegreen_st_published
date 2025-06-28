"""
Statistical significance analysis for coffee flavor distinctiveness

This module implements various statistical tests to identify which flavors
are significantly associated with specific geographic regions or sellers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class FlavorStatisticalAnalyzer:
    """Analyze statistical significance of flavor-region associations"""
    
    def __init__(self, contingency_data: pd.DataFrame):
        """
        Initialize with contingency format data
        
        Args:
            contingency_data: DataFrame from coffee_data_extractor.prepare_contingency_format()
        """
        self.data = contingency_data
        self._prepare_flavor_columns()
    
    def _prepare_flavor_columns(self):
        """Identify and categorize flavor columns"""
        self.has_columns = [col for col in self.data.columns if col.startswith('has_')]
        self.count_columns = [col for col in self.data.columns if col.startswith('count_')]
        self.freq_columns = [col for col in self.data.columns if col.startswith('freq_')]
        
        # Extract unique flavors at each level
        self.flavors_by_level = {
            'family': [],
            'genus': [],
            'species': []
        }
        
        for col in self.has_columns:
            parts = col.split('_', 2)  # has_level_flavor
            if len(parts) >= 3:
                level = parts[1]
                flavor = parts[2]
                if level in self.flavors_by_level:
                    self.flavors_by_level[level].append(flavor)
        
        # Remove duplicates
        for level in self.flavors_by_level:
            self.flavors_by_level[level] = list(set(self.flavors_by_level[level]))
    
    def build_contingency_table(self, unit_name: str, unit_type: str, 
                               flavor: str, taxonomy_level: str) -> np.ndarray:
        """
        Build 2x2 contingency table for a unit-flavor combination
        
        Args:
            unit_name: Name of the geographic unit or seller
            unit_type: Type of unit ('country', 'region', 'seller')
            flavor: Flavor name
            taxonomy_level: Taxonomy level ('family', 'genus', 'species')
            
        Returns:
            2x2 numpy array [[a, b], [c, d]] where:
            - a: Has flavor & in unit
            - b: No flavor & in unit
            - c: Has flavor & outside unit
            - d: No flavor & outside unit
        """
        # Filter data by unit type
        type_data = self.data[self.data['unit_type'] == unit_type].copy()
        
        # Get has column name
        has_col = f'has_{taxonomy_level}_{flavor}'
        if has_col not in type_data.columns:
            return None
        
        # Create masks
        in_unit = type_data['unit'] == unit_name
        has_flavor = type_data[has_col] == True
        
        # Build contingency table
        a = ((in_unit) & (has_flavor)).sum()  # In unit with flavor
        b = ((in_unit) & (~has_flavor)).sum()  # In unit without flavor
        c = ((~in_unit) & (has_flavor)).sum()  # Outside unit with flavor
        d = ((~in_unit) & (~has_flavor)).sum()  # Outside unit without flavor
        
        return np.array([[a, b], [c, d]])
    
    def perform_statistical_test(self, contingency_table: np.ndarray, 
                               min_expected: float = 5.0) -> Dict[str, float]:
        """
        Perform appropriate statistical test based on expected frequencies
        
        Args:
            contingency_table: 2x2 contingency table
            min_expected: Minimum expected frequency for chi-square test
            
        Returns:
            Dict with test results including p-value, test used, etc.
        """
        if contingency_table is None:
            return {'p_value': np.nan, 'test_used': 'none'}
        
        # Calculate expected frequencies
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)
        total = contingency_table.sum()
        
        if total == 0:
            return {'p_value': np.nan, 'test_used': 'none'}
        
        expected = np.outer(row_totals, col_totals) / total
        
        # Choose test based on expected frequencies
        if np.min(expected) < min_expected or total < 20:
            # Use Fisher's exact test
            try:
                oddsratio, p_value = stats.fisher_exact(contingency_table)
                test_used = 'fisher'
            except:
                p_value = np.nan
                oddsratio = np.nan
                test_used = 'fisher_failed'
        else:
            # Use Chi-square test with Yates' correction
            try:
                chi2, p_value, dof, expected_freq = stats.chi2_contingency(
                    contingency_table, correction=True
                )
                # Calculate odds ratio manually
                a, b = contingency_table[0]
                c, d = contingency_table[1]
                if b > 0 and c > 0:
                    oddsratio = (a * d) / (b * c)
                else:
                    oddsratio = np.inf if a > 0 and d > 0 else np.nan
                test_used = 'chi2_yates'
            except:
                p_value = np.nan
                oddsratio = np.nan
                test_used = 'chi2_failed'
        
        # Calculate Cramér's V for effect size
        try:
            chi2_val = ((contingency_table[0,0] * contingency_table[1,1] - 
                        contingency_table[0,1] * contingency_table[1,0]) ** 2 * total) / \
                      (row_totals[0] * row_totals[1] * col_totals[0] * col_totals[1])
            cramers_v = np.sqrt(chi2_val / total)
        except:
            cramers_v = np.nan
        
        # Calculate confidence interval for odds ratio
        try:
            if not np.isnan(oddsratio) and oddsratio > 0:
                log_or = np.log(oddsratio)
                se_log_or = np.sqrt(sum(1/x if x > 0 else np.inf 
                                       for x in contingency_table.flatten()))
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
            else:
                ci_lower = np.nan
                ci_upper = np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan
        
        return {
            'p_value': p_value,
            'test_used': test_used,
            'odds_ratio': oddsratio,
            'odds_ratio_ci_lower': ci_lower,
            'odds_ratio_ci_upper': ci_upper,
            'cramers_v': cramers_v,
            'contingency_table': contingency_table.tolist()
        }
    
    def analyze_all_associations(self, unit_type: str = 'country', 
                               taxonomy_level: str = 'family',
                               min_occurrences: int = 5) -> pd.DataFrame:
        """
        Analyze all unit-flavor associations for a given unit type and taxonomy level
        
        Args:
            unit_type: Type of unit to analyze
            taxonomy_level: Taxonomy level to analyze
            min_occurrences: Minimum occurrences of flavor to include
            
        Returns:
            DataFrame with statistical test results
        """
        results = []
        
        # Filter data by unit type
        type_data = self.data[self.data['unit_type'] == unit_type]
        units = type_data['unit'].unique()
        
        # Get flavors for this taxonomy level
        flavors = self.flavors_by_level[taxonomy_level]
        
        for unit in units:
            for flavor in flavors:
                # Check if flavor meets minimum occurrence threshold
                count_col = f'count_{taxonomy_level}_{flavor}'
                if count_col in type_data.columns:
                    unit_data = type_data[type_data['unit'] == unit]
                    if not unit_data.empty:
                        flavor_count = unit_data[count_col].iloc[0]
                        if flavor_count < min_occurrences:
                            continue
                
                # Build contingency table
                cont_table = self.build_contingency_table(
                    unit, unit_type, flavor, taxonomy_level
                )
                
                if cont_table is not None:
                    # Perform statistical test
                    test_results = self.perform_statistical_test(cont_table)
                    
                    # Add metadata
                    test_results.update({
                        'unit_name': unit,
                        'unit_type': unit_type,
                        'taxonomy_level': taxonomy_level,
                        'flavor': flavor,
                        'unit_occurrences': cont_table[0, 0],
                        'unit_total': cont_table[0].sum(),
                        'global_occurrences': cont_table[:, 0].sum(),
                        'global_total': cont_table.sum()
                    })
                    
                    results.append(test_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction
        if len(results_df) > 0 and 'p_value' in results_df.columns:
            # Remove NaN p-values for correction
            valid_mask = ~results_df['p_value'].isna()
            if valid_mask.sum() > 0:
                # Apply Benjamini-Hochberg FDR correction
                rejected, corrected_pvals, _, _ = multipletests(
                    results_df.loc[valid_mask, 'p_value'],
                    method='fdr_bh',
                    alpha=0.05
                )
                results_df.loc[valid_mask, 'p_value_corrected'] = corrected_pvals
                results_df.loc[valid_mask, 'is_significant'] = rejected
            
            # Fill NaN values
            results_df['p_value_corrected'] = results_df['p_value_corrected'].fillna(np.nan)
            results_df['is_significant'] = results_df['is_significant'].fillna(False).infer_objects(copy=False)
        
        return results_df
    
    def get_significant_associations(self, results_df: pd.DataFrame,
                                   p_threshold: float = 0.05,
                                   effect_size_threshold: float = 0.1,
                                   odds_ratio_threshold: float = 2.0) -> pd.DataFrame:
        """
        Filter results to get only significant associations
        
        Args:
            results_df: Results from analyze_all_associations
            p_threshold: P-value threshold (uses corrected p-values)
            effect_size_threshold: Minimum Cramér's V
            odds_ratio_threshold: Minimum odds ratio
            
        Returns:
            Filtered DataFrame with significant associations
        """
        # Filter by corrected p-value
        sig_df = results_df[
            (results_df['p_value_corrected'] < p_threshold) &
            (results_df['cramers_v'] > effect_size_threshold) &
            (results_df['odds_ratio'] > odds_ratio_threshold)
        ].copy()
        
        # Add interpretation
        sig_df['interpretation'] = sig_df.apply(
            lambda row: self._interpret_result(row), axis=1
        )
        
        # Sort by significance and effect size
        sig_df['combined_score'] = (
            -np.log10(sig_df['p_value_corrected'] + 1e-10) * 
            sig_df['cramers_v'] * 
            np.log(sig_df['odds_ratio'] + 1)
        )
        sig_df = sig_df.sort_values('combined_score', ascending=False)
        
        return sig_df
    
    def _interpret_result(self, row: pd.Series) -> str:
        """Generate human-readable interpretation of results"""
        if row['odds_ratio'] > 1:
            times_more = f"{row['odds_ratio']:.1f}x"
            interpretation = (
                f"{row['flavor']} is {times_more} more likely in {row['unit_name']} "
                f"(p={row['p_value_corrected']:.4f}, effect={row['cramers_v']:.3f})"
            )
        else:
            times_less = f"{1/row['odds_ratio']:.1f}x" if row['odds_ratio'] > 0 else "significantly"
            interpretation = (
                f"{row['flavor']} is {times_less} less likely in {row['unit_name']} "
                f"(p={row['p_value_corrected']:.4f}, effect={row['cramers_v']:.3f})"
            )
        
        return interpretation
    
    def analyze_hierarchical_significance(self, unit_name: str, unit_type: str) -> Dict[str, Any]:
        """
        Analyze significance across taxonomy hierarchy for a specific unit
        
        Args:
            unit_name: Name of the unit to analyze
            unit_type: Type of unit
            
        Returns:
            Hierarchical analysis results
        """
        hierarchical_results = {}
        
        for taxonomy_level in ['family', 'genus', 'species']:
            # Get all flavors at this level
            flavors = self.flavors_by_level[taxonomy_level]
            level_results = []
            
            for flavor in flavors:
                cont_table = self.build_contingency_table(
                    unit_name, unit_type, flavor, taxonomy_level
                )
                
                if cont_table is not None and cont_table[0, 0] > 0:
                    test_results = self.perform_statistical_test(cont_table)
                    test_results.update({
                        'flavor': flavor,
                        'occurrences': cont_table[0, 0],
                        'unit_frequency': cont_table[0, 0] / cont_table[0].sum() if cont_table[0].sum() > 0 else 0
                    })
                    level_results.append(test_results)
            
            # Sort by p-value
            level_results.sort(key=lambda x: x.get('p_value', 1))
            hierarchical_results[taxonomy_level] = level_results
        
        return {
            'unit_name': unit_name,
            'unit_type': unit_type,
            'results_by_level': hierarchical_results
        }


def run_full_statistical_analysis(contingency_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Run complete statistical analysis across all unit types and taxonomy levels
    
    Args:
        contingency_data: Contingency format data from extraction
        
    Returns:
        Dictionary of results DataFrames by analysis type
    """
    analyzer = FlavorStatisticalAnalyzer(contingency_data)
    results = {}
    
    # Analyze each combination
    for unit_type in ['country', 'region', 'seller']:
        for taxonomy_level in ['family', 'genus', 'species']:
            key = f'{unit_type}_{taxonomy_level}'
            
            # Run analysis
            all_results = analyzer.analyze_all_associations(
                unit_type=unit_type,
                taxonomy_level=taxonomy_level,
                min_occurrences=3
            )
            
            # Get significant results
            sig_results = analyzer.get_significant_associations(
                all_results,
                p_threshold=0.05,
                effect_size_threshold=0.1,
                odds_ratio_threshold=1.5
            )
            
            results[f'{key}_all'] = all_results
            results[f'{key}_significant'] = sig_results
    
    return results