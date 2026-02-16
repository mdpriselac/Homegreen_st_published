"""
Price Analysis Module

Analyzes price distributions across origins, processing methods, varietals,
and flavor profiles. Uses non-parametric tests appropriate for skewed price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from collections import defaultdict


class PriceAnalyzer:
    """Analyze coffee pricing patterns across features"""

    MIN_GROUP_SIZE = 5

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.full_df = cross_feature_df.copy()
        # Work with coffees that have price data
        self.df = self.full_df[
            self.full_df['avg_price'].notna() &
            (self.full_df['avg_price'] > 0)
        ].copy()

    def compute_price_overview(self) -> Dict[str, Any]:
        """Overall price distribution statistics"""
        if self.df.empty:
            return {'has_data': False}

        prices = self.df['avg_price']
        return {
            'has_data': True,
            'total_with_price': len(self.df),
            'total_without_price': len(self.full_df) - len(self.df),
            'coverage_rate': len(self.df) / len(self.full_df) if len(self.full_df) > 0 else 0,
            'mean': float(prices.mean()),
            'median': float(prices.median()),
            'std': float(prices.std()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'q25': float(prices.quantile(0.25)),
            'q75': float(prices.quantile(0.75)),
            'histogram': self._compute_histogram(prices),
        }

    def price_by_category(self, category_col: str, label_col: str = None,
                          min_group_size: int = None) -> Dict[str, Any]:
        """
        Price distributions grouped by a categorical variable.
        Uses Kruskal-Wallis for overall significance.
        """
        if min_group_size is None:
            min_group_size = self.MIN_GROUP_SIZE
        if label_col is None:
            label_col = category_col

        if self.df.empty or category_col not in self.df.columns:
            return {'has_data': False, 'groups': []}

        group_stats = []
        kw_groups = []
        kw_names = []

        for name, group in self.df.groupby(category_col):
            if pd.isna(name) or not name:
                continue

            prices = group['avg_price'].dropna()
            if len(prices) < min_group_size:
                continue

            group_stats.append({
                'name': str(name),
                'count': len(prices),
                'mean': float(prices.mean()),
                'median': float(prices.median()),
                'std': float(prices.std()),
                'min': float(prices.min()),
                'max': float(prices.max()),
                'q25': float(prices.quantile(0.25)),
                'q75': float(prices.quantile(0.75)),
            })

            kw_groups.append(prices.values)
            kw_names.append(str(name))

        group_stats.sort(key=lambda x: x['median'], reverse=True)

        # Kruskal-Wallis test
        kw_result = self._kruskal_wallis(kw_groups, kw_names)

        return {
            'has_data': len(group_stats) > 0,
            'groups': group_stats,
            'kruskal_wallis': kw_result,
            'category': label_col,
        }

    def price_by_country(self) -> Dict[str, Any]:
        """Price distributions by country"""
        return self.price_by_category('country', 'Country')

    def price_by_process(self) -> Dict[str, Any]:
        """Price distributions by process method"""
        return self.price_by_category('process_type', 'Process Method')

    def price_by_varietal(self) -> Dict[str, Any]:
        """Price distributions by varietal (expanding multi-varietal coffees)"""
        if self.df.empty:
            return {'has_data': False, 'groups': []}

        # Expand varietals: each varietal gets its own row
        expanded_rows = []
        for _, row in self.df.iterrows():
            varietals = row.get('varietals', [])
            if not varietals or not isinstance(varietals, list):
                continue
            for v in varietals:
                if v and str(v).strip():
                    new_row = row.copy()
                    new_row['single_varietal'] = str(v).strip()
                    expanded_rows.append(new_row)

        if not expanded_rows:
            return {'has_data': False, 'groups': []}

        expanded_df = pd.DataFrame(expanded_rows)
        # Temporarily replace df for price_by_category
        original_df = self.df
        self.df = expanded_df
        result = self.price_by_category('single_varietal', 'Varietal')
        self.df = original_df
        return result

    def price_by_flavor(self, taxonomy_level: str = 'family') -> Dict[str, Any]:
        """
        Price correlations with flavor attributes.
        For each flavor, compares mean price of coffees WITH vs WITHOUT the flavor.
        """
        if self.df.empty:
            return {'has_data': False, 'flavors': []}

        flavor_col = f'flavor_{taxonomy_level.rstrip("s")}' if taxonomy_level.endswith('s') else f'flavor_{taxonomy_level}'
        # Map to the right column name
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        if list_col not in self.df.columns:
            return {'has_data': False, 'flavors': []}

        # Collect all unique flavors at this level
        all_flavors = set()
        for flavor_list in self.df[list_col]:
            if isinstance(flavor_list, list):
                all_flavors.update(flavor_list)

        flavor_stats = []
        for flavor in all_flavors:
            if not flavor:
                continue

            has_flavor = self.df[self.df[list_col].apply(
                lambda x: flavor in x if isinstance(x, list) else False
            )]
            lacks_flavor = self.df[self.df[list_col].apply(
                lambda x: flavor not in x if isinstance(x, list) else True
            )]

            if len(has_flavor) < self.MIN_GROUP_SIZE or len(lacks_flavor) < self.MIN_GROUP_SIZE:
                continue

            prices_with = has_flavor['avg_price']
            prices_without = lacks_flavor['avg_price']

            # Mann-Whitney U test
            try:
                u_stat, p_value = stats.mannwhitneyu(
                    prices_with, prices_without, alternative='two-sided'
                )
                is_significant = p_value < 0.05
            except Exception:
                u_stat, p_value, is_significant = None, None, False

            flavor_stats.append({
                'flavor': flavor,
                'count_with': len(has_flavor),
                'count_without': len(lacks_flavor),
                'mean_price_with': float(prices_with.mean()),
                'mean_price_without': float(prices_without.mean()),
                'median_price_with': float(prices_with.median()),
                'median_price_without': float(prices_without.median()),
                'price_difference': float(prices_with.median() - prices_without.median()),
                'price_ratio': float(prices_with.median() / prices_without.median()) if prices_without.median() > 0 else None,
                'p_value': float(p_value) if p_value is not None else None,
                'is_significant': is_significant,
            })

        # Sort by price difference
        flavor_stats.sort(key=lambda x: x['price_difference'], reverse=True)

        return {
            'has_data': len(flavor_stats) > 0,
            'flavors': flavor_stats,
            'taxonomy_level': taxonomy_level,
        }

    def find_premium_indicators(self) -> List[Dict[str, Any]]:
        """
        Rank features most associated with higher prices.
        Combines results from all categorical analyses.
        """
        indicators = []

        # Country effects
        country_result = self.price_by_country()
        if country_result['has_data']:
            for group in country_result['groups'][:5]:
                indicators.append({
                    'feature': 'Country',
                    'value': group['name'],
                    'median_price': group['median'],
                    'count': group['count'],
                    'type': 'origin',
                })

        # Process effects
        process_result = self.price_by_process()
        if process_result['has_data']:
            for group in process_result['groups'][:5]:
                indicators.append({
                    'feature': 'Process',
                    'value': group['name'],
                    'median_price': group['median'],
                    'count': group['count'],
                    'type': 'process',
                })

        # Flavor effects
        flavor_result = self.price_by_flavor('family')
        if flavor_result['has_data']:
            sig_flavors = [f for f in flavor_result['flavors'] if f['is_significant']]
            for f in sig_flavors[:5]:
                indicators.append({
                    'feature': 'Flavor Family',
                    'value': f['flavor'],
                    'median_price': f['median_price_with'],
                    'count': f['count_with'],
                    'price_premium': f['price_difference'],
                    'type': 'flavor',
                })

        # Sort by median price descending
        indicators.sort(key=lambda x: x['median_price'], reverse=True)
        return indicators

    def _compute_histogram(self, series: pd.Series, bins: int = 20) -> Dict[str, List]:
        """Compute histogram data for frontend rendering"""
        counts, bin_edges = np.histogram(series.dropna(), bins=bins)
        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_labels': [
                f"${bin_edges[i]:.2f}-${bin_edges[i+1]:.2f}"
                for i in range(len(bin_edges) - 1)
            ]
        }

    def _kruskal_wallis(self, groups: List[np.ndarray],
                        group_names: List[str]) -> Dict[str, Any]:
        """Run Kruskal-Wallis test"""
        if len(groups) < 2:
            return {'has_data': False}

        try:
            h_stat, p_value = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            k = len(groups)
            epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else 0

            return {
                'has_data': True,
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'is_significant': p_value < 0.05,
                'n_groups': k,
                'effect_size': float(epsilon_sq),
            }
        except Exception:
            return {'has_data': False}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all price analyses and return combined results"""
        return {
            'overview': self.compute_price_overview(),
            'by_country': self.price_by_country(),
            'by_process': self.price_by_process(),
            'by_varietal': self.price_by_varietal(),
            'by_flavor_family': self.price_by_flavor('family'),
            'by_flavor_genus': self.price_by_flavor('genus'),
            'by_flavor_species': self.price_by_flavor('species'),
            'premium_indicators': self.find_premium_indicators(),
        }
