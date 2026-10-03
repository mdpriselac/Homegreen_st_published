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

from analytics.processing.varietals import expand_varietals
from analytics.processing.stat_utils import apply_bh, eta_squared_h


class PriceAnalyzer:
    """Analyze coffee pricing patterns across features"""

    MIN_GROUP_SIZE = 5

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.full_df = cross_feature_df.copy()
        # Work with coffees that have price data
        self.df = self.full_df[
            self.full_df['price_per_lb'].notna() &
            (self.full_df['price_per_lb'] > 0)
        ].copy()

    def compute_price_overview(self) -> Dict[str, Any]:
        """Overall price distribution statistics"""
        if self.df.empty:
            return {'has_data': False}

        prices = self.df['price_per_lb']
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
            **self._window_info(),
        }

    def _window_info(self) -> Dict[str, Any]:
        """Time window covered by the priced coffees, and active/expired mix"""
        info: Dict[str, Any] = {}
        if 'first_observed' in self.df.columns:
            first = pd.to_datetime(self.df['first_observed'], errors='coerce').min()
            if pd.notna(first):
                info['window_start'] = first.strftime('%Y-%m-%d')
        if 'last_observed' in self.df.columns:
            last = pd.to_datetime(self.df['last_observed'], errors='coerce').max()
            if pd.notna(last):
                info['window_end'] = last.strftime('%Y-%m-%d')
        if 'is_active' in self.df.columns:
            active = self.df['is_active'].fillna(False).astype(bool)
            info['n_active'] = int(active.sum())
            info['n_expired'] = int((~active).sum())
        return info

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

            prices = group['price_per_lb'].dropna()
            if len(prices) < min_group_size:
                continue

            rest = self.df.loc[~self.df.index.isin(group.index), 'price_per_lb'].dropna()
            p_group = None
            if len(rest) >= min_group_size:
                try:
                    p_group = float(stats.mannwhitneyu(prices, rest, alternative='two-sided')[1])
                except Exception:
                    p_group = None

            group_stats.append({
                'name': str(name),
                'p_value': p_group,   # Mann-Whitney: group vs all other coffees
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

        # BH across the groups of this category (raw p kept in p_value)
        apply_bh(group_stats)
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
        """Price distributions by varietal (single-varietal coffees only)"""
        if self.df.empty:
            return {'has_data': False, 'groups': []}

        # price_by_category runs a Kruskal-Wallis test, so use single-varietal
        # coffees only (independent rows, no duplicate per-varietal copies).
        expanded_df = expand_varietals(self.df, 'single')
        if expanded_df.empty:
            return {'has_data': False, 'groups': []}

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

            prices_with = has_flavor['price_per_lb']
            prices_without = lacks_flavor['price_per_lb']

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

        # BH across all flavors tested at this taxonomy level; significance = q < 0.05
        apply_bh(flavor_stats)

        # Sort by price difference
        flavor_stats.sort(key=lambda x: x['price_difference'], reverse=True)

        return {
            'has_data': len(flavor_stats) > 0,
            'flavors': flavor_stats,
            'taxonomy_level': taxonomy_level,
        }

    def _group_vs_rest(self, category_col: str, feature: str, kind: str,
                       overall_median: float) -> List[Dict[str, Any]]:
        """Per-group premium (group median - overall median) with a Mann-Whitney
        test of the group vs all other coffees, BH-corrected across the groups."""
        rows = []
        for name, group in self.df.groupby(category_col):
            if pd.isna(name) or not name:
                continue
            prices = group['price_per_lb'].dropna()
            rest = self.df.loc[~self.df.index.isin(group.index), 'price_per_lb'].dropna()
            if len(prices) < self.MIN_GROUP_SIZE or len(rest) < self.MIN_GROUP_SIZE:
                continue
            try:
                p_value = float(stats.mannwhitneyu(prices, rest, alternative='two-sided')[1])
            except Exception:
                continue
            rows.append({
                'feature': feature,
                'value': str(name),
                'median_price': float(prices.median()),
                'price_premium': float(prices.median() - overall_median),
                'count': int(len(prices)),
                'p_value': p_value,
                'type': kind,
            })
        apply_bh(rows)
        return rows

    def find_premium_indicators(self) -> List[Dict[str, Any]]:
        """
        Rank features most associated with higher prices.

        premium = group median - overall median. Only positive, statistically
        significant premiums are returned, sorted by premium descending.
        Significance everywhere is Benjamini-Hochberg adjusted q < 0.05, applied
        within each feature's candidate set (countries, processes, flavor
        families). Country/process: group-vs-rest Mann-Whitney. Flavor: the
        with/without Mann-Whitney p-value from price_by_flavor.
        """
        if self.df.empty:
            return []
        overall_median = float(self.df['price_per_lb'].median())
        candidates = []
        candidates += self._group_vs_rest('country', 'Country', 'origin', overall_median)
        candidates += self._group_vs_rest('process_type', 'Process', 'process', overall_median)

        flavor_result = self.price_by_flavor('family')
        if flavor_result['has_data']:
            flavor_rows = [{
                'feature': 'Flavor Family',
                'value': f['flavor'],
                'median_price': f['median_price_with'],
                'price_premium': f['median_price_with'] - overall_median,
                'count': f['count_with'],
                'p_value': f['p_value'],
                'q_value': f['q_value'],          # BH over all flavor families (price_by_flavor)
                'is_significant': f['is_significant'],
                'type': 'flavor',
            } for f in flavor_result['flavors'] if f.get('p_value') is not None]
            candidates += flavor_rows

        indicators = [c for c in candidates
                      if c.get('is_significant') and c['price_premium'] > 0]
        indicators.sort(key=lambda x: x['price_premium'], reverse=True)
        return indicators

    def _compute_histogram(self, series: pd.Series, bins: int = 20) -> Dict[str, List]:
        """Compute histogram data for frontend rendering"""
        """Histogram with log-spaced bins (prices are right-skewed)"""
        values = series.dropna()
        values = values[values > 0]
        if values.empty:
            return {'counts': [], 'bin_edges': [], 'bin_labels': [], 'log_bins': True}
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            hi = lo * 1.01
        counts, bin_edges = np.histogram(values, bins=np.geomspace(lo, hi, bins + 1))
        return {
            'log_bins': True,
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
            eta2 = eta_squared_h(h_stat, n, k)

            return {
                'has_data': True,
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'is_significant': bool(p_value < 0.05),
                'n_groups': k,
                'effect_size': float(eta2),
                'eta_squared_h': float(eta2),
                'effect_size_metric': 'eta2_h',
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
