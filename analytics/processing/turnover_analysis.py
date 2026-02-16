"""
Seller Turnover and Coffee Lifespan Analysis

Analyzes how long coffees remain listed, turnover rates by seller,
and lifespan patterns across origins, process types, and price ranges.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from datetime import datetime


class SellerTurnoverAnalyzer:
    """Analyze coffee listing lifespans and seller turnover patterns"""

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.df = cross_feature_df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare temporal data for analysis"""
        self.df['first_observed'] = pd.to_datetime(self.df['first_observed'], errors='coerce')
        self.df['last_observed'] = pd.to_datetime(self.df['last_observed'], errors='coerce')

        # Compute lifespan only for coffees with both dates
        mask = self.df['first_observed'].notna() & self.df['last_observed'].notna()
        self.df.loc[mask, 'lifespan_days'] = (
            self.df.loc[mask, 'last_observed'] - self.df.loc[mask, 'first_observed']
        ).dt.days

        # Expired coffees: have definitive end dates
        self.expired_df = self.df[
            (self.df['is_active'] == False) &
            self.df['lifespan_days'].notna() &
            (self.df['lifespan_days'] >= 0)
        ].copy()

        # All coffees with valid dates (for seasonal analysis)
        self.dated_df = self.df[self.df['first_observed'].notna()].copy()

    def compute_coffee_lifespans(self) -> Dict[str, Any]:
        """Compute overall lifespan statistics for expired coffees"""
        if self.expired_df.empty:
            return {'has_data': False}

        lifespans = self.expired_df['lifespan_days']

        return {
            'has_data': True,
            'total_expired': len(self.expired_df),
            'total_active': len(self.df[self.df['is_active'] == True]),
            'median_days': float(lifespans.median()),
            'mean_days': float(lifespans.mean()),
            'std_days': float(lifespans.std()),
            'min_days': int(lifespans.min()),
            'max_days': int(lifespans.max()),
            'q25_days': float(lifespans.quantile(0.25)),
            'q75_days': float(lifespans.quantile(0.75)),
            'histogram': self._compute_histogram(lifespans),
        }

    def _compute_histogram(self, series: pd.Series, bins: int = 20) -> Dict[str, List]:
        """Compute histogram data for frontend rendering"""
        counts, bin_edges = np.histogram(series.dropna(), bins=bins)
        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_labels': [
                f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                for i in range(len(bin_edges) - 1)
            ]
        }

    def analyze_lifespan_by_seller(self) -> Dict[str, Any]:
        """Analyze lifespan and turnover metrics by seller"""
        if self.expired_df.empty:
            return {'has_data': False, 'sellers': []}

        seller_stats = []
        all_coffees_by_seller = self.df.groupby('seller')

        for seller, group in all_coffees_by_seller:
            if pd.isna(seller) or not seller:
                continue

            expired = group[
                (group['is_active'] == False) &
                group['lifespan_days'].notna() &
                (group['lifespan_days'] >= 0)
            ]
            active = group[group['is_active'] == True]

            if len(expired) < 2:
                continue

            lifespans = expired['lifespan_days']
            seller_stats.append({
                'seller': seller,
                'total_coffees': len(group),
                'active_count': len(active),
                'expired_count': len(expired),
                'turnover_rate': len(expired) / len(group) if len(group) > 0 else 0,
                'median_lifespan': float(lifespans.median()),
                'mean_lifespan': float(lifespans.mean()),
                'std_lifespan': float(lifespans.std()) if len(lifespans) > 1 else 0,
                'min_lifespan': int(lifespans.min()),
                'max_lifespan': int(lifespans.max()),
            })

        seller_stats.sort(key=lambda x: x['median_lifespan'])

        # Statistical test: are lifespans different across sellers?
        test_result = self._kruskal_wallis_test('seller')

        return {
            'has_data': True,
            'sellers': seller_stats,
            'kruskal_wallis': test_result,
        }

    def analyze_lifespan_by_origin(self) -> Dict[str, Any]:
        """Analyze whether certain origins sell faster/slower"""
        if self.expired_df.empty:
            return {'has_data': False, 'countries': []}

        country_stats = []
        for country, group in self.expired_df.groupby('country'):
            if pd.isna(country) or not country or len(group) < 3:
                continue

            lifespans = group['lifespan_days']
            country_stats.append({
                'country': country,
                'count': len(group),
                'median_lifespan': float(lifespans.median()),
                'mean_lifespan': float(lifespans.mean()),
                'std_lifespan': float(lifespans.std()) if len(lifespans) > 1 else 0,
            })

        country_stats.sort(key=lambda x: x['median_lifespan'])

        test_result = self._kruskal_wallis_test('country')

        return {
            'has_data': True,
            'countries': country_stats,
            'kruskal_wallis': test_result,
        }

    def analyze_lifespan_by_process(self) -> Dict[str, Any]:
        """Analyze whether certain process types move faster"""
        if self.expired_df.empty:
            return {'has_data': False, 'processes': []}

        process_stats = []
        for process, group in self.expired_df.groupby('process_type'):
            if pd.isna(process) or not process or len(group) < 3:
                continue

            lifespans = group['lifespan_days']
            process_stats.append({
                'process': process,
                'count': len(group),
                'median_lifespan': float(lifespans.median()),
                'mean_lifespan': float(lifespans.mean()),
                'std_lifespan': float(lifespans.std()) if len(lifespans) > 1 else 0,
            })

        process_stats.sort(key=lambda x: x['median_lifespan'])

        test_result = self._kruskal_wallis_test('process_type')

        return {
            'has_data': True,
            'processes': process_stats,
            'kruskal_wallis': test_result,
        }

    def analyze_lifespan_by_price(self) -> Dict[str, Any]:
        """Analyze correlation between price and listing duration"""
        price_lifespan_df = self.expired_df[
            self.expired_df['avg_price'].notna() &
            (self.expired_df['avg_price'] > 0)
        ].copy()

        if len(price_lifespan_df) < 5:
            return {'has_data': False}

        # Spearman correlation (non-parametric)
        corr, p_value = stats.spearmanr(
            price_lifespan_df['avg_price'],
            price_lifespan_df['lifespan_days']
        )

        # Price quartile analysis
        price_lifespan_df['price_quartile'] = pd.qcut(
            price_lifespan_df['avg_price'], q=4, labels=['Budget', 'Mid-Low', 'Mid-High', 'Premium'],
            duplicates='drop'
        )

        quartile_stats = []
        for quartile, group in price_lifespan_df.groupby('price_quartile', observed=True):
            lifespans = group['lifespan_days']
            quartile_stats.append({
                'quartile': str(quartile),
                'count': len(group),
                'median_lifespan': float(lifespans.median()),
                'mean_lifespan': float(lifespans.mean()),
                'avg_price': float(group['avg_price'].mean()),
            })

        # Scatter data for plotting
        scatter_data = {
            'prices': price_lifespan_df['avg_price'].tolist(),
            'lifespans': price_lifespan_df['lifespan_days'].tolist(),
        }

        return {
            'has_data': True,
            'correlation': float(corr),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'n_observations': len(price_lifespan_df),
            'quartile_stats': quartile_stats,
            'scatter_data': scatter_data,
        }

    def get_seller_turnover_summary(self) -> List[Dict[str, Any]]:
        """Summary table: coffees per seller, active count, median lifespan, turnover rate"""
        summary = []
        for seller, group in self.df.groupby('seller'):
            if pd.isna(seller) or not seller:
                continue

            expired = group[
                (group['is_active'] == False) &
                group['lifespan_days'].notna() &
                (group['lifespan_days'] >= 0)
            ]
            active = group[group['is_active'] == True]

            median_lifespan = float(expired['lifespan_days'].median()) if len(expired) > 0 else None

            summary.append({
                'seller': seller,
                'total_coffees': len(group),
                'active': len(active),
                'expired': len(expired),
                'turnover_rate': round(len(expired) / len(group), 2) if len(group) > 0 else 0,
                'median_lifespan_days': median_lifespan,
                'unique_countries': len(group['country'].dropna().unique()),
            })

        summary.sort(key=lambda x: x['total_coffees'], reverse=True)
        return summary

    def compute_seasonal_patterns(self) -> Dict[str, Any]:
        """Analyze monthly patterns of coffee appearance and disappearance"""
        if self.dated_df.empty:
            return {'has_data': False}

        # Monthly appearances
        appearances = self.dated_df['first_observed'].dt.to_period('M').value_counts().sort_index()
        appearance_data = [
            {'month': str(period), 'count': int(count)}
            for period, count in appearances.items()
        ]

        # Monthly disappearances (expired coffees)
        expired_with_dates = self.expired_df[self.expired_df['last_observed'].notna()]
        if not expired_with_dates.empty:
            disappearances = expired_with_dates['last_observed'].dt.to_period('M').value_counts().sort_index()
            disappearance_data = [
                {'month': str(period), 'count': int(count)}
                for period, count in disappearances.items()
            ]
        else:
            disappearance_data = []

        return {
            'has_data': True,
            'appearances': appearance_data,
            'disappearances': disappearance_data,
        }

    def _kruskal_wallis_test(self, group_col: str) -> Dict[str, Any]:
        """Run Kruskal-Wallis test on lifespan across groups"""
        groups = []
        group_names = []

        for name, group in self.expired_df.groupby(group_col):
            if pd.isna(name) or not name:
                continue
            lifespans = group['lifespan_days'].dropna()
            if len(lifespans) >= 3:
                groups.append(lifespans.values)
                group_names.append(str(name))

        if len(groups) < 2:
            return {'has_data': False}

        try:
            h_stat, p_value = stats.kruskal(*groups)
            # Effect size: epsilon-squared
            n = sum(len(g) for g in groups)
            k = len(groups)
            epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else 0

            return {
                'has_data': True,
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'is_significant': p_value < 0.05,
                'n_groups': len(groups),
                'effect_size': float(epsilon_sq),
                'group_names': group_names,
            }
        except Exception:
            return {'has_data': False}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all turnover analyses and return combined results"""
        return {
            'lifespan_overview': self.compute_coffee_lifespans(),
            'by_seller': self.analyze_lifespan_by_seller(),
            'by_origin': self.analyze_lifespan_by_origin(),
            'by_process': self.analyze_lifespan_by_process(),
            'by_price': self.analyze_lifespan_by_price(),
            'seller_summary': self.get_seller_turnover_summary(),
            'seasonal_patterns': self.compute_seasonal_patterns(),
        }
