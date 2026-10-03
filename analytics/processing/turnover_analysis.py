"""
Seller Turnover and Coffee Lifespan Analysis (survival analysis)

How long coffees stay listed, turnover by seller, and lifespan patterns across
origins, process types and price ranges.

Data caveats handled here
  * Expiry tracking was unreliable before TRACKING_START (2025-06-18): inactive
    coffees with last_observed <= 2025-06-17 expired at an unknown time. They
    are EXCLUDED from lifespan analysis.
  * Coffees first observed on/before INITIAL_INVENTORY_CUTOFF (2024-04-05) are
    the starting inventory: their true listing date is unknown
    (left-censored). They are EXCLUDED from lifespan analysis and from monthly
    "new listings".
  * last_observed = last scrape date the coffee was seen listed (active or
    expired). Active coffees are right-censored at last_observed.
  * Lifespans are measured on the scrape grid: each sighting stands for about
    one scrape interval I (median gap between scrape dates in the tracked
    window, 7 days). Expired coffees (events): birth lies in (previous scrape,
    first_observed] and death in (last_observed, next scrape], so
    duration = (last - first) + I (I/2 at each end). Active coffees
    (right-censored): duration = (last - first) + I/2 (birth uncertainty only).
    A coffee seen in one scrape therefore has duration I, not 0.
  * Lifespan statistics are Kaplan-Meier (median, quartiles) and log-rank tests
    (group vs all other eligible coffees, Benjamini-Hochberg across the groups
    of one dimension).
  * Sellers with no listing seen in the last SELLER_INACTIVE_DAYS (60) days of
    the window are treated as out of business and left out of turnover rates;
    they are reported in by_seller['excluded_inactive_sellers'].
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats

from analytics.processing.stat_utils import apply_bh, kaplan_meier, logrank_p

INITIAL_INVENTORY_CUTOFF = pd.Timestamp('2024-04-05')
TRACKING_START = pd.Timestamp('2025-06-18')
SELLER_INACTIVE_DAYS = 60
DURATION_DEFINITION = (
    "Each sighting stands for one scrape interval I (median gap between scrapes). "
    "Expired coffee: (last_observed - first_observed) + I. "
    "Active coffee (right-censored): (last_observed - first_observed) + I/2.")
MIN_GROUP_N = 3   # smallest group (eligible coffees) reported


class SellerTurnoverAnalyzer:
    """Analyze coffee listing lifespans and seller turnover patterns"""

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.df = cross_feature_df.copy()
        self._prepare_data()

    # ------------------------------------------------------------------
    # Preparation / censoring
    # ------------------------------------------------------------------

    def _prepare_data(self):
        """Classify every coffee and build the survival sample."""
        df = self.df
        df['first_observed'] = pd.to_datetime(df['first_observed'], errors='coerce')
        df['last_observed'] = pd.to_datetime(df['last_observed'], errors='coerce')
        df['_active'] = df['is_active'].map(
            lambda v: bool(v) if isinstance(v, (bool, np.bool_)) else np.nan)

        has_dates = df['first_observed'].notna() & df['last_observed'].notna()
        df['duration_days'] = np.where(
            has_dates, (df['last_observed'] - df['first_observed']).dt.days, np.nan)

        # median gap between distinct scrape dates in the reliably tracked period
        scrape_dates = pd.Series(sorted(set(
            df.loc[df['first_observed'] >= TRACKING_START, 'first_observed'].dt.normalize()
        ) | set(
            df.loc[df['last_observed'] >= TRACKING_START, 'last_observed'].dt.normalize()
        )))
        gaps = scrape_dates.diff().dt.days.dropna()
        self.scrape_interval_days = float(gaps.median()) if len(gaps) else 7.0

        valid = has_dates & df['_active'].notna() & (df['duration_days'] >= 0)
        left = valid & (df['first_observed'] <= INITIAL_INVENTORY_CUTOFF)
        pre_tracking = (valid & ~left & (df['_active'] == False)
                        & (df['last_observed'] < TRACKING_START))
        eligible = valid & ~left & ~pre_tracking

        df['lifespan_days'] = df['duration_days']          # back-compat column
        self.survival_df = df[eligible].copy()
        # event observed = coffee expired; active coffees are right-censored
        self.survival_df['event'] = (self.survival_df['_active'] == False).astype(bool)
        # scrape-grid duration: raw span + I (expired) or + I/2 (active)
        raw = self.survival_df['duration_days'].astype(float)
        self.survival_df['duration'] = raw + np.where(
            self.survival_df['event'], self.scrape_interval_days,
            self.scrape_interval_days / 2.0)

        self.expired_df = self.survival_df[self.survival_df['event']].copy()
        self.dated_df = df[df['first_observed'].notna()].copy()

        self.window_end = df['last_observed'].max()
        self.first_scrape = df['first_observed'].min()
        self.exclusions = {
            'total_coffees': int(len(df)),
            'invalid_or_missing_dates': int((~valid).sum()),
            'left_censored_initial_inventory': int(left.sum()),
            'expired_before_tracking': int(pre_tracking.sum()),
            'eligible': int(eligible.sum()),
            'eligible_expired': int(self.survival_df['event'].sum()),
            'eligible_active_censored': int((~self.survival_df['event']).sum()),
        }

    def observation_window(self) -> Dict[str, Any]:
        def fmt(t):
            return t.strftime('%Y-%m-%d') if pd.notna(t) else None
        return {
            'window_start': fmt(INITIAL_INVENTORY_CUTOFF),
            'window_end': fmt(self.window_end),
            'first_scrape': fmt(self.first_scrape),
            'initial_inventory_cutoff': fmt(INITIAL_INVENTORY_CUTOFF),
            'tracking_start': fmt(TRACKING_START),
            'seller_inactive_days': SELLER_INACTIVE_DAYS,
            'scrape_interval_days': self.scrape_interval_days,
            'duration_definition': DURATION_DEFINITION,
            'exclusions': dict(self.exclusions),
        }

    def _inactive_sellers(self) -> List[str]:
        """Sellers with no coffee seen listed in the last 60 days of the window."""
        if pd.isna(self.window_end) or 'seller' not in self.df.columns:
            return []
        cutoff = self.window_end - pd.Timedelta(days=SELLER_INACTIVE_DAYS)
        last_seen = self.df.dropna(subset=['seller']).groupby('seller')['last_observed'].max()
        return sorted(str(s) for s, t in last_seen.items() if pd.isna(t) or t < cutoff)

    # ------------------------------------------------------------------
    # Survival helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _km(frame: pd.DataFrame) -> Dict[str, Any]:
        return kaplan_meier(frame['duration'].to_numpy(), frame['event'].to_numpy())

    def _group_table(self, frame: pd.DataFrame, group_col: str, name_key: str,
                     min_n: int = MIN_GROUP_N) -> List[Dict[str, Any]]:
        """Per-group KM median + log-rank (group vs rest), BH across groups."""
        rows = []
        for name, group in frame.groupby(group_col):
            if pd.isna(name) or not name or len(group) < min_n:
                continue
            km = self._km(group)
            rest = frame[frame[group_col] != name]
            p = logrank_p(group['duration'].to_numpy(), group['event'].to_numpy(),
                          rest['duration'].to_numpy(), rest['event'].to_numpy())
            rows.append({
                name_key: name,
                'count': int(len(group)),
                'events': km['n_events'],
                'censored': int(len(group) - km['n_events']),
                'median_lifespan': km['median'],       # Kaplan-Meier; None if S(t) never <= 0.5
                'q25_lifespan': km['q25'],
                'q75_lifespan': km['q75'],
                'p_value': p,                           # log-rank, group vs all other eligible
            })
        apply_bh(rows)
        rows.sort(key=lambda r: (r['median_lifespan'] is None, r['median_lifespan'] or 0))
        return rows

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    def compute_coffee_lifespans(self) -> Dict[str, Any]:
        """Overall Kaplan-Meier lifespan statistics (censoring-aware)"""
        sdf = self.survival_df
        if sdf.empty:
            return {'has_data': False, 'observation_window': self.observation_window()}

        km = self._km(sdf)
        observed = self.expired_df['duration']
        return {
            'has_data': True,
            'total_expired': int(sdf['event'].sum()),
            'total_active': int((~sdf['event']).sum()),
            'median_days': km['median'],               # Kaplan-Meier median
            'q25_days': km['q25'],
            'q75_days': km['q75'],
            # Descriptive only, from expired coffees (biased low: censored coffees
            # are still listed). Do not present as the typical lifespan.
            'mean_days': float(observed.mean()) if len(observed) else None,
            'std_days': float(observed.std()) if len(observed) > 1 else None,
            'min_days': float(sdf['duration'].min()),
            'max_days': float(sdf['duration'].max()),
            'mean_is_biased_note': 'mean/std use expired coffees only',
            'km_curve': {'times': km['times'], 'survival': km['survival']},
            'histogram': self._compute_histogram(observed) if len(observed) else
                         {'counts': [], 'bin_edges': [], 'bin_labels': []},
            'observation_window': self.observation_window(),
        }

    def _compute_histogram(self, series: pd.Series, bins: int = 20) -> Dict[str, List]:
        """Histogram of observed (expired) lifespans for frontend rendering"""
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
        """Kaplan-Meier lifespan per seller (inactive sellers excluded)"""
        inactive = self._inactive_sellers()
        sdf = self.survival_df[~self.survival_df['seller'].isin(inactive)]
        if sdf.empty:
            return {'has_data': False, 'sellers': [],
                    'excluded_inactive_sellers': inactive,
                    'observation_window': self.observation_window()}

        rows = self._group_table(sdf, 'seller', 'seller', min_n=2)
        for r in rows:
            r['total_coffees'] = r['count']
            r['active_count'] = r['censored']
            r['expired_count'] = r['events']
            r['turnover_rate'] = r['events'] / r['count'] if r['count'] else 0
        return {
            'has_data': len(rows) > 0,
            'sellers': rows,
            'excluded_inactive_sellers': inactive,
            'test': 'log-rank (seller vs all other eligible coffees), BH across sellers',
            'observation_window': self.observation_window(),
        }

    def analyze_lifespan_by_origin(self) -> Dict[str, Any]:
        """Do certain origins sell faster/slower? (Kaplan-Meier + log-rank)"""
        if self.survival_df.empty:
            return {'has_data': False, 'countries': []}
        rows = self._group_table(self.survival_df, 'country', 'country')
        return {
            'has_data': len(rows) > 0,
            'countries': rows,
            'test': 'log-rank (country vs all other eligible coffees), BH across countries',
        }

    def analyze_lifespan_by_process(self) -> Dict[str, Any]:
        """Do certain process types move faster? (Kaplan-Meier + log-rank)"""
        if self.survival_df.empty:
            return {'has_data': False, 'processes': []}
        rows = self._group_table(self.survival_df, 'process_type', 'process')
        return {
            'has_data': len(rows) > 0,
            'processes': rows,
            'test': 'log-rank (process vs all other eligible coffees), BH across processes',
        }

    def analyze_lifespan_by_price(self) -> Dict[str, Any]:
        """Price vs listing duration: quartile KM medians + budget-vs-premium log-rank"""
        pdf = self.survival_df[
            self.survival_df['price_per_lb'].notna() & (self.survival_df['price_per_lb'] > 0)
        ].copy()
        if len(pdf) < 20:
            return {'has_data': False}

        pdf['price_quartile'] = pd.qcut(
            pdf['price_per_lb'], q=4, labels=['Budget', 'Mid-Low', 'Mid-High', 'Premium'],
            duplicates='drop')

        quartile_stats = []
        for quartile, group in pdf.groupby('price_quartile', observed=True):
            km = self._km(group)
            quartile_stats.append({
                'quartile': str(quartile),
                'count': int(len(group)),
                'events': km['n_events'],
                'median_lifespan': km['median'],
                'mean_price_per_lb': float(group['price_per_lb'].mean()),
            })

        cats = pdf['price_quartile'].cat.categories
        low = pdf[pdf['price_quartile'] == cats[0]]
        high = pdf[pdf['price_quartile'] == cats[-1]]
        p_value = logrank_p(low['duration'].to_numpy(), low['event'].to_numpy(),
                            high['duration'].to_numpy(), high['event'].to_numpy())

        # Spearman among expired coffees only (censored coffees would bias it);
        # reported for reference, not as the headline test.
        exp = pdf[pdf['event']]
        corr = p_spear = None
        if len(exp) >= 5:
            corr, p_spear = stats.spearmanr(exp['price_per_lb'], exp['duration'])

        return {
            'has_data': True,
            'correlation': float(corr) if corr is not None else None,
            'correlation_p_value': float(p_spear) if p_spear is not None else None,
            'correlation_note': 'Spearman on expired coffees only (censoring-biased)',
            'p_value': p_value,                         # log-rank, lowest vs highest quartile
            'is_significant': bool(p_value is not None and p_value < 0.05),
            'test': 'log-rank, Budget vs Premium quartile',
            'n_observations': int(len(pdf)),
            'quartile_stats': quartile_stats,
            'scatter_data': {
                'prices': exp['price_per_lb'].tolist(),
                'lifespans': exp['duration'].tolist(),
            },
        }

    def get_seller_turnover_summary(self) -> List[Dict[str, Any]]:
        """Per-seller table over eligible coffees; inactive sellers omitted."""
        inactive = set(self._inactive_sellers())
        summary = []
        for seller, group in self.survival_df.groupby('seller'):
            if pd.isna(seller) or not seller or seller in inactive:
                continue
            km = self._km(group)
            n = len(group)
            summary.append({
                'seller': seller,
                'total_coffees': int(n),
                'active': int(n - km['n_events']),
                'expired': km['n_events'],
                'turnover_rate': round(km['n_events'] / n, 2) if n else 0,
                'median_lifespan_days': km['median'],
                'unique_countries': int(group['country'].dropna().nunique()),
            })
        summary.sort(key=lambda x: x['total_coffees'], reverse=True)
        return summary

    def compute_seasonal_patterns(self) -> Dict[str, Any]:
        """Monthly new listings (excluding the starting inventory) and removals."""
        if self.dated_df.empty:
            return {'has_data': False}

        first_month = self.first_scrape.to_period('M')
        new = self.dated_df[
            (self.dated_df['first_observed'] > INITIAL_INVENTORY_CUTOFF)
            & (self.dated_df['first_observed'].dt.to_period('M') != first_month)
        ]
        appearances = new['first_observed'].dt.to_period('M').value_counts().sort_index()
        appearance_data = [{'month': str(m), 'count': int(c)} for m, c in appearances.items()]

        # Removals: expired coffees whose expiry is reliably tracked
        removed = self.df[
            (self.df['_active'] == False) & self.df['last_observed'].notna()
            & (self.df['last_observed'] >= TRACKING_START)
        ]
        disappearance_data = [
            {'month': str(m), 'count': int(c)}
            for m, c in removed['last_observed'].dt.to_period('M').value_counts().sort_index().items()
        ]

        return {
            'has_data': True,
            'appearances': appearance_data,
            'disappearances': disappearance_data,
            'excluded_first_scrape_month': str(first_month),
            'disappearances_from': TRACKING_START.strftime('%Y-%m-%d'),
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all turnover analyses and return combined results"""
        return {
            'observation_window': self.observation_window(),
            'lifespan_overview': self.compute_coffee_lifespans(),
            'by_seller': self.analyze_lifespan_by_seller(),
            'by_origin': self.analyze_lifespan_by_origin(),
            'by_process': self.analyze_lifespan_by_process(),
            'by_price': self.analyze_lifespan_by_price(),
            'seller_summary': self.get_seller_turnover_summary(),
            'seasonal_patterns': self.compute_seasonal_patterns(),
        }
