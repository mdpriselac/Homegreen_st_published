"""
Multi-Way Interaction Analysis Module

Detects interaction effects where combinations of features (origin, process,
varietal) produce unexpected flavor profiles or price premiums beyond what
either feature alone would predict.

Two outcome types:
1. Flavor interactions: emergent/suppressed flavors in feature combos
2. Price interactions: premium/discount synergies in feature combos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from collections import Counter

from analytics.processing.varietals import expand_varietals
from analytics.processing.stat_utils import apply_bh, permutation_premium_test


class InteractionAnalyzer:
    """Analyze multi-way interactions between coffee features"""

    MIN_GROUP_SIZE = 10
    N_PERMUTATIONS = 2000
    PERMUTATION_SEED = 20240405

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.df = cross_feature_df.copy()
        self._expanded_df = None

    # =========================================================================
    # Flavor Interactions
    # =========================================================================

    def analyze_two_way_flavor_interactions(
        self,
        feature_a: str,
        feature_b: str,
        taxonomy_level: str = 'family',
        top_n: int = 30,
    ) -> Dict[str, Any]:
        """
        Compute (Feature_A × Feature_B) → Flavor interaction effects.
        Binomial p-values (raw p_value) are BH-adjusted over every combination x
        flavor tested in this call; is_significant = q < 0.05.

        Detects flavors that appear more/less often in a specific combination
        than you'd predict from either feature alone (under independence).
        """
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        working = self._get_working_df([feature_a, feature_b], list_col)
        if working.empty or len(working) < 20:
            return {'has_data': False}

        # Compute global and marginal flavor rates
        global_rates = self._compute_global_rates(working, list_col)
        marginal_a = self._compute_marginal_rates(working, feature_a, list_col)
        marginal_b = self._compute_marginal_rates(working, feature_b, list_col)

        if not global_rates:
            return {'has_data': False}

        # Compute per-combination flavor profiles and interaction effects
        combination_profiles = {}
        all_effects = []
        tested = []     # every binomial test run (BH family = this interaction type)

        for (val_a, val_b), group in working.groupby([feature_a, feature_b]):
            if len(group) < self.MIN_GROUP_SIZE:
                continue

            combo_key = f"{val_a}|{val_b}"
            observed_counts, group_n = self._compute_group_flavor_counts(group, list_col)
            observed_rates = {f: c / group_n for f, c in observed_counts.items()} if group_n else {}

            # Store combination profile
            flavors_list = [
                {'flavor': f, 'observed_rate': r, 'count': int(observed_counts[f])}
                for f, r in sorted(observed_rates.items(), key=lambda x: -x[1])
            ]
            combination_profiles[combo_key] = {
                'feature_a_value': str(val_a),
                'feature_b_value': str(val_b),
                'sample_size': len(group),
                'flavors': flavors_list,
            }

            # Compute interaction effects for each flavor
            rates_a = marginal_a.get(val_a, {})
            rates_b = marginal_b.get(val_b, {})

            for flavor in global_rates:
                obs = observed_rates.get(flavor, 0)
                ra = rates_a.get(flavor, 0)
                rb = rates_b.get(flavor, 0)
                rg = global_rates[flavor]

                expected = self._expected_rate_independence(ra, rb, rg)
                if expected is None:
                    continue

                score = obs - expected
                ratio = obs / expected if expected > 0 else None

                # Significance test
                obs_count = int(observed_counts.get(flavor, 0))   # exact integer count
                _, p_val = self._test_interaction_significance(
                    obs_count, group_n, expected
                )

                effect = {
                    'feature_a_value': str(val_a),
                    'feature_b_value': str(val_b),
                    'flavor': flavor,
                    'observed_rate': round(obs, 4),
                    'expected_rate': round(expected, 4),
                    'rate_a': round(ra, 4),
                    'rate_b': round(rb, 4),
                    'interaction_score': round(score, 4),
                    'interaction_ratio': round(ratio, 3) if ratio is not None else None,
                    'effect_type': 'emergent' if score > 0 else 'suppressed',
                    'sample_size': len(group),
                    'is_significant': False,
                    'p_value': p_val,
                }
                tested.append(effect)
                if abs(score) >= 0.02:   # skip negligible effects (still in the BH family)
                    all_effects.append(effect)

        self._finalise_tests(tested)

        # Sort and select top effects
        all_effects.sort(key=lambda x: abs(x['interaction_score']), reverse=True)
        emergent = [e for e in all_effects if e['effect_type'] == 'emergent']
        suppressed = [e for e in all_effects if e['effect_type'] == 'suppressed']

        return {
            'has_data': len(combination_profiles) > 0,
            'combination_profiles': combination_profiles,
            'interaction_effects': all_effects[:top_n * 2],
            'top_emergent': emergent[:top_n],
            'top_suppressed': suppressed[:top_n],
            'feature_a_label': self._feature_label(feature_a),
            'feature_b_label': self._feature_label(feature_b),
            'taxonomy_level': taxonomy_level,
            'valid_combinations': len(combination_profiles),
        }

    def analyze_three_way_flavor_interactions(
        self,
        taxonomy_level: str = 'family',
        top_n: int = 30,
    ) -> Dict[str, Any]:
        """
        Compute (Country × Process × Varietal) → Flavor interaction effects.
        Family level only due to sparsity.
        """
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        working = self._get_working_df(
            ['country', 'process_type', 'single_varietal'], list_col
        )
        if working.empty or len(working) < 30:
            return {'has_data': False}

        global_rates = self._compute_global_rates(working, list_col)
        marginal_country = self._compute_marginal_rates(working, 'country', list_col)
        marginal_process = self._compute_marginal_rates(working, 'process_type', list_col)
        marginal_varietal = self._compute_marginal_rates(working, 'single_varietal', list_col)

        if not global_rates:
            return {'has_data': False}

        combination_profiles = {}
        all_effects = []
        tested = []

        for (country, process, varietal), group in working.groupby(
            ['country', 'process_type', 'single_varietal']
        ):
            if len(group) < self.MIN_GROUP_SIZE:
                continue

            combo_key = f"{country}|{process}|{varietal}"
            observed_counts, group_n = self._compute_group_flavor_counts(group, list_col)
            observed_rates = {f: c / group_n for f, c in observed_counts.items()} if group_n else {}

            flavors_list = [
                {'flavor': f, 'observed_rate': r, 'count': int(observed_counts[f])}
                for f, r in sorted(observed_rates.items(), key=lambda x: -x[1])
            ]
            combination_profiles[combo_key] = {
                'country': str(country),
                'process': str(process),
                'varietal': str(varietal),
                'sample_size': len(group),
                'flavors': flavors_list,
            }

            rc = marginal_country.get(country, {})
            rp = marginal_process.get(process, {})
            rv = marginal_varietal.get(varietal, {})

            for flavor in global_rates:
                obs = observed_rates.get(flavor, 0)
                r_c = rc.get(flavor, 0)
                r_p = rp.get(flavor, 0)
                r_v = rv.get(flavor, 0)
                rg = global_rates[flavor]

                expected = self._expected_rate_3way(r_c, r_p, r_v, rg)
                if expected is None:
                    continue

                score = obs - expected

                obs_count = int(observed_counts.get(flavor, 0))   # exact integer count
                _, p_val = self._test_interaction_significance(
                    obs_count, group_n, expected
                )

                effect = {
                    'country': str(country),
                    'process': str(process),
                    'varietal': str(varietal),
                    'flavor': flavor,
                    'observed_rate': round(obs, 4),
                    'expected_rate': round(expected, 4),
                    'interaction_score': round(score, 4),
                    'effect_type': 'emergent' if score > 0 else 'suppressed',
                    'sample_size': len(group),
                    'is_significant': False,
                    'p_value': p_val,
                }
                tested.append(effect)
                if abs(score) >= 0.03:
                    all_effects.append(effect)

        self._finalise_tests(tested)

        all_effects.sort(key=lambda x: abs(x['interaction_score']), reverse=True)
        emergent = [e for e in all_effects if e['effect_type'] == 'emergent']
        suppressed = [e for e in all_effects if e['effect_type'] == 'suppressed']

        return {
            'has_data': len(combination_profiles) > 0,
            'combination_profiles': combination_profiles,
            'interaction_effects': all_effects[:top_n * 2],
            'top_emergent': emergent[:top_n],
            'top_suppressed': suppressed[:top_n],
            'taxonomy_level': taxonomy_level,
            'valid_combinations': len(combination_profiles),
        }

    # =========================================================================
    # Price Interactions
    # =========================================================================

    def analyze_two_way_price_interactions(
        self,
        feature_a: str,
        feature_b: str,
        top_n: int = 30,
    ) -> Dict[str, Any]:
        """
        Compute (Feature_A × Feature_B) → Price interaction effects.

        Detects combos where the median price differs from what you'd expect
        given each feature's individual price effect (additive model).

        For flavor features (feature name starts with 'flavor_'), treats each
        flavor as a binary variable (has/doesn't have).

        Significance (p_value raw, q_value = Benjamini-Hochberg over all combos of
        this call, is_significant = q < 0.05) tests the interaction premium itself:
            premium = median(price | A&B)
                      - (median(price | A) + median(price | B) - median(price))
        i.e. how far the combination's median is from the additive expectation.
        Permutation test (N_PERMUTATIONS = 2000, seeded): B membership is permuted
        among rows within A and within not-A, which preserves |A|, |B|, |A&B| and
        the price distribution inside A; p = (1 + #{|premium*| >= |premium|}) / 2001.
        (The old combo-vs-rest Mann-Whitney did not test this quantity.)
        """
        is_flavor_a = feature_a.startswith('flavor_')
        is_flavor_b = feature_b.startswith('flavor_')

        # Prepare working data
        working = self.df[
            self.df['price_per_lb'].notna() & (self.df['price_per_lb'] > 0)
        ].copy()

        # Handle varietal expansion
        if feature_a == 'single_varietal' or feature_b == 'single_varietal':
            working = self._expand_from_df(working)
            if working.empty:
                return {'has_data': False}

        # For non-flavor categorical features, filter nulls
        for feat in [feature_a, feature_b]:
            if not feat.startswith('flavor_'):
                working = working[working[feat].notna()]

        if len(working) < 20:
            return {'has_data': False}

        global_median = float(working['price_per_lb'].median())

        # Compute marginal prices
        marginal_a = self._compute_marginal_prices(working, feature_a, is_flavor_a)
        marginal_b = self._compute_marginal_prices(working, feature_b, is_flavor_b)

        # Handle flavor × flavor or flavor × categorical combos
        if is_flavor_a or is_flavor_b:
            return self._flavor_price_interactions(
                working, feature_a, feature_b, is_flavor_a, is_flavor_b,
                marginal_a, marginal_b, global_median, top_n
            )

        # Standard categorical × categorical price interactions
        combinations = []
        prices_all = working['price_per_lb'].to_numpy(dtype=float)
        for (val_a, val_b), group in working.groupby([feature_a, feature_b]):
            prices = group['price_per_lb']
            if len(prices) < self.MIN_GROUP_SIZE:
                continue

            obs_median = float(prices.median())
            price_a = marginal_a.get(val_a, global_median)
            price_b = marginal_b.get(val_b, global_median)
            expected = price_a + price_b - global_median
            premium = obs_median - expected

            # Test of the interaction premium itself (not combo-vs-rest)
            _, p_val = permutation_premium_test(
                prices_all,
                (working[feature_a] == val_a).to_numpy(),
                (working[feature_b] == val_b).to_numpy(),
                n_perm=self.N_PERMUTATIONS, seed=self.PERMUTATION_SEED,
            )

            combinations.append({
                'feature_a_value': str(val_a),
                'feature_b_value': str(val_b),
                'sample_size': len(group),
                'median_price': round(obs_median, 2),
                'mean_price': round(float(prices.mean()), 2),
                'expected_price': round(expected, 2),
                'price_premium': round(premium, 2),
                'marginal_a_price': round(price_a, 2),
                'marginal_b_price': round(price_b, 2),
                'effect_type': 'premium' if premium > 0 else 'discount',
                'is_significant': False,
                'p_value': p_val,
            })

        self._finalise_tests(combinations)

        combinations.sort(key=lambda x: abs(x['price_premium']), reverse=True)
        premiums = [c for c in combinations if c['effect_type'] == 'premium']
        discounts = [c for c in combinations if c['effect_type'] == 'discount']

        return {
            'has_data': len(combinations) > 0,
            'combinations': combinations,
            'top_premiums': premiums[:top_n],
            'top_discounts': discounts[:top_n],
            'feature_a_label': self._feature_label(feature_a),
            'feature_b_label': self._feature_label(feature_b),
            'global_median_price': round(global_median, 2),
            'valid_combinations': len(combinations),
        }

    def _flavor_price_interactions(
        self,
        working: pd.DataFrame,
        feature_a: str,
        feature_b: str,
        is_flavor_a: bool,
        is_flavor_b: bool,
        marginal_a: Dict,
        marginal_b: Dict,
        global_median: float,
        top_n: int,
    ) -> Dict[str, Any]:
        """Handle price interactions where one or both features are flavor lists."""
        list_col_a = feature_a if is_flavor_a else None
        list_col_b = feature_b if is_flavor_b else None

        # Get all flavor values for flavor features
        flavors_a = set()
        if is_flavor_a:
            for flist in working[feature_a]:
                if isinstance(flist, list):
                    flavors_a.update(f for f in flist if f)
        else:
            flavors_a = set(working[feature_a].dropna().unique())

        flavors_b = set()
        if is_flavor_b:
            for flist in working[feature_b]:
                if isinstance(flist, list):
                    flavors_b.update(f for f in flist if f)
        else:
            flavors_b = set(working[feature_b].dropna().unique())

        combinations = []
        prices_all = working['price_per_lb'].to_numpy(dtype=float)
        for val_a in flavors_a:
            # Filter for feature_a
            if is_flavor_a:
                mask_a = working[feature_a].apply(
                    lambda x: val_a in x if isinstance(x, list) else False
                )
            else:
                mask_a = working[feature_a] == val_a

            for val_b in flavors_b:
                # Filter for feature_b
                if is_flavor_b:
                    mask_b = working[feature_b].apply(
                        lambda x: val_b in x if isinstance(x, list) else False
                    )
                else:
                    mask_b = working[feature_b] == val_b

                group = working[mask_a & mask_b]
                prices = group['price_per_lb']
                if len(prices) < self.MIN_GROUP_SIZE:
                    continue

                obs_median = float(prices.median())
                price_a = marginal_a.get(val_a, global_median)
                price_b = marginal_b.get(val_b, global_median)
                expected = price_a + price_b - global_median
                premium = obs_median - expected

                _, p_val = permutation_premium_test(
                    prices_all, mask_a.to_numpy(), mask_b.to_numpy(),
                    n_perm=self.N_PERMUTATIONS, seed=self.PERMUTATION_SEED,
                )

                combinations.append({
                    'feature_a_value': str(val_a),
                    'feature_b_value': str(val_b),
                    'sample_size': len(group),
                    'median_price': round(obs_median, 2),
                    'mean_price': round(float(prices.mean()), 2),
                    'expected_price': round(expected, 2),
                    'price_premium': round(premium, 2),
                    'marginal_a_price': round(price_a, 2),
                    'marginal_b_price': round(price_b, 2),
                    'effect_type': 'premium' if premium > 0 else 'discount',
                    'is_significant': False,
                    'p_value': p_val,
                })

        self._finalise_tests(combinations)

        combinations.sort(key=lambda x: abs(x['price_premium']), reverse=True)
        premiums = [c for c in combinations if c['effect_type'] == 'premium']
        discounts = [c for c in combinations if c['effect_type'] == 'discount']

        return {
            'has_data': len(combinations) > 0,
            'combinations': combinations,
            'top_premiums': premiums[:top_n],
            'top_discounts': discounts[:top_n],
            'feature_a_label': self._feature_label(feature_a),
            'feature_b_label': self._feature_label(feature_b),
            'global_median_price': round(global_median, 2),
            'valid_combinations': len(combinations),
        }

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _get_working_df(self, features: List[str], list_col: str) -> pd.DataFrame:
        """Get DataFrame filtered to rows with all required features + flavors."""
        needs_varietal = 'single_varietal' in features
        if needs_varietal:
            df = self._expand_varietals()
            if df.empty:
                return pd.DataFrame()
        else:
            df = self.df.copy()

        # Filter to rows with flavors
        df = df[df['has_flavors']].copy()

        # Filter nulls for each required feature
        for feat in features:
            if feat in df.columns:
                df = df[df[feat].notna()]

        return df.reset_index(drop=True)

    def _expand_varietals(self) -> pd.DataFrame:
        """Single-varietal coffees only (cached).

        Every interaction result here carries a significance test (binomial /
        Mann-Whitney), so rows must be independent coffees: multi-varietal
        coffees are excluded rather than duplicated per varietal.
        """
        if self._expanded_df is None:
            self._expanded_df = expand_varietals(self.df, 'single')
        return self._expanded_df

    def _expand_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Single-varietal expansion of an already-filtered DataFrame."""
        return expand_varietals(df, 'single')

    def _compute_global_rates(
        self, df: pd.DataFrame, list_col: str
    ) -> Dict[str, float]:
        """Compute flavor rate across all coffees."""
        counts = Counter()
        total = 0
        for _, row in df.iterrows():
            flavors = row.get(list_col, [])
            if isinstance(flavors, list):
                total += 1
                for f in set(flavors):
                    if f:
                        counts[f] += 1
        if total == 0:
            return {}
        return {f: c / total for f, c in counts.items()}

    def _compute_marginal_rates(
        self, df: pd.DataFrame, feature_col: str, list_col: str
    ) -> Dict[str, Dict[str, float]]:
        """Compute flavor rates per value of feature_col."""
        result = {}
        for val, group in df.groupby(feature_col):
            counts = Counter()
            total = 0
            for _, row in group.iterrows():
                flavors = row.get(list_col, [])
                if isinstance(flavors, list):
                    total += 1
                    for f in set(flavors):
                        if f:
                            counts[f] += 1
            if total > 0:
                result[val] = {f: c / total for f, c in counts.items()}
        return result

    def _compute_group_flavor_counts(
        self, group: pd.DataFrame, list_col: str
    ) -> Tuple[Dict[str, int], int]:
        """Integer flavor counts for a group and the number of coffees counted."""
        counts = Counter()
        total = 0
        for _, row in group.iterrows():
            flavors = row.get(list_col, [])
            if isinstance(flavors, list):
                total += 1
                for f in set(flavors):
                    if f:
                        counts[f] += 1
        return dict(counts), total

    def _compute_group_flavor_rates(
        self, group: pd.DataFrame, list_col: str
    ) -> Dict[str, float]:
        """Compute flavor rates for a single group."""
        counts = Counter()
        total = 0
        for _, row in group.iterrows():
            flavors = row.get(list_col, [])
            if isinstance(flavors, list):
                total += 1
                for f in set(flavors):
                    if f:
                        counts[f] += 1
        if total == 0:
            return {}
        return {f: c / total for f, c in counts.items()}

    def _compute_marginal_prices(
        self, df: pd.DataFrame, feature: str, is_flavor: bool
    ) -> Dict[str, float]:
        """Compute median price per value of a feature."""
        if is_flavor:
            # For flavor list columns, compute median for coffees with each flavor
            all_flavors = set()
            for flist in df[feature]:
                if isinstance(flist, list):
                    all_flavors.update(f for f in flist if f)

            result = {}
            for flavor in all_flavors:
                mask = df[feature].apply(
                    lambda x: flavor in x if isinstance(x, list) else False
                )
                prices = df.loc[mask, 'price_per_lb']
                if len(prices) >= self.MIN_GROUP_SIZE:
                    result[flavor] = float(prices.median())
            return result
        else:
            result = {}
            for val, group in df.groupby(feature):
                prices = group['price_per_lb']
                if len(prices) >= self.MIN_GROUP_SIZE:
                    result[val] = float(prices.median())
            return result

    def _expected_rate_independence(
        self, rate_a: float, rate_b: float, global_rate: float
    ) -> Optional[float]:
        """Expected rate under independence: P(f|A)*P(f|B)/P(f), capped at 1."""
        if global_rate <= 0:
            return None
        expected = rate_a * rate_b / global_rate
        return min(expected, 1.0)

    def _expected_rate_3way(
        self, rate_a: float, rate_b: float, rate_c: float, global_rate: float
    ) -> Optional[float]:
        """Expected rate for 3-way independence: P(f|A)*P(f|B)*P(f|C)/P(f)^2."""
        if global_rate <= 0:
            return None
        expected = rate_a * rate_b * rate_c / (global_rate ** 2)
        return min(expected, 1.0)

    def _finalise_tests(self, rows: List[Dict[str, Any]]) -> None:
        """BH-adjust one family of tests (one interaction type) in place:
        adds q_value, sets is_significant = q < 0.05, then rounds the raw p for display."""
        apply_bh(rows)
        for r in rows:
            if r.get('p_value') is not None:
                r['p_value'] = round(float(r['p_value']), 6)
            if r.get('q_value') is not None:
                r['q_value'] = round(float(r['q_value']), 6)

    def _test_interaction_significance(
        self, observed_count: int, total: int, expected_rate: float
    ) -> Tuple[bool, Optional[float]]:
        """Binomial test for whether observed count differs from expected."""
        if total <= 0 or expected_rate <= 0 or expected_rate >= 1:
            return False, None
        try:
            result = stats.binomtest(observed_count, total, expected_rate)
            p_val = float(result.pvalue)
            return p_val < 0.05, p_val
        except Exception:
            return False, None

    def _feature_label(self, col_name: str) -> str:
        """Human-readable label for a feature column."""
        labels = {
            'country': 'Country',
            'region': 'Region',
            'process_type': 'Process Method',
            'single_varietal': 'Varietal',
            'flavor_families': 'Flavor Family',
            'flavor_genera': 'Flavor Genus',
            'flavor_species': 'Flavor Species',
        }
        return labels.get(col_name, col_name)

    # =========================================================================
    # Full analysis for cache
    # =========================================================================

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all interaction analyses for cache generation."""
        result = {}

        # --- Flavor interactions (2-way) ---
        flavor_2way = [
            ('origin_process', 'country', 'process_type'),
            ('process_varietal', 'process_type', 'single_varietal'),
            ('origin_varietal', 'country', 'single_varietal'),
        ]
        for key, fa, fb in flavor_2way:
            for level in ['family', 'genus']:
                cache_key = f'{key}_flavor_{level}'
                result[cache_key] = self.analyze_two_way_flavor_interactions(
                    fa, fb, level
                )

        # --- Flavor interactions (3-way) ---
        result['three_way_flavor_family'] = (
            self.analyze_three_way_flavor_interactions('family')
        )

        # --- Price interactions (2-way) ---
        price_2way = [
            ('origin_flavor_price', 'country', 'flavor_families'),
            ('origin_process_price', 'country', 'process_type'),
            ('origin_varietal_price', 'country', 'single_varietal'),
            ('flavor_process_price', 'flavor_families', 'process_type'),
            ('flavor_varietal_price', 'flavor_families', 'single_varietal'),
        ]
        for key, fa, fb in price_2way:
            result[key] = self.analyze_two_way_price_interactions(fa, fb)

        return result
