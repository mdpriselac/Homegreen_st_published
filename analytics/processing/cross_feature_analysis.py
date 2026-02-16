"""
Cross-Feature Analysis Module

Analyzes relationships between all pairs of categorical and continuous features:
process method, varietal, origin, flavor, and price.
Uses Chi-square/Cramer's V for categorical pairs and Kruskal-Wallis for mixed pairs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from collections import Counter, defaultdict
from itertools import combinations


class CrossFeatureAnalyzer:
    """Analyze relationships between all pairs of coffee features"""

    MIN_GROUP_SIZE = 5

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.df = cross_feature_df.copy()

    # -------------------------------------------------------------------------
    # Process Method Analysis
    # -------------------------------------------------------------------------

    def analyze_process_by_origin(self, min_coffees: int = 10) -> Dict[str, Any]:
        """Process method distribution by country and region"""
        result = {
            'by_country': self._categorical_distribution(
                'country', 'process_type', min_coffees, 'Country', 'Process Method'
            ),
            'by_region': self._categorical_distribution(
                'region', 'process_type', min_coffees, 'Region', 'Process Method'
            ),
        }
        return result

    # -------------------------------------------------------------------------
    # Varietal Analysis
    # -------------------------------------------------------------------------

    def analyze_varietal_by_origin(self, min_coffees: int = 10,
                                   top_n_varietals: int = 15) -> Dict[str, Any]:
        """Varietal distribution by country"""
        expanded = self._expand_varietals()
        if expanded.empty:
            return {'has_data': False}

        # Get top varietals overall
        varietal_counts = expanded['single_varietal'].value_counts()
        top_varietals = varietal_counts.head(top_n_varietals).index.tolist()

        # Filter to top varietals and countries with enough data
        filtered = expanded[expanded['single_varietal'].isin(top_varietals)]
        country_counts = filtered['country'].value_counts()
        valid_countries = country_counts[country_counts >= min_coffees].index.tolist()
        filtered = filtered[filtered['country'].isin(valid_countries)]

        if filtered.empty:
            return {'has_data': False}

        # Build heatmap: country x varietal counts
        heatmap = pd.crosstab(filtered['country'], filtered['single_varietal'])

        # Normalize by row (percentage within each country)
        heatmap_pct = heatmap.div(heatmap.sum(axis=1), axis=0)

        # Chi-square test on the contingency table
        chi2_result = self._chi_square_test(heatmap)

        return {
            'has_data': True,
            'heatmap_counts': {
                'rows': heatmap.index.tolist(),
                'columns': heatmap.columns.tolist(),
                'values': heatmap.values.tolist(),
            },
            'heatmap_pct': {
                'rows': heatmap_pct.index.tolist(),
                'columns': heatmap_pct.columns.tolist(),
                'values': heatmap_pct.values.tolist(),
            },
            'top_varietals': top_varietals,
            'chi_square': chi2_result,
        }

    # -------------------------------------------------------------------------
    # Flavor by Process / Varietal
    # -------------------------------------------------------------------------

    def analyze_flavor_by_process(self, taxonomy_level: str = 'family') -> Dict[str, Any]:
        """Flavor profiles grouped by processing method"""
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        df = self.df[self.df['process_type'].notna() & self.df['has_flavors']].copy()
        if df.empty:
            return {'has_data': False}

        processes = df['process_type'].value_counts()
        valid_processes = processes[processes >= self.MIN_GROUP_SIZE].index.tolist()

        profiles = {}
        for process in valid_processes:
            process_df = df[df['process_type'] == process]
            flavor_counts = Counter()
            total = 0
            for _, row in process_df.iterrows():
                flavors = row.get(list_col, [])
                if isinstance(flavors, list):
                    total += 1
                    for f in set(flavors):
                        if f:
                            flavor_counts[f] += 1

            if total > 0:
                profiles[process] = {
                    'total_coffees': total,
                    'flavors': [
                        {
                            'flavor': flavor,
                            'count': count,
                            'rate': count / total,
                        }
                        for flavor, count in flavor_counts.most_common()
                    ],
                }

        # Find distinctive flavors per process (compare rate to global rate)
        global_counts = Counter()
        global_total = 0
        for _, row in df.iterrows():
            flavors = row.get(list_col, [])
            if isinstance(flavors, list):
                global_total += 1
                for f in set(flavors):
                    if f:
                        global_counts[f] += 1

        distinctive = {}
        for process, profile in profiles.items():
            distinctive[process] = []
            for flavor_info in profile['flavors']:
                flavor = flavor_info['flavor']
                local_rate = flavor_info['rate']
                global_rate = global_counts[flavor] / global_total if global_total > 0 else 0

                if global_rate > 0:
                    ratio = local_rate / global_rate
                    diff = local_rate - global_rate
                    distinctive[process].append({
                        'flavor': flavor,
                        'local_rate': local_rate,
                        'global_rate': global_rate,
                        'ratio': ratio,
                        'difference': diff,
                    })

            distinctive[process].sort(key=lambda x: x['ratio'], reverse=True)

        return {
            'has_data': True,
            'profiles': profiles,
            'distinctive': distinctive,
            'taxonomy_level': taxonomy_level,
        }

    def analyze_flavor_by_varietal(self, taxonomy_level: str = 'family',
                                    top_n_varietals: int = 10) -> Dict[str, Any]:
        """Flavor profiles grouped by varietal"""
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        expanded = self._expand_varietals()
        if expanded.empty:
            return {'has_data': False}

        expanded = expanded[expanded['has_flavors']].copy()

        # Top varietals by count
        varietal_counts = expanded['single_varietal'].value_counts()
        top_varietals = varietal_counts[varietal_counts >= self.MIN_GROUP_SIZE].head(top_n_varietals).index.tolist()

        profiles = {}
        for varietal in top_varietals:
            var_df = expanded[expanded['single_varietal'] == varietal]
            flavor_counts = Counter()
            total = 0
            for _, row in var_df.iterrows():
                flavors = row.get(list_col, [])
                if isinstance(flavors, list):
                    total += 1
                    for f in set(flavors):
                        if f:
                            flavor_counts[f] += 1

            if total > 0:
                profiles[varietal] = {
                    'total_coffees': total,
                    'flavors': [
                        {'flavor': flavor, 'count': count, 'rate': count / total}
                        for flavor, count in flavor_counts.most_common()
                    ],
                }

        return {
            'has_data': len(profiles) > 0,
            'profiles': profiles,
            'taxonomy_level': taxonomy_level,
        }

    # -------------------------------------------------------------------------
    # Process x Varietal
    # -------------------------------------------------------------------------

    def analyze_process_by_varietal(self, top_n_varietals: int = 15) -> Dict[str, Any]:
        """Association between process method and varietal"""
        expanded = self._expand_varietals()
        if expanded.empty:
            return {'has_data': False}

        expanded = expanded[expanded['process_type'].notna()].copy()

        varietal_counts = expanded['single_varietal'].value_counts()
        top_varietals = varietal_counts[varietal_counts >= self.MIN_GROUP_SIZE].head(top_n_varietals).index.tolist()

        filtered = expanded[expanded['single_varietal'].isin(top_varietals)]
        if filtered.empty:
            return {'has_data': False}

        heatmap = pd.crosstab(filtered['single_varietal'], filtered['process_type'])
        heatmap_pct = heatmap.div(heatmap.sum(axis=1), axis=0)

        chi2_result = self._chi_square_test(heatmap)

        return {
            'has_data': True,
            'heatmap_counts': {
                'rows': heatmap.index.tolist(),
                'columns': heatmap.columns.tolist(),
                'values': heatmap.values.tolist(),
            },
            'heatmap_pct': {
                'rows': heatmap_pct.index.tolist(),
                'columns': heatmap_pct.columns.tolist(),
                'values': heatmap_pct.values.tolist(),
            },
            'chi_square': chi2_result,
        }

    # -------------------------------------------------------------------------
    # Generic Association Strength
    # -------------------------------------------------------------------------

    def compute_all_associations(self) -> List[Dict[str, Any]]:
        """
        Scan all meaningful feature pairs and compute association strength.
        Returns sorted list of strongest associations.
        """
        results = []

        # Categorical x Categorical pairs
        cat_pairs = [
            ('country', 'process_type', 'Country', 'Process Method'),
            ('country', 'single_varietal', 'Country', 'Varietal'),
            ('process_type', 'single_varietal', 'Process Method', 'Varietal'),
        ]

        # For varietal we need expanded data
        expanded = self._expand_varietals()

        for col_a, col_b, label_a, label_b in cat_pairs:
            if 'varietal' in col_b:
                source_df = expanded
            else:
                source_df = self.df

            if col_a not in source_df.columns or col_b not in source_df.columns:
                continue

            valid = source_df[[col_a, col_b]].dropna()
            if len(valid) < 20:
                continue

            contingency = pd.crosstab(valid[col_a], valid[col_b])
            # Filter small categories
            contingency = contingency.loc[
                contingency.sum(axis=1) >= self.MIN_GROUP_SIZE,
                contingency.sum(axis=0) >= self.MIN_GROUP_SIZE
            ]

            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2_result = self._chi_square_test(contingency)
            if chi2_result.get('has_data'):
                results.append({
                    'feature_a': label_a,
                    'feature_b': label_b,
                    'test': 'Chi-square',
                    'statistic': chi2_result.get('chi2'),
                    'p_value': chi2_result.get('p_value'),
                    'effect_size': chi2_result.get('cramers_v'),
                    'effect_label': "Cramer's V",
                    'is_significant': chi2_result.get('is_significant'),
                    'n_observations': chi2_result.get('n_observations'),
                })

        # Flavor family x categorical features
        for cat_col, cat_label in [('country', 'Country'), ('process_type', 'Process Method')]:
            assoc = self._flavor_category_association(cat_col, cat_label, 'family')
            if assoc:
                results.append(assoc)

        # Continuous (price) x categorical features
        for cat_col, cat_label in [
            ('country', 'Country'),
            ('process_type', 'Process Method'),
        ]:
            kw_result = self._kruskal_wallis_association(
                cat_col, 'avg_price', cat_label, 'Price'
            )
            if kw_result:
                results.append(kw_result)

        # Sort by effect size descending
        results.sort(key=lambda x: abs(x.get('effect_size', 0)), reverse=True)
        return results

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _expand_varietals(self) -> pd.DataFrame:
        """Expand multi-varietal coffees into one row per varietal"""
        rows = []
        for _, row in self.df.iterrows():
            varietals = row.get('varietals', [])
            if not isinstance(varietals, list) or not varietals:
                continue
            for v in varietals:
                if v and str(v).strip():
                    new_row = row.copy()
                    new_row['single_varietal'] = str(v).strip()
                    rows.append(new_row)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).reset_index(drop=True)

    def _categorical_distribution(self, group_col: str, value_col: str,
                                   min_group_size: int, group_label: str,
                                   value_label: str) -> Dict[str, Any]:
        """Compute distribution of value_col within each group_col category"""
        valid = self.df[[group_col, value_col]].dropna()
        if valid.empty:
            return {'has_data': False}

        # Filter groups with enough data
        group_counts = valid[group_col].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index.tolist()
        valid = valid[valid[group_col].isin(valid_groups)]

        if valid.empty:
            return {'has_data': False}

        # Cross-tabulation
        crosstab = pd.crosstab(valid[group_col], valid[value_col])
        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0)

        # Chi-square test
        chi2_result = self._chi_square_test(crosstab)

        # Per-group breakdown
        groups = []
        for group_name in crosstab.index:
            total = int(crosstab.loc[group_name].sum())
            distribution = {}
            for val in crosstab.columns:
                count = int(crosstab.loc[group_name, val])
                pct = float(crosstab_pct.loc[group_name, val])
                distribution[val] = {'count': count, 'pct': pct}
            groups.append({
                'name': str(group_name),
                'total': total,
                'distribution': distribution,
            })

        groups.sort(key=lambda x: x['total'], reverse=True)

        # Stacked bar data
        stacked_data = {
            'groups': crosstab.index.tolist(),
            'categories': crosstab.columns.tolist(),
            'counts': crosstab.values.tolist(),
            'percentages': crosstab_pct.values.tolist(),
        }

        return {
            'has_data': True,
            'groups': groups,
            'stacked_data': stacked_data,
            'chi_square': chi2_result,
            'group_label': group_label,
            'value_label': value_label,
        }

    def _chi_square_test(self, contingency_table: pd.DataFrame) -> Dict[str, Any]:
        """Run Chi-square test and compute Cramer's V"""
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return {'has_data': False}

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            k = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0

            return {
                'has_data': True,
                'chi2': float(chi2),
                'p_value': float(p_value),
                'dof': int(dof),
                'cramers_v': float(cramers_v),
                'is_significant': p_value < 0.05,
                'n_observations': int(n),
                'effect_interpretation': self._interpret_cramers_v(cramers_v),
            }
        except Exception:
            return {'has_data': False}

    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramer's V effect size"""
        if v < 0.1:
            return 'negligible'
        elif v < 0.3:
            return 'small'
        elif v < 0.5:
            return 'medium'
        else:
            return 'large'

    def _flavor_category_association(self, cat_col: str, cat_label: str,
                                      taxonomy_level: str) -> Optional[Dict[str, Any]]:
        """Compute association between flavor presence and a categorical variable"""
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        valid = self.df[self.df[cat_col].notna() & self.df['has_flavors']].copy().reset_index(drop=True)
        if len(valid) < 20:
            return None

        # Get all flavors
        all_flavors = set()
        for flist in valid[list_col]:
            if isinstance(flist, list):
                all_flavors.update(f for f in flist if f)

        if not all_flavors:
            return None

        # Build binary matrix: each row is a coffee, columns are flavors
        cat_values = valid[cat_col].value_counts()
        valid_cats = cat_values[cat_values >= self.MIN_GROUP_SIZE].index.tolist()
        valid = valid[valid[cat_col].isin(valid_cats)].reset_index(drop=True)

        # For each flavor, create a contingency table with the categorical variable
        # Aggregate: compute average Cramer's V across all flavors
        v_values = []
        for flavor in all_flavors:
            valid['has_flavor'] = valid[list_col].apply(
                lambda x: flavor in x if isinstance(x, list) else False
            )
            ct = pd.crosstab(valid[cat_col], valid['has_flavor'])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                result = self._chi_square_test(ct)
                if result.get('has_data'):
                    v_values.append(result['cramers_v'])

        if not v_values:
            return None

        avg_v = np.mean(v_values)
        return {
            'feature_a': cat_label,
            'feature_b': f'Flavor ({taxonomy_level})',
            'test': "Avg Cramer's V across flavors",
            'statistic': None,
            'p_value': None,
            'effect_size': float(avg_v),
            'effect_label': "Avg Cramer's V",
            'is_significant': None,
            'n_observations': len(valid),
        }

    def _kruskal_wallis_association(self, cat_col: str, cont_col: str,
                                     cat_label: str, cont_label: str) -> Optional[Dict[str, Any]]:
        """Compute Kruskal-Wallis association between categorical and continuous"""
        valid = self.df[[cat_col, cont_col]].dropna()
        if len(valid) < 20:
            return None

        groups = []
        for name, group in valid.groupby(cat_col):
            vals = group[cont_col].dropna()
            if len(vals) >= self.MIN_GROUP_SIZE:
                groups.append(vals.values)

        if len(groups) < 2:
            return None

        try:
            h_stat, p_value = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            k = len(groups)
            epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else 0

            return {
                'feature_a': cat_label,
                'feature_b': cont_label,
                'test': 'Kruskal-Wallis',
                'statistic': float(h_stat),
                'p_value': float(p_value),
                'effect_size': float(epsilon_sq),
                'effect_label': 'Epsilon-squared',
                'is_significant': p_value < 0.05,
                'n_observations': n,
            }
        except Exception:
            return None

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all cross-feature analyses"""
        return {
            'process_by_origin': self.analyze_process_by_origin(),
            'varietal_by_origin': self.analyze_varietal_by_origin(),
            'flavor_by_process_family': self.analyze_flavor_by_process('family'),
            'flavor_by_process_genus': self.analyze_flavor_by_process('genus'),
            'flavor_by_varietal_family': self.analyze_flavor_by_varietal('family'),
            'flavor_by_varietal_genus': self.analyze_flavor_by_varietal('genus'),
            'process_by_varietal': self.analyze_process_by_varietal(),
            'all_associations': self.compute_all_associations(),
        }
