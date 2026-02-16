"""
Flavor Co-occurrence Analysis

Builds co-occurrence matrices and computes Pointwise Mutual Information (PMI)
to identify statistically surprising flavor combinations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from itertools import combinations
import math


class FlavorCooccurrenceAnalyzer:
    """Analyze flavor co-occurrence patterns across coffees"""

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.df = cross_feature_df.copy()
        self._cooccurrence_cache = {}
        self._pmi_cache = {}

    def build_cooccurrence_matrix(self, taxonomy_level: str = 'family') -> pd.DataFrame:
        """
        Build flavor x flavor co-occurrence count matrix.
        Each cell (i,j) = number of coffees containing both flavor i and flavor j.
        """
        if taxonomy_level in self._cooccurrence_cache:
            return self._cooccurrence_cache[taxonomy_level]

        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        if list_col not in self.df.columns:
            return pd.DataFrame()

        # Count co-occurrences
        cooccurrence_counts = defaultdict(int)
        flavor_counts = Counter()
        total_coffees = 0

        for _, row in self.df.iterrows():
            flavors = row.get(list_col, [])
            if not isinstance(flavors, list) or len(flavors) < 1:
                continue

            total_coffees += 1
            unique_flavors = sorted(set(f for f in flavors if f))

            # Count individual flavors
            for f in unique_flavors:
                flavor_counts[f] += 1

            # Count pairs
            for f1, f2 in combinations(unique_flavors, 2):
                cooccurrence_counts[(f1, f2)] += 1

        if not cooccurrence_counts:
            return pd.DataFrame()

        # Build matrix
        all_flavors = sorted(flavor_counts.keys())
        matrix = pd.DataFrame(0, index=all_flavors, columns=all_flavors)

        for (f1, f2), count in cooccurrence_counts.items():
            matrix.loc[f1, f2] = count
            matrix.loc[f2, f1] = count

        # Diagonal = self-count
        for f in all_flavors:
            matrix.loc[f, f] = flavor_counts[f]

        self._cooccurrence_cache[taxonomy_level] = matrix
        return matrix

    def compute_pmi(self, taxonomy_level: str = 'family') -> pd.DataFrame:
        """
        Compute Pointwise Mutual Information for flavor pairs.
        PMI = log2(P(a,b) / (P(a) * P(b)))
        Positive PMI = flavors appear together more than expected.
        Negative PMI = flavors appear together less than expected.
        """
        if taxonomy_level in self._pmi_cache:
            return self._pmi_cache[taxonomy_level]

        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        if list_col not in self.df.columns:
            return pd.DataFrame()

        # Count individual and pair occurrences
        flavor_counts = Counter()
        pair_counts = defaultdict(int)
        total_coffees = 0

        for _, row in self.df.iterrows():
            flavors = row.get(list_col, [])
            if not isinstance(flavors, list) or len(flavors) < 1:
                continue

            total_coffees += 1
            unique_flavors = sorted(set(f for f in flavors if f))

            for f in unique_flavors:
                flavor_counts[f] += 1

            for f1, f2 in combinations(unique_flavors, 2):
                pair_counts[(f1, f2)] += 1

        if not pair_counts or total_coffees == 0:
            return pd.DataFrame()

        # Compute PMI for each pair
        pmi_records = []
        for (f1, f2), pair_count in pair_counts.items():
            p_pair = pair_count / total_coffees
            p_f1 = flavor_counts[f1] / total_coffees
            p_f2 = flavor_counts[f2] / total_coffees

            if p_f1 > 0 and p_f2 > 0 and p_pair > 0:
                pmi = math.log2(p_pair / (p_f1 * p_f2))
                # Normalized PMI: PMI / -log2(P(a,b))
                npmi = pmi / (-math.log2(p_pair)) if p_pair < 1 else 0

                pmi_records.append({
                    'flavor_1': f1,
                    'flavor_2': f2,
                    'count_1': flavor_counts[f1],
                    'count_2': flavor_counts[f2],
                    'cooccurrence_count': pair_count,
                    'p_pair': p_pair,
                    'p_1': p_f1,
                    'p_2': p_f2,
                    'pmi': pmi,
                    'npmi': npmi,
                    'total_coffees': total_coffees,
                })

        pmi_df = pd.DataFrame(pmi_records)
        if not pmi_df.empty:
            pmi_df = pmi_df.sort_values('pmi', ascending=False)

        self._pmi_cache[taxonomy_level] = pmi_df
        return pmi_df

    def get_cooccurring_flavors(self, target_flavor: str, taxonomy_level: str = 'family',
                                 top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top co-occurring flavors for a given flavor (replaces placeholder)"""
        col_map = {
            'family': 'flavor_families',
            'genus': 'flavor_genera',
            'species': 'flavor_species',
        }
        list_col = col_map.get(taxonomy_level, 'flavor_families')

        if list_col not in self.df.columns:
            return []

        # Find coffees containing the target flavor
        target_coffees = self.df[self.df[list_col].apply(
            lambda x: target_flavor in x if isinstance(x, list) else False
        )]

        if target_coffees.empty:
            return []

        total_target = len(target_coffees)

        # Count co-occurring flavors
        cooccur_counts = Counter()
        for _, row in target_coffees.iterrows():
            flavors = row.get(list_col, [])
            if isinstance(flavors, list):
                for f in flavors:
                    if f and f != target_flavor:
                        cooccur_counts[f] += 1

        # Compute co-occurrence rates
        results = []
        for flavor, count in cooccur_counts.most_common(top_n):
            results.append({
                'flavor': flavor,
                'cooccurrence_count': count,
                'cooccurrence_rate': count / total_target if total_target > 0 else 0,
                'target_flavor': target_flavor,
                'target_count': total_target,
            })

        return results

    def get_distinctive_flavor_combinations(self, taxonomy_level: str = 'family',
                                             min_support: int = 5) -> List[Dict[str, Any]]:
        """
        Find flavor pairs that are statistically unusual (high PMI).
        Returns pairs sorted by PMI, filtered by minimum co-occurrence count.
        """
        pmi_df = self.compute_pmi(taxonomy_level)

        if pmi_df.empty:
            return []

        # Filter by minimum support
        filtered = pmi_df[pmi_df['cooccurrence_count'] >= min_support].copy()

        if filtered.empty:
            return []

        # Top positive PMI (appear together more than expected)
        top_positive = filtered.nlargest(15, 'pmi')
        # Top negative PMI (appear together less than expected)
        top_negative = filtered.nsmallest(10, 'pmi')

        results = {
            'surprising_pairs': top_positive.to_dict('records'),
            'avoiding_pairs': top_negative.to_dict('records'),
        }

        return results

    def get_cooccurrence_summary(self, taxonomy_level: str = 'family') -> Dict[str, Any]:
        """Get a summary of co-occurrence patterns for cache generation"""
        matrix = self.build_cooccurrence_matrix(taxonomy_level)
        pmi_df = self.compute_pmi(taxonomy_level)
        distinctive = self.get_distinctive_flavor_combinations(taxonomy_level)

        # Convert matrix to serializable format
        matrix_data = {}
        if not matrix.empty:
            matrix_data = {
                'labels': matrix.index.tolist(),
                'values': matrix.values.tolist(),
            }

        # Top co-occurring pairs by count
        top_pairs = []
        if not pmi_df.empty:
            top_by_count = pmi_df.nlargest(20, 'cooccurrence_count')
            top_pairs = top_by_count.to_dict('records')

        return {
            'matrix': matrix_data,
            'top_pairs_by_count': top_pairs,
            'distinctive_combinations': distinctive,
            'taxonomy_level': taxonomy_level,
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all co-occurrence analyses"""
        return {
            'family_level': self.get_cooccurrence_summary('family'),
            'genus_level': self.get_cooccurrence_summary('genus'),
            'species_level': self.get_cooccurrence_summary('species'),
        }
