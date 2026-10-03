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

from scipy import sparse, stats

from analytics.processing.distinctiveness import EXCLUDED_FLAVORS

MIN_PAIR_COUNT = 3      # a pair must co-occur in at least this many coffees
TOP_N_PER_FLAVOR = 10   # co-occurring flavors kept per flavor
MIN_FLAVOR_SUPPORT = 3  # avoidance: each flavor of a pair must be listed in >= this many coffees
TOP_N_AVOIDING = 10     # avoiding pairs reported per level
AVOID_Q_THRESHOLD = 0.05


class FlavorCooccurrenceAnalyzer:
    """Analyze flavor co-occurrence patterns across coffees"""

    def __init__(self, cross_feature_df: pd.DataFrame):
        self.df = cross_feature_df.copy()
        # Same catch-all exclusions as the distinctiveness analysis (family 'Other')
        for level, col in (('family', 'flavor_families'), ('genus', 'flavor_genera'),
                           ('species', 'flavor_species')):
            excluded = EXCLUDED_FLAVORS.get(level)
            if excluded and col in self.df.columns:
                self.df[col] = self.df[col].map(
                    lambda fl, ex=excluded: [f for f in fl if f not in ex] if isinstance(fl, list) else fl)
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

    def compute_avoiding_pairs(self, taxonomy_level: str = 'family',
                               min_flavor_coffees: int = MIN_FLAVOR_SUPPORT,
                               top_n: int = TOP_N_AVOIDING,
                               q_threshold: float = AVOID_Q_THRESHOLD) -> List[Dict[str, Any]]:
        """Flavor pairs that appear together LESS often than chance would predict.

        Considers ALL pairs of flavors that each appear in at least
        ``min_flavor_coffees`` coffees, including pairs that never co-occur
        (these are never counted by the pair table). Expected count =
        n_a * n_b / N over coffees with flavor notes. A one-sided Fisher exact
        test ('less') gives p; p is Benjamini-Hochberg corrected over every pair
        tested. Reported: q < q_threshold, ordered by observed/expected (lowest
        first), then by larger expected count.
        """
        col_map = {'family': 'flavor_families', 'genus': 'flavor_genera', 'species': 'flavor_species'}
        list_col = col_map.get(taxonomy_level, 'flavor_families')
        if list_col not in self.df.columns:
            return []

        vocab: Dict[str, int] = {}
        rows, cols = [], []
        n_coffees = 0
        for fl in self.df[list_col]:
            if not isinstance(fl, list):
                continue
            uniq = {f for f in fl if f}
            if not uniq:
                continue
            for f in uniq:
                rows.append(n_coffees)
                cols.append(vocab.setdefault(f, len(vocab)))
            n_coffees += 1
        if n_coffees == 0 or not vocab:
            return []

        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)),
                              shape=(n_coffees, len(vocab)))
        names = [None] * len(vocab)
        for f, j in vocab.items():
            names[j] = f
        counts = np.asarray(X.sum(axis=0)).ravel()
        keep = np.where(counts >= min_flavor_coffees)[0]
        if len(keep) < 2:
            return []
        Xk = X[:, keep]
        C = np.asarray((Xk.T @ Xk).todense())          # observed pair counts
        nk = counts[keep]
        i_idx, j_idx = np.triu_indices(len(keep), k=1)
        obs = C[i_idx, j_idx]
        n1, n2 = nk[i_idx], nk[j_idx]
        expected = n1 * n2 / n_coffees
        # P(X <= obs) under the hypergeometric null == one-sided Fisher exact 'less'
        p = stats.hypergeom.cdf(obs, n_coffees, n1, n2)
        q = stats.false_discovery_control(p, method='bh')
        sel = np.where((q < q_threshold) & (obs < expected))[0]
        if len(sel) == 0:
            return []
        ratio = obs[sel] / expected[sel]
        order = sel[np.lexsort((-expected[sel], ratio))][:top_n]
        out = []
        for k in order:
            out.append({
                'flavor_1': names[keep[i_idx[k]]], 'flavor_2': names[keep[j_idx[k]]],
                'count_1': int(n1[k]), 'count_2': int(n2[k]),
                'cooccurrence_count': int(obs[k]),
                'expected_count': float(expected[k]),
                'ratio': float(obs[k] / expected[k]),
                'p_value': float(p[k]), 'q_value': float(q[k]),
                'total_coffees': int(n_coffees),
            })
        return out

    def get_distinctive_flavor_combinations(self, taxonomy_level: str = 'family',
                                             min_support: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Surprising pairs (high PMI, co-occurring in at least min_support coffees) and
        avoiding pairs (see compute_avoiding_pairs: all supported pairs, including
        those that never co-occur).
        """
        results: Dict[str, List[Dict[str, Any]]] = {
            'surprising_pairs': [],
            'avoiding_pairs': self.compute_avoiding_pairs(taxonomy_level),
        }
        pmi_df = self.compute_pmi(taxonomy_level)
        if not pmi_df.empty:
            filtered = pmi_df[pmi_df['cooccurrence_count'] >= min_support]
            if not filtered.empty:
                results['surprising_pairs'] = filtered.nlargest(15, 'pmi').to_dict('records')
        return results

    def conditional_by_flavor(self, taxonomy_level: str = 'family',
                              top_n: int = TOP_N_PER_FLAVOR,
                              min_count: int = MIN_PAIR_COUNT) -> Dict[str, List[Dict[str, Any]]]:
        """For every flavor A: its top co-occurring flavors B as P(B | A).

        Built from the full pair table (not a global top-N), so every flavor
        has an entry. Includes the base rate P(B) over flavored coffees and
        the ratio P(B|A) / P(B) so the page can say how much more often B
        appears alongside A than overall.
        """
        pmi_df = self.compute_pmi(taxonomy_level)
        if pmi_df.empty:
            return {}
        pmi_df = pmi_df[pmi_df['cooccurrence_count'] >= min_count]
        a_side = pd.DataFrame({
            'flavor': pmi_df['flavor_1'], 'other': pmi_df['flavor_2'],
            'n_flavor': pmi_df['count_1'], 'n_other': pmi_df['count_2'],
            'count': pmi_df['cooccurrence_count'], 'total': pmi_df['total_coffees']})
        b_side = pd.DataFrame({
            'flavor': pmi_df['flavor_2'], 'other': pmi_df['flavor_1'],
            'n_flavor': pmi_df['count_2'], 'n_other': pmi_df['count_1'],
            'count': pmi_df['cooccurrence_count'], 'total': pmi_df['total_coffees']})
        both = pd.concat([a_side, b_side], ignore_index=True)
        both['p_b_given_a'] = both['count'] / both['n_flavor']
        both['p_b'] = both['n_other'] / both['total']
        both['ratio'] = both['p_b_given_a'] / both['p_b']
        out: Dict[str, List[Dict[str, Any]]] = {}
        for flavor, g in both.groupby('flavor'):
            g = g.sort_values(['count', 'p_b_given_a'], ascending=False).head(top_n)
            out[str(flavor)] = [{
                'flavor': str(r.other), 'cooccurrence_count': int(r.count),
                'n_flavor': int(r.n_flavor), 'p_b_given_a': float(r.p_b_given_a),
                'p_b': float(r.p_b), 'ratio': float(r.ratio),
            } for r in g.itertuples()]
        return out

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
            'by_flavor': self.conditional_by_flavor(taxonomy_level),
            'taxonomy_level': taxonomy_level,
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all co-occurrence analyses"""
        return {
            'family_level': self.get_cooccurrence_summary('family'),
            'genus_level': self.get_cooccurrence_summary('genus'),
            'species_level': self.get_cooccurrence_summary('species'),
        }
