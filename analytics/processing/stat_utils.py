"""
Shared statistical helpers: multiple-comparison correction, effect sizes,
chi-square validity guard, permutation test for price interactions and a small
Kaplan-Meier implementation.

Pure numpy/scipy/pandas (no streamlit).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# -----------------------------------------------------------------------------
# Multiple comparisons
# -----------------------------------------------------------------------------

def bh_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg q-values. None/NaN p-values stay None and are not
    counted in the family size."""
    p = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
         for v in p_values]
    idx = [i for i, v in enumerate(p) if v is not None]
    out: List[Optional[float]] = [None] * len(p)
    if idx:
        q = stats.false_discovery_control([p[i] for i in idx], method='bh')
        for i, qi in zip(idx, q):
            out[i] = float(qi)
    return out


def apply_bh(rows: List[Dict[str, Any]], p_key: str = 'p_value',
             q_key: str = 'q_value', sig_key: str = 'is_significant',
             alpha: float = ALPHA) -> None:
    """Add q_value and is_significant (q < alpha) to each dict, in place.
    The raw p_value is left untouched."""
    qs = bh_adjust([r.get(p_key) for r in rows])
    for r, q in zip(rows, qs):
        r[q_key] = q
        r[sig_key] = bool(q is not None and q < alpha)


# -----------------------------------------------------------------------------
# Effect sizes
# -----------------------------------------------------------------------------

def eta_squared_h(h_stat: float, n: int, k: int) -> float:
    """Kruskal-Wallis eta-squared based on H: (H - k + 1) / (n - k).

    (This is eta^2_H, NOT epsilon^2, which is H / ((n^2 - 1) / (n + 1)).)
    """
    return float((h_stat - k + 1) / (n - k)) if n > k else 0.0


# -----------------------------------------------------------------------------
# Chi-square with a Cochran validity guard
# -----------------------------------------------------------------------------

def cochran_ok(expected: np.ndarray) -> bool:
    """Cochran's rule: >= 80% of expected counts >= 5 and none < 1."""
    expected = np.asarray(expected, dtype=float)
    return bool((expected >= 5).mean() >= 0.8 and expected.min() >= 1)


def _collapse_rare(table: pd.DataFrame, min_total: int, collapse_cols: bool
                   ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Merge rows (and optionally columns) whose total < min_total into 'Other'.

    If the merged 'Other' bucket is itself still < min_total it is dropped
    (listed under 'dropped') because it would only add near-empty cells.
    """
    collapsed = {'rows': [], 'cols': [], 'dropped': []}
    t = table.copy()
    rows_small = t.index[t.sum(axis=1) < min_total]
    if len(rows_small):
        collapsed['rows'] = [str(r) for r in rows_small]
        other = t.loc[rows_small].sum(axis=0)
        t = t.drop(index=rows_small)
        if other.sum() >= min_total:
            t.loc['Other'] = other
        else:
            collapsed['dropped'].append('rows:Other')
    if collapse_cols:
        cols_small = t.columns[t.sum(axis=0) < min_total]
        if len(cols_small):
            collapsed['cols'] = [str(c) for c in cols_small]
            other = t[cols_small].sum(axis=1)
            t = t.drop(columns=cols_small)
            if other.sum() >= min_total:
                t['Other'] = other
            else:
                collapsed['dropped'].append('cols:Other')
    return t, collapsed


def _chi2(table: pd.DataFrame) -> Optional[Dict[str, Any]]:
    t = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return None
    chi2, p, dof, expected = stats.chi2_contingency(t)
    n = float(t.values.sum())
    k = min(t.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0
    return {'chi2': float(chi2), 'p_value': float(p), 'dof': int(dof),
            'cramers_v': v, 'n_observations': int(round(n)),
            'valid': cochran_ok(expected), 'table': t, 'expected': expected}


def monte_carlo_chi2_p(table: pd.DataFrame, n_sims: int = 2000, seed: int = 0) -> float:
    """Monte Carlo chi-square p-value with fixed margins.

    The table is expanded to one (row, column) pair per observation; the column
    labels are permuted (which keeps both margins fixed, i.e. samples the
    multivariate-hypergeometric null) and the chi-square statistic recomputed.
    p = (1 + #{chi2* >= chi2_obs}) / (n_sims + 1). Seeded, deterministic.
    """
    counts = table.to_numpy(dtype=int)
    nr, nc = counts.shape
    rows = np.repeat(np.arange(nr), counts.sum(axis=1))
    cols = np.repeat(np.tile(np.arange(nc), nr), counts.ravel())
    n = len(rows)
    expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / n
    obs = float(((counts - expected) ** 2 / expected).sum())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_sims):
        perm = rng.permutation(cols)
        sim = np.bincount(rows * nc + perm, minlength=nr * nc).reshape(nr, nc)
        stat = ((sim - expected) ** 2 / expected).sum()
        exceed += stat >= obs - 1e-9
    return float((1 + exceed) / (n_sims + 1))


def validated_chi_square(table: pd.DataFrame, min_category_total: int = 10,
                         collapse_cols: bool = True, n_sims: int = 2000,
                         seed: int = 20240405, min_n: int = 30) -> Dict[str, Any]:
    """Chi-square / Cramer's V with a Cochran guard.

    1. Table valid (Cochran) -> asymptotic p, status 'ok'.
    2. Else rows/cols with total < min_category_total are merged ONCE into
       'Other' (dropped if still < min_category_total). Valid now -> asymptotic
       p, status 'collapsed'.
    3. Else Monte Carlo p-value with fixed margins (n_sims seeded permutations),
       status 'monte_carlo' (on the collapsed table if any collapse happened).
    'insufficient' (has_data False, no p) only when < 2 rows or < 2 columns
    remain, or n < min_n. Cramer's V always comes from the tested observed table.
    """
    insufficient = {'has_data': False, 'status': 'insufficient',
                    'insufficient_data': True,
                    'message': 'insufficient data for a chi-square test'}
    empty_collapse = {'rows': [], 'cols': [], 'dropped': []}
    if table is None or table.empty:
        return dict(insufficient)
    try:
        first = _chi2(table)
        if first is None or first['n_observations'] < min_n:
            return dict(insufficient)
        result, status, collapsed = first, 'ok', empty_collapse
        if not first['valid']:
            ct, collapsed = _collapse_rare(table, min_category_total, collapse_cols)
            did_collapse = bool(collapsed['rows'] or collapsed['cols'])
            second = _chi2(ct) if did_collapse else None
            if did_collapse:
                if second is None or second['n_observations'] < min_n:
                    return dict(insufficient)
                result, status = second, 'collapsed'
            else:
                collapsed = empty_collapse
            if not result['valid']:
                result = dict(result)
                result['p_value'] = monte_carlo_chi2_p(result['table'], n_sims, seed)
                result['p_value_method'] = 'monte_carlo'
                result['n_simulations'] = n_sims
                status = 'monte_carlo'
        out = {k: v for k, v in result.items() if k not in ('table', 'expected', 'valid')}
        out.setdefault('p_value_method', 'asymptotic')
        out.update(has_data=True, status=status, collapsed=collapsed,
                   collapse_threshold=min_category_total if (collapsed['rows'] or collapsed['cols']) else None)
        return out
    except Exception:
        return dict(insufficient)


# -----------------------------------------------------------------------------
# Permutation test for a price interaction premium
# -----------------------------------------------------------------------------

def permutation_premium_test(prices: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
                             n_perm: int = 2000, seed: int = 0,
                             chunk: int = 250) -> Tuple[float, Optional[float]]:
    """Test the interaction premium of the cell A&B.

    premium = median(price | A&B) - (median(price | A) + median(price | B) - median(price))
    (the module's additive definition of the expected price).

    Null: B is unrelated to price given A. B membership is permuted among the
    rows *within* A and within not-A, which keeps |A|, |B| and |A&B| (hence all
    marginal counts) fixed. Two-sided p = (1 + #{|T*| >= |T_obs|}) / (n_perm + 1),
    deterministic for a given seed.

    Returns (observed_premium, p_value); p is None if the cell is degenerate.
    """
    prices = np.asarray(prices, dtype=float)
    ma = np.asarray(mask_a, dtype=bool)
    mb = np.asarray(mask_b, dtype=bool)
    cell = ma & mb
    n_cell, n_b = int(cell.sum()), int(mb.sum())
    if n_cell == 0:
        return float('nan'), None

    med_all = float(np.median(prices))
    y_a, y_na = prices[ma], prices[~ma]
    med_a = float(np.median(y_a))
    obs = float(np.median(prices[cell]) - (med_a + float(np.median(prices[mb])) - med_all))

    n_out = n_b - n_cell                      # B members outside A
    if n_cell == len(y_a) and n_out == len(y_na):
        return obs, None                      # nothing to permute
    rng = np.random.default_rng(seed)
    exceed, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        cell_idx = np.argpartition(rng.random((m, len(y_a))), n_cell - 1, axis=1)[:, :n_cell] \
            if n_cell < len(y_a) else np.tile(np.arange(len(y_a)), (m, 1))
        cell_vals = y_a[cell_idx]
        if n_out > 0:
            out_idx = np.argpartition(rng.random((m, len(y_na))), n_out - 1, axis=1)[:, :n_out] \
                if n_out < len(y_na) else np.tile(np.arange(len(y_na)), (m, 1))
            b_vals = np.concatenate([cell_vals, y_na[out_idx]], axis=1)
        else:
            b_vals = cell_vals
        t = np.median(cell_vals, axis=1) - (med_a + np.median(b_vals, axis=1) - med_all)
        exceed += int((np.abs(t) >= abs(obs) - 1e-12).sum())
        done += m
    return obs, float((1 + exceed) / (n_perm + 1))


# -----------------------------------------------------------------------------
# Kaplan-Meier
# -----------------------------------------------------------------------------

def kaplan_meier(durations: Sequence[float], events: Sequence[bool]) -> Dict[str, Any]:
    """Kaplan-Meier estimate for right-censored data (events: True = event seen).

    Returns times (distinct event times), survival (S just after each), n,
    n_events, median (smallest t with S(t) <= 0.5, None if never reached) and a
    quantile(q) callable-free helper via `quantile`.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    n = len(d)
    times, surv = [], []
    s = 1.0
    for t in np.unique(d[e]):
        at_risk = int((d >= t).sum())
        n_ev = int(((d == t) & e).sum())
        s *= 1.0 - n_ev / at_risk
        times.append(float(t))
        surv.append(s)
    res = {'times': times, 'survival': surv, 'n': n, 'n_events': int(e.sum())}
    res['median'] = km_quantile(times, surv, 0.5)
    res['q25'] = km_quantile(times, surv, 0.75)   # time when S drops to 0.75
    res['q75'] = km_quantile(times, surv, 0.25)   # time when S drops to 0.25
    return res


def km_quantile(times: List[float], surv: List[float], survival_level: float) -> Optional[float]:
    """Smallest event time with S(t) <= survival_level; None if never reached."""
    for t, s in zip(times, surv):
        if s <= survival_level + 1e-12:
            return float(t)
    return None


def logrank_p(durations_a, events_a, durations_b, events_b) -> Optional[float]:
    """Two-sample log-rank p-value (scipy.stats.logrank); None if not computable."""
    try:
        if len(durations_a) < 1 or len(durations_b) < 1:
            return None
        x = stats.CensoredData(uncensored=np.asarray(durations_a)[np.asarray(events_a, bool)],
                               right=np.asarray(durations_a)[~np.asarray(events_a, bool)])
        y = stats.CensoredData(uncensored=np.asarray(durations_b)[np.asarray(events_b, bool)],
                               right=np.asarray(durations_b)[~np.asarray(events_b, bool)])
        res = stats.logrank(x, y)
        p = float(res.pvalue)
        return None if np.isnan(p) else p
    except Exception:
        return None
