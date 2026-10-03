"""
Flavor distinctiveness, computed over individual coffees.

For a unit type T (country / region / seller), a unit u, a taxonomy level L
(family / genus / species) and a flavor f, with coffees as the unit of analysis:

    a = coffees in u that list f          b = coffees in u without f
    c = coffees not in u that list f      d = coffees not in u without f

    share_in  = a / (a + b)               share_rest = c / (c + d)
    lift      = share_in / share_rest     (None when share_rest == 0)

Over-representation is tested with a one-sided Fisher exact test
(alternative='greater'; computed via the equivalent hypergeometric tail).
Only (u, f) pairs with n_u >= MIN_UNIT_COFFEES and a >= MIN_FLAVOR_COFFEES are
tested. P-values are Benjamini-Hochberg corrected within each (T, L) family of
tests; distinctive = q < Q_THRESHOLD (0.05) AND lift >= MIN_LIFT (1.5; a lift of
None, i.e. no other coffee lists the flavor, counts as passing). All tested rows
are kept in the flavor table; only distinctive ones feed signatures and counts.
A smoothed log-odds ratio (0.5 added to each
cell) and its standard error are reported for ranking and confidence intervals.

Unit-level summaries (per unit, per level):
  * signature: the top TOP_N_SIGNATURE distinctive flavors by smoothed
    log-odds. If none is distinctive the list is empty ("no flavor stands
    out"); it is never padded with non-significant flavors.
  * jsd: Jensen-Shannon divergence (base 2, range 0-1) between the unit's
    flavor distribution and the rest-of-market distribution. The unit side is
    shrunk toward the market: ALPHA pseudo-coffees whose flavor shares equal
    the overall market shares are added, so tiny units are not inflated.
  * diversity: Shannon entropy of the shrunk unit distribution, normalised
    by log(number of flavors present in the market), range 0-1.
Distributions are PER-COFFEE-NORMALISED: each coffee contributes total weight 1,
split evenly across its distinct flavors at the level, so a unit/seller that
writes many notes per coffee does not look different from one that writes few
(the Fisher presence test is unaffected).

Seller-support rule (origin units only: country, region): a flavor is
distinctive only if the coffees behind it (a) come from >= MIN_SELLERS distinct
sellers and (b) no single seller supplies more than MAX_SELLER_SHARE of them,
so one seller's tasting vocabulary cannot masquerade as an origin trait. Rows
carry n_sellers_supporting, top_seller_share, top_seller and seller_supported
(not applied to seller units; coffees with a missing seller count as one
"unknown" seller). The rule does not change p or q (BH is unchanged).

EXCLUDED_FLAVORS removes catch-all terms (family 'Other') from a level before
any counting; coffees keep their other flavors and their genus/species terms.

INPUT CONTRACT
One row per coffee, with columns:
    coffee_id        unique id
    country          str; NaN/None/'' when missing
    region_key       str (country + subregion); NaN/None/'' when missing
    seller           str; NaN/None/'' when missing
    flavors_family, flavors_genus, flavors_species
                     lists of strings (missing/None already removed).
                     Duplicates within a coffee are counted once.
A coffee that has an empty flavor list is still a coffee (it counts in n_u
and in the "rest").

MISSING UNIT VALUES
For unit type T, coffees whose T value is missing are excluded from the T
analysis entirely: they are neither in any unit nor in the "rest" baseline.
Coffees in units with fewer than MIN_UNIT_COFFEES coffees are not reported as
units, but they DO remain in the "rest" baseline for other units.

All outputs use native Python types (None instead of NaN/inf) via
``to_records`` so they can be written to the JSON cache.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

# ---- Tunable defaults -------------------------------------------------------
MIN_UNIT_COFFEES = 10     # smallest unit (country/region/seller) that is analysed
MIN_FLAVOR_COFFEES = 3    # a flavor must appear in >= this many of the unit's coffees to be tested
ALPHA = 10                # shrinkage pseudo-coffees (distributed per market shares)
TOP_N_SIGNATURE = 3       # flavors in a unit's signature
Q_THRESHOLD = 0.05        # BH-adjusted significance threshold
MIN_LIFT = 1.5            # minimum effect size (share_in / share_rest) to call a flavor distinctive
MIN_SELLERS = 2           # origin units: the flavor's coffees must come from >= this many sellers ...
MAX_SELLER_SHARE = 0.75   # ... and no single seller may supply more than this share of them
# Catch-all terms removed from the analysis at a level (their genera/species stay).
EXCLUDED_FLAVORS = {'family': {'Other'}}
# -----------------------------------------------------------------------------

UNIT_TYPE_COLUMNS = {'country': 'country', 'region': 'region_key', 'seller': 'seller'}
LEVELS = ('family', 'genus', 'species')

ORIGIN_UNIT_TYPES = ('country', 'region')   # seller-support rule applies to these

FLAVOR_COLUMNS = ['unit', 'flavor', 'a', 'b', 'c', 'd', 'share_in', 'share_rest',
                  'lift', 'log_odds', 'log_odds_se', 'p', 'q',
                  'n_sellers_supporting', 'top_seller_share', 'top_seller',
                  'seller_supported', 'distinctive']
UNIT_COLUMNS = ['unit', 'n_coffees', 'n_distinctive_flavors', 'signature', 'jsd', 'diversity']


def _clean_unit_series(s: pd.Series) -> pd.Series:
    s = s.where(s.notna(), None)
    return s.map(lambda v: None if v is None or (isinstance(v, str) and not v.strip()) else v)


def _flavor_matrix(lists: pd.Series, excluded=frozenset()) -> Tuple[sparse.csr_matrix, List[str]]:
    """Binary coffee x flavor matrix (each flavor counted once per coffee).
    Terms in ``excluded`` are dropped."""
    vocab: Dict[str, int] = {}
    rows, cols = [], []
    for i, fl in enumerate(lists):
        if not isinstance(fl, (list, tuple, set, np.ndarray)):
            continue
        for f in set(x for x in fl if isinstance(x, str) and x and x not in excluded):
            j = vocab.setdefault(f, len(vocab))
            rows.append(i)
            cols.append(j)
    mat = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(lists), len(vocab)),
    )
    names = [None] * len(vocab)
    for f, j in vocab.items():
        names[j] = f
    return mat, names


def _jsd_bits(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in bits for rows of p vs rows of r."""
    m = 0.5 * (p + r)
    with np.errstate(divide='ignore', invalid='ignore'):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0.0).sum(axis=1)
        kl_r = np.where(r > 0, r * np.log2(r / m), 0.0).sum(axis=1)
    return np.clip(0.5 * (kl_p + kl_r), 0.0, 1.0)


def _normalised_distributions(X, codes, n_units, keep_units, alpha):
    """Per-coffee-normalised flavor distributions.

    Each coffee contributes total weight 1 split evenly across its distinct
    flavors at the level (coffees with no flavor contribute nothing), so
    sellers/units that write many notes per coffee do not look different from
    ones that write few. Unit side is shrunk toward the market: ALPHA
    pseudo-coffees (weight 1 each) distributed per market weight shares.

    Returns (p_dist, r_dist, present): rows aligned to keep_units, columns
    restricted to the ``present`` flavor mask; each row sums to 1.
    """
    N = X.shape[0]
    k = np.asarray(X.sum(axis=1)).ravel().astype(float)
    inv = np.divide(1.0, k, out=np.zeros_like(k), where=k > 0)
    W = sparse.diags(inv) @ X
    wtot = np.asarray(W.sum(axis=0)).ravel()
    present = wtot > 0
    U = sparse.csr_matrix((np.ones(N), (codes, np.arange(N))), shape=(n_units, N))
    Aw = np.asarray((U @ W).todense())[keep_units]
    if not present.any():
        return None, None, present
    market = wtot / wtot.sum()
    shrunk = (Aw + alpha * market)[:, present]
    p_dist = shrunk / shrunk.sum(axis=1, keepdims=True)
    rest = np.clip(wtot - Aw, 0, None)[:, present]
    r_dist = rest / np.maximum(rest.sum(axis=1, keepdims=True), 1e-12)
    return p_dist, r_dist, present


def _seller_support(X, unit_codes, seller_codes, seller_names, n_units, ui_global, fi):
    """For each tested (unit, flavor) pair: number of distinct sellers among the
    unit's coffees listing the flavor, the top seller's share of them, and its name."""
    N = X.shape[0]
    n_s = len(seller_names)
    combo = unit_codes.astype(np.int64) * n_s + seller_codes
    ccodes, cuniq = pd.factorize(combo)
    Ucs = sparse.csr_matrix((np.ones(N), (ccodes, np.arange(N))), shape=(len(cuniq), N))
    M = (Ucs @ X).tocsr()
    combo_unit = (cuniq // n_s).astype(int)
    combo_seller = (cuniq % n_s).astype(int)
    n_sell = np.zeros(len(ui_global), dtype=int)
    top_share = np.zeros(len(ui_global))
    top_name = [None] * len(ui_global)
    order = np.argsort(ui_global, kind='mergesort')
    start = 0
    while start < len(order):
        u = ui_global[order[start]]
        end = start
        while end < len(order) and ui_global[order[end]] == u:
            end += 1
        idx = order[start:end]
        rows = np.where(combo_unit == u)[0]
        sub = M[rows].toarray()                       # sellers-in-unit x F
        f = fi[idx]
        cnt = sub[:, f]                               # sellers x pairs
        tot = cnt.sum(axis=0)
        n_sell[idx] = (cnt > 0).sum(axis=0)
        mx = cnt.max(axis=0)
        top_share[idx] = mx / np.maximum(tot, 1)
        am = cnt.argmax(axis=0)
        for pos, i in enumerate(idx):
            nm = seller_names[combo_seller[rows[am[pos]]]]
            top_name[i] = None if nm == _UNKNOWN_SELLER else str(nm)
        start = end
    return n_sell, top_share, top_name


_UNKNOWN_SELLER = '__unknown_seller__'


def compute_distinctiveness(df: pd.DataFrame, unit_type: str, level: str,
                            min_unit_coffees: int = MIN_UNIT_COFFEES,
                            min_flavor_coffees: int = MIN_FLAVOR_COFFEES,
                            alpha: float = ALPHA,
                            top_n: int = TOP_N_SIGNATURE,
                            q_threshold: float = Q_THRESHOLD,
                            min_lift: float = MIN_LIFT,
                            min_sellers: int = MIN_SELLERS,
                            max_seller_share: float = MAX_SELLER_SHARE
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (flavor_df, unit_df) for one (unit_type, level). See module docstring."""
    unit_col = UNIT_TYPE_COLUMNS[unit_type]
    flavor_col = f'flavors_{level}'

    work = df[[unit_col, flavor_col, 'seller']].copy() if unit_col != 'seller' \
        else df[[unit_col, flavor_col]].copy()
    work[unit_col] = _clean_unit_series(work[unit_col])
    work = work[work[unit_col].notna()].reset_index(drop=True)   # missing T excluded entirely
    N = len(work)

    empty = (pd.DataFrame(columns=FLAVOR_COLUMNS), pd.DataFrame(columns=UNIT_COLUMNS))
    if N == 0:
        return empty

    X, flavors = _flavor_matrix(work[flavor_col], EXCLUDED_FLAVORS.get(level, frozenset()))  # N x F
    F = len(flavors)
    codes, units = pd.factorize(work[unit_col], sort=True)
    n_u = np.bincount(codes, minlength=len(units))
    U = sparse.csr_matrix((np.ones(N), (codes, np.arange(N))), shape=(len(units), N))
    A = np.asarray((U @ X).todense()).astype(np.int64)          # units x F  (= a)
    col = np.asarray(X.sum(axis=0)).ravel().astype(np.int64)    # coffees with f overall

    keep_units = np.where((n_u >= min_unit_coffees) & (N - n_u > 0))[0]
    if len(keep_units) == 0 or F == 0:
        unit_rows = [{'unit': str(units[i]), 'n_coffees': int(n_u[i]), 'n_distinctive_flavors': 0,
                      'signature': [], 'jsd': None, 'diversity': None} for i in keep_units]
        return pd.DataFrame(columns=FLAVOR_COLUMNS), pd.DataFrame(unit_rows, columns=UNIT_COLUMNS)

    # ---- per-(unit, flavor) tests ------------------------------------------
    ui, fi = np.where(A[keep_units] >= min_flavor_coffees)
    ui_global = keep_units[ui]
    a = A[ui_global, fi]
    nu = n_u[ui_global]
    K = col[fi]
    b = nu - a
    c = K - a
    d = (N - nu) - c

    # One-sided Fisher exact (greater) == hypergeometric upper tail P(X >= a)
    p = stats.hypergeom.sf(a - 1, N, K, nu)
    q = stats.false_discovery_control(p, method='bh') if len(p) else p

    share_in = a / nu
    share_rest = c / (c + d)
    with np.errstate(divide='ignore', invalid='ignore'):
        lift = np.where(share_rest > 0, share_in / share_rest, np.nan)
    sa, sb, sc, sd = a + .5, b + .5, c + .5, d + .5
    log_odds = np.log(sa * sd / (sb * sc))
    log_odds_se = np.sqrt(1 / sa + 1 / sb + 1 / sc + 1 / sd)

    # ---- seller support (origin units only) ----------------------------------
    if unit_type in ORIGIN_UNIT_TYPES:
        sell = _clean_unit_series(work['seller']).fillna(_UNKNOWN_SELLER)
        scodes, snames = pd.factorize(sell)
        n_sell, top_share, top_name = _seller_support(X, codes, scodes, list(snames),
                                                      len(units), ui_global, fi)
        seller_ok = (n_sell >= min_sellers) & (top_share <= max_seller_share + 1e-9)
    else:
        n_sell = np.full(len(ui_global), np.nan)
        top_share = np.full(len(ui_global), np.nan)
        top_name = [None] * len(ui_global)
        seller_ok = np.ones(len(ui_global), dtype=bool)

    flavor_df = pd.DataFrame({
        'unit': [str(units[i]) for i in ui_global],
        'flavor': [flavors[j] for j in fi],
        'a': a, 'b': b, 'c': c, 'd': d,
        'share_in': share_in, 'share_rest': share_rest, 'lift': lift,
        'log_odds': log_odds, 'log_odds_se': log_odds_se,
        'p': p, 'q': q,
        'n_sellers_supporting': n_sell, 'top_seller_share': top_share, 'top_seller': top_name,
        'seller_supported': seller_ok,
        'distinctive': (q < q_threshold) & (np.isnan(lift) | (lift >= min_lift)) & seller_ok,
    }, columns=FLAVOR_COLUMNS)
    flavor_df = flavor_df.sort_values(['unit', 'log_odds'], ascending=[True, False],
                                      kind='mergesort').reset_index(drop=True)

    # ---- unit summaries (per-coffee-normalised distributions) ----------------
    p_dist, r_dist, present = _normalised_distributions(X, codes, len(units), keep_units, alpha)
    n_present = int(present.sum())
    if p_dist is None:
        jsd = np.full(len(keep_units), np.nan)
        diversity = np.full(len(keep_units), np.nan)
    else:
        jsd = _jsd_bits(p_dist, r_dist)
        with np.errstate(divide='ignore', invalid='ignore'):
            ent = -np.where(p_dist > 0, p_dist * np.log(p_dist), 0.0).sum(axis=1)
        diversity = ent / np.log(n_present) if n_present > 1 else np.zeros(len(keep_units))

    dist_rows = flavor_df[flavor_df['distinctive']]
    sig_by_unit = {u: g.nlargest(top_n, 'log_odds')['flavor'].tolist()
                   for u, g in dist_rows.groupby('unit')}
    ndist = dist_rows.groupby('unit').size().to_dict()

    unit_df = pd.DataFrame({
        'unit': [str(units[i]) for i in keep_units],
        'n_coffees': n_u[keep_units].astype(int),
        'n_distinctive_flavors': [int(ndist.get(str(units[i]), 0)) for i in keep_units],
        'signature': [sig_by_unit.get(str(units[i]), []) for i in keep_units],
        'jsd': jsd,
        'diversity': diversity,
    }, columns=UNIT_COLUMNS)
    return flavor_df, unit_df


def unit_share_vectors(df: pd.DataFrame, unit_type: str, level: str,
                       min_unit_coffees: int = MIN_UNIT_COFFEES,
                       alpha: float = ALPHA) -> Dict[str, Dict[str, float]]:
    """Shrunk, per-coffee-normalised flavor distributions per reported unit:
    {unit: {flavor: share}} with shares summing to 1.

    Same construction as the unit JSD (see ``_normalised_distributions``), so two
    units can be compared with ``jsd_between`` without re-reading the data.
    """
    unit_col = UNIT_TYPE_COLUMNS[unit_type]
    work = df[[unit_col, f'flavors_{level}']].copy()
    work[unit_col] = _clean_unit_series(work[unit_col])
    work = work[work[unit_col].notna()].reset_index(drop=True)
    N = len(work)
    if N == 0:
        return {}
    X, flavors = _flavor_matrix(work[f'flavors_{level}'], EXCLUDED_FLAVORS.get(level, frozenset()))
    if not flavors:
        return {}
    codes, units = pd.factorize(work[unit_col], sort=True)
    n_u = np.bincount(codes, minlength=len(units))
    keep = np.where((n_u >= min_unit_coffees) & (N - n_u > 0))[0]
    if len(keep) == 0:
        return {}
    p_dist, _, present = _normalised_distributions(X, codes, len(units), keep, alpha)
    if p_dist is None:
        return {}
    names = [flavors[j] for j in np.where(present)[0]]
    return {str(units[i]): {nm: float(p_dist[r, j]) for j, nm in enumerate(names)}
            for r, i in enumerate(keep)}


def jsd_between(shares_a: Dict[str, float], shares_b: Dict[str, float]) -> float:
    """Jensen-Shannon divergence (bits, 0-1) between two share vectors (renormalised)."""
    keys = sorted(set(shares_a) | set(shares_b))
    if not keys:
        return 0.0
    p = np.array([shares_a.get(k, 0.0) for k in keys], dtype=float)
    r = np.array([shares_b.get(k, 0.0) for k in keys], dtype=float)
    if p.sum() <= 0 or r.sum() <= 0:
        return 0.0
    return float(_jsd_bits((p / p.sum())[None, :], (r / r.sum())[None, :])[0])


def compute_all(df: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, pd.DataFrame]]:
    """All (unit_type, level) combinations: {'country_family': {'flavors': df, 'units': df}, ...}"""
    out = {}
    for t in UNIT_TYPE_COLUMNS:
        for lvl in LEVELS:
            fl, un = compute_distinctiveness(df, t, lvl, **kwargs)
            out[f'{t}_{lvl}'] = {'flavors': fl, 'units': un}
    return out


def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """JSON-cache-friendly records: native types, NaN/inf -> None."""
    records = []
    for row in df.to_dict('records'):
        rec = {}
        for k, v in row.items():
            if isinstance(v, (np.bool_, bool)):
                v = bool(v)
            elif isinstance(v, (np.integer, int)):
                v = int(v)
            elif isinstance(v, (np.floating, float)):
                v = None if (np.isnan(v) or np.isinf(v)) else float(v)
            elif isinstance(v, (list, tuple, np.ndarray)):
                v = list(v)
            rec[k] = v
        records.append(rec)
    return records


def format_distinctive_sentence(row: Dict[str, Any], unit_type: str = 'country') -> str:
    """Plain-language sentence for one distinctive (unit, flavor) row.

    e.g. "48% of coffees from Kenya list berry vs 12% of other coffees (4.0x as often)"
    """
    where = {'country': 'coffees from', 'region': 'coffees from', 'seller': 'coffees sold by'}.get(
        unit_type, 'coffees from')
    share_in = float(row['share_in']) * 100
    share_rest = float(row['share_rest']) * 100
    base = f"{share_in:.0f}% of {where} {row['unit']} list {row['flavor']} vs {share_rest:.0f}% of other coffees"
    lift = row.get('lift')
    if lift is None or (isinstance(lift, float) and (np.isnan(lift) or np.isinf(lift))):
        return base + " (no other coffee lists it)"
    return base + f" ({lift:.1f}x as often)"
