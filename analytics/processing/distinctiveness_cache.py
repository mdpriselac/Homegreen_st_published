"""
Cache builders for the distinctiveness components (pure functions).

Input: the per-coffee ``cross_feature_df`` from the extractor (columns country,
region [= country+subregion key], seller, flavor_families/genera/species).
Output: JSON-friendly dicts that the generator writes to the frontend cache:

  distinctiveness_profiles  {"{unit_type}_{unit}": profile}   (Explore, Compare)
  distinctiveness_by_flavor {"{unit_type}_{level}": {"columns": [...], "rows": [[...]]}}
  rankings                  {'distinctive_profile'|'varied_profile': {type: {level: [rows]}}}
  meta                      parameters, counts, key findings
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from analytics.constants import unit_label
from analytics.processing import distinctiveness as dist

BY_FLAVOR_COLUMNS = ['unit', 'flavor', 'a', 'b', 'c', 'd', 'share_in', 'share_rest',
                     'lift', 'log_odds', 'log_odds_se', 'q',
                     'n_sellers_supporting', 'top_seller_share', 'distinctive']
PROFILE_FLAVOR_FIELDS = ['flavor', 'a', 'b', 'c', 'd', 'share_in', 'share_rest',
                         'lift', 'log_odds', 'log_odds_se', 'q',
                         'n_sellers_supporting', 'top_seller_share', 'top_seller']
N_KEY_FINDINGS = 5
KEY_FINDING_LEVELS = ('genus',)
KEY_FINDING_MIN_A = 10


def prepare_distinctiveness_input(cross_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Per-coffee frame in the distinctiveness input contract."""
    df = cross_feature_df
    return pd.DataFrame({
        'coffee_id': df['coffee_id'].values,
        'country': df['country'].values,
        'region_key': df['region'].values,       # country+subregion key (never bare subregion)
        'seller': df['seller'].values,
        'flavors_family': df['flavor_families'].values,
        'flavors_genus': df['flavor_genera'].values,
        'flavors_species': df['flavor_species'].values,
    })


def _clean(series: pd.Series) -> pd.Series:
    return dist._clean_unit_series(series)


def _unit_overview(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Counts of related units per unit (no flavor_parse_rate)."""
    d = df.copy()
    for c in ('country', 'region_key', 'seller'):
        d[c] = _clean(d[c])
    out: Dict[str, Dict[str, Dict[str, Any]]] = {'country': {}, 'region': {}, 'seller': {}}
    for u, g in d[d['country'].notna()].groupby('country'):
        out['country'][str(u)] = {'n_coffees': int(len(g)),
                                  'n_regions': int(g['region_key'].dropna().nunique()),
                                  'n_sellers': int(g['seller'].dropna().nunique())}
    for u, g in d[d['region_key'].notna()].groupby('region_key'):
        out['region'][str(u)] = {'n_coffees': int(len(g)),
                                 'country': str(g['country'].dropna().iloc[0]) if g['country'].notna().any() else None,
                                 'n_sellers': int(g['seller'].dropna().nunique())}
    for u, g in d[d['seller'].notna()].groupby('seller'):
        out['seller'][str(u)] = {'n_coffees': int(len(g)),
                                 'n_countries': int(g['country'].dropna().nunique()),
                                 'n_regions': int(g['region_key'].dropna().nunique())}
    return out


def build_distinctiveness_components(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Run compute_all and build every new cache component from it."""
    results = dist.compute_all(df, **kwargs)
    overview = _unit_overview(df)
    alpha = kwargs.get('alpha', dist.ALPHA)
    min_unit = kwargs.get('min_unit_coffees', dist.MIN_UNIT_COFFEES)

    # ---- profiles ------------------------------------------------------------
    profiles: Dict[str, Dict[str, Any]] = {}
    for t in dist.UNIT_TYPE_COLUMNS:
        for lvl in dist.LEVELS:
            fl = results[f'{t}_{lvl}']['flavors']
            un = results[f'{t}_{lvl}']['units']
            tested = fl.groupby('unit').size().to_dict() if not fl.empty else {}
            drows = fl[fl['distinctive']] if not fl.empty else fl
            by_unit = {u: g.sort_values('log_odds', ascending=False)
                       for u, g in drows.groupby('unit')} if not drows.empty else {}
            for rec in dist.to_records(un):
                u = rec['unit']
                key = f'{t}_{u}'
                prof = profiles.setdefault(key, {
                    'unit_name': u, 'unit_type': t, 'label': unit_label(t, u),
                    'n_coffees': rec['n_coffees'],
                    'overview': overview[t].get(u, {'n_coffees': rec['n_coffees']}),
                    'levels': {},
                })
                flav = (dist.to_records(by_unit[u][PROFILE_FLAVOR_FIELDS])
                        if u in by_unit else [])
                prof['levels'][lvl] = {
                    'signature': rec['signature'],
                    'n_distinctive': rec['n_distinctive_flavors'],
                    'n_tested': int(tested.get(u, 0)),
                    'jsd': rec['jsd'], 'diversity': rec['diversity'],
                    'flavors': flav,
                }
        # family-level shrunk share vectors for pairwise comparison
        for u, vec in dist.unit_share_vectors(df, t, 'family', min_unit, alpha).items():
            if f'{t}_{u}' in profiles:
                profiles[f'{t}_{u}']['family_shares'] = vec

    # ---- by-flavor tables ----------------------------------------------------
    by_flavor = {}
    for k, v in results.items():
        fl = v['flavors']
        by_flavor[k] = {
            'columns': BY_FLAVOR_COLUMNS,
            'rows': [[r[c] for c in BY_FLAVOR_COLUMNS] for r in dist.to_records(fl[BY_FLAVOR_COLUMNS])]
            if not fl.empty else [],
        }

    # ---- rankings -------------------------------------------------------------
    rankings: Dict[str, Any] = {'distinctive_profile': {}, 'varied_profile': {}}
    for t in dist.UNIT_TYPE_COLUMNS:
        for name in rankings:
            rankings[name][t] = {}
        for lvl in dist.LEVELS:
            recs = [r for r in dist.to_records(results[f'{t}_{lvl}']['units']) if r['jsd'] is not None]
            for name, field in (('distinctive_profile', 'jsd'), ('varied_profile', 'diversity')):
                rows = sorted(recs, key=lambda r: r[field], reverse=True)
                rankings[name][t][lvl] = [{
                    'unit': r['unit'], 'label': unit_label(t, r['unit']),
                    'n_coffees': r['n_coffees'], 'score': r[field],
                    'signature': r['signature'],
                } for r in rows]

    # ---- meta ------------------------------------------------------------------
    meta = {
        'params': {
            'min_unit_coffees': min_unit,
            'min_flavor_coffees': kwargs.get('min_flavor_coffees', dist.MIN_FLAVOR_COFFEES),
            'alpha': alpha,
            'top_n_signature': kwargs.get('top_n', dist.TOP_N_SIGNATURE),
            'q_threshold': kwargs.get('q_threshold', dist.Q_THRESHOLD),
            'min_lift': kwargs.get('min_lift', dist.MIN_LIFT),
            'min_sellers': kwargs.get('min_sellers', dist.MIN_SELLERS),
            'max_seller_share': kwargs.get('max_seller_share', dist.MAX_SELLER_SHARE),
            'excluded_flavors': {k: sorted(v) for k, v in dist.EXCLUDED_FLAVORS.items()},
        },
        'n_coffees_total': int(len(df)),
        'by_unit_type': {},
        'key_findings': build_key_findings(results),
    }
    for t, col in dist.UNIT_TYPE_COLUMNS.items():
        present = int(_clean(df[col]).notna().sum())
        meta['by_unit_type'][t] = {
            'n_coffees_with_value': present,
            'n_coffees_missing': int(len(df) - present),
            'n_units': {lvl: int(len(results[f'{t}_{lvl}']['units'])) for lvl in dist.LEVELS},
            'n_distinctive_pairs': {lvl: int(results[f'{t}_{lvl}']['flavors']['distinctive'].sum())
                                    if not results[f'{t}_{lvl}']['flavors'].empty else 0
                                    for lvl in dist.LEVELS},
        }

    return {'profiles': profiles, 'by_flavor': by_flavor, 'rankings': rankings, 'meta': meta}


def build_key_findings(results: Dict[str, Dict[str, pd.DataFrame]],
                       n: int = N_KEY_FINDINGS, min_a: int = KEY_FINDING_MIN_A) -> List[Dict[str, Any]]:
    """Top distinctive country-level genus findings (a >= KEY_FINDING_MIN_A; the
    seller-support rule is already part of `distinctive`), one per country,
    ranked by lift (undefined lift = strongest), then q."""
    cands = []
    for lvl in KEY_FINDING_LEVELS:
        fl = results[f'country_{lvl}']['flavors']
        if fl.empty:
            continue
        d = fl[fl['distinctive'] & (fl['a'] >= min_a)]
        for rec in dist.to_records(d):
            rec['level'] = lvl
            cands.append(rec)
    cands.sort(key=lambda r: (-(r['lift'] if r['lift'] is not None else float('inf')), r['q']))
    out, seen = [], set()
    for r in cands:
        if r['unit'] in seen:
            continue
        seen.add(r['unit'])
        out.append({
            'finding': dist.format_distinctive_sentence(r, 'country'),
            'examples': [],
            'unit_type': 'country', 'unit': r['unit'], 'level': r['level'], 'flavor': r['flavor'],
            'a': r['a'], 'q': r['q'], 'lift': r['lift'],
            'n_sellers_supporting': r['n_sellers_supporting'],
            'top_seller_share': r['top_seller_share'], 'top_seller': r['top_seller'],
        })
        if len(out) >= n:
            break
    return out
