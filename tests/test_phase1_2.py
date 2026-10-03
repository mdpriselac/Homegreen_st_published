import inspect
import json

import numpy as np
import pandas as pd
import pytest

from analytics.processing.price_analysis import PriceAnalyzer


def make_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # Expensive significant country, cheap significant country, neutral, small
    spec = {'Pricey': (12, 40), 'Cheap': (4, 40), 'Mid1': (7, 40), 'Mid2': (7, 40), 'Tiny': (50, 3)}
    for country, (center, n) in spec.items():
        for _ in range(n):
            rows.append({
                'country': country,
                'process_type': 'Washed' if rng.random() < .5 else 'Natural',
                'price_per_lb': float(center * rng.lognormal(0, 0.1)),
                'varietals': ['Bourbon'],
                'flavor_families': ['Fruity'] if country == 'Pricey' else ['Nutty'],
                'flavor_genera': [], 'flavor_species': [],
                'first_observed': '2025-01-01', 'last_observed': '2025-06-01',
                'is_active': rng.random() < .5,
            })
    return pd.DataFrame(rows)


def test_premium_indicators_positive_significant_ranked():
    az = PriceAnalyzer(make_df())
    ind = az.find_premium_indicators()
    assert ind, "expected at least one indicator"
    prem = [i['price_premium'] for i in ind]
    assert all(p > 0 for p in prem)
    assert prem == sorted(prem, reverse=True)
    assert all(i['is_significant'] for i in ind)
    countries = [i['value'] for i in ind if i['feature'] == 'Country']
    assert 'Pricey' in countries
    assert 'Cheap' not in countries and 'Tiny' not in countries
    overall = az.df['price_per_lb'].median()
    top = next(i for i in ind if i['value'] == 'Pricey' and i['feature'] == 'Country')
    assert top['price_premium'] == pytest.approx(top['median_price'] - overall)


def test_iqr_in_group_stats_and_histogram_log_bins():
    az = PriceAnalyzer(make_df())
    res = az.price_by_country()
    g = next(x for x in res['groups'] if x['name'] == 'Pricey')
    sub = az.df[az.df['country'] == 'Pricey']['price_per_lb']
    assert g['q25'] == pytest.approx(sub.quantile(.25))
    assert g['q75'] == pytest.approx(sub.quantile(.75))
    h = az.compute_price_overview()['histogram']
    edges = np.array(h['bin_edges'])
    ratios = edges[1:] / edges[:-1]
    assert h['log_bins'] and np.allclose(ratios, ratios[0])
    assert sum(h['counts']) == len(az.df)


def test_overview_window_info():
    ov = PriceAnalyzer(make_df()).compute_price_overview()
    assert ov['window_start'] == '2025-01-01' and ov['window_end'] == '2025-06-01'
    assert ov['n_active'] + ov['n_expired'] == ov['total_with_price']


def test_price_tab_handles_old_cache_without_iqr():
    from page_apps.analytics import price_tab
    old = {'has_data': True, 'category': 'Country', 'kruskal_wallis': {},
           'groups': [{'name': 'A', 'count': 5, 'mean': 5, 'median': 5, 'std': 1, 'min': 1, 'max': 9},
                      {'name': 'B', 'count': 6, 'mean': 6, 'median': 6, 'std': 1, 'min': 1, 'max': 9}]}
    price_tab._render_price_by_category(old, 'Country')  # must not raise
    new = {**old, 'groups': [dict(g, q25=g['median'] - 1, q75=g['median'] + 1) for g in old['groups']]}
    price_tab._render_price_by_category(new, 'Country')
    price_tab._render_premium_indicators([{'feature': 'Country', 'value': 'x', 'median_price': 5, 'count': 6}])  # legacy


def test_format_key_finding_generator_shape():
    from page_apps.analytics_page import format_key_finding
    f = {'finding': 'Strong consensus on 2 distinctive family-level flavors for country units',
         'examples': [{'unit': 'Kenya', 'flavor': 'berry', 'methods': ['statistical', 'tfidf']}]}
    title, lines, metrics = format_key_finding(f)
    assert title == f['finding']
    assert lines == ['**Examples:**', '- Kenya: berry (statistical, tfidf)']
    assert metrics == {}
    t2, l2, m2 = format_key_finding({'title': 'T', 'description': 'D', 'metrics': {'a': 1}})
    assert (t2, l2, m2) == ('T', ['D'], {'a': 1})


def test_page_flags_and_ranking_option():
    import page_apps.analytics_page as ap
    assert ap.SHOW_TURNOVER_TAB is True      # shown by default (ANALYTICS_SHOW_TURNOVER=0 hides it)
    assert 'Highest Volume' not in inspect.getsource(ap)
    assert not hasattr(ap, 'generate_cache_with_progress')
    assert 'Update Cache' not in inspect.getsource(ap)


def test_loader_does_not_memoize_failure(tmp_path):
    from analytics.frontend import cached_data_loader as cdl
    cdl._read_cache_file.clear()
    loader = cdl.CachedDataLoader(cache_dir=str(tmp_path))
    f = tmp_path / 'frontend_cache.json'
    assert loader.load_overview_data() == {}          # missing
    f.write_text('{not json')
    assert loader.load_overview_data() == {}          # corrupt
    f.write_text(json.dumps({'overview_data': {'ok': 1}}))
    assert loader.load_overview_data() == {'ok': 1}   # recovers: failure not memoized
    f.write_text(json.dumps({'overview_data': {'ok': 2}}))
    cdl._read_cache_file.clear()
    loader2 = cdl.CachedDataLoader(cache_dir=str(tmp_path))
    assert loader2.load_overview_data() == {'ok': 2}  # clear -> re-read


def test_loader_returns_private_copies(tmp_path):
    from analytics.frontend import cached_data_loader as cdl
    cdl._read_cache_file.clear()
    (tmp_path / 'frontend_cache.json').write_text(json.dumps({
        'overview_data': {'a': [1, 2], 'n': {'x': 1}},
        'distinctiveness_profiles': {'country_Kenya': {'unit_type': 'country', 'unit_name': 'Kenya', 'k': [1]}},
    }))
    loader = cdl.CachedDataLoader(cache_dir=str(tmp_path))
    first = loader.load_overview_data()
    first['a'].append(99); first['n']['x'] = 'mutated'; first.pop('a')
    assert loader.load_overview_data() == {'a': [1, 2], 'n': {'x': 1}}
    prof = loader.load_unit_profile('Kenya', 'country')
    prof['k'].append(2)
    assert loader.load_unit_profile('Kenya', 'country')['k'] == [1]
    assert loader.get_available_units('country') == ['Kenya']
    # a fresh loader (another session) sees pristine data too
    assert cdl.CachedDataLoader(cache_dir=str(tmp_path)).load_overview_data()['a'] == [1, 2]


def test_premium_flavor_bh_correction_and_q_threshold():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(300):
        flav = [f'F{j}' for j in range(12) if rng.random() < 0.3]
        price = 6 * rng.lognormal(0, 0.3) * (1.6 if 'F0' in flav else 1.0)
        rows.append({'country': 'A' if i % 2 else 'B', 'process_type': 'Washed', 'price_per_lb': price,
                     'varietals': [], 'flavor_families': flav, 'flavor_genera': [], 'flavor_species': []})
    az = PriceAnalyzer(pd.DataFrame(rows))
    raw = {f['flavor']: f['p_value'] for f in az.price_by_flavor('family')['flavors']}
    ind = [i for i in az.find_premium_indicators() if i['type'] == 'flavor']
    assert 'F0' in [i['value'] for i in ind]
    qs = {i['value']: i['q_value'] for i in ind}
    assert all(q < 0.05 for q in qs.values())
    # BH-adjusted q is never smaller than the raw p, and F0's q matches scipy
    from scipy import stats
    names = list(raw)
    adj = dict(zip(names, stats.false_discovery_control([raw[n] for n in names], method='bh')))
    for n, q in qs.items():
        assert q == pytest.approx(adj[n]) and q >= raw[n]
