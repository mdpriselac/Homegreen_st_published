import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_phase2_2b import synthetic_cross_feature   # noqa: E402

from analytics.processing.cooccurrence_analysis import (
    FlavorCooccurrenceAnalyzer, MIN_FLAVOR_SUPPORT)


# ---- avoiding pairs ---------------------------------------------------------------

def avoidance_df():
    rows = []
    for i in range(100):
        fl = []
        if i < 40:
            fl.append('A')                  # A in coffees 0-39
        elif i < 80:
            fl.append('B')                  # B in coffees 40-79: never together with A
        if i % 2 == 0:
            fl.append('C')                  # C independent of A and B
        if i in (3, 90):
            fl.append('Rare')               # support 2 < MIN_FLAVOR_SUPPORT
        if not fl:
            fl.append('D')                  # D in the 20 leftover coffees
        rows.append({'flavor_families': fl})
    return pd.DataFrame(rows)


def test_avoiding_pairs_include_never_cooccurring_pairs():
    az = FlavorCooccurrenceAnalyzer(avoidance_df())
    # the old pair table never contains the A+B pair (zero co-occurrences)
    pmi = az.compute_pmi('family')
    assert not ((pmi.flavor_1 == 'A') & (pmi.flavor_2 == 'B')).any()
    avoid = az.compute_avoiding_pairs('family')
    ab = next(p for p in avoid if {p['flavor_1'], p['flavor_2']} == {'A', 'B'})
    assert ab['cooccurrence_count'] == 0 and ab['ratio'] == 0
    assert ab['expected_count'] == pytest.approx(40 * 40 / 100)
    assert ab['q_value'] < 0.05 and ab['q_value'] >= ab['p_value']
    # independent pair and low-support flavor are not reported
    assert not any('C' in (p['flavor_1'], p['flavor_2']) and 'A' in (p['flavor_1'], p['flavor_2'])
                   for p in avoid)
    assert not any('Rare' in (p['flavor_1'], p['flavor_2']) for p in avoid)
    assert MIN_FLAVOR_SUPPORT == 3
    json.dumps(avoid, allow_nan=False)


def test_avoiding_pairs_matches_fisher_less():
    from scipy import stats
    az = FlavorCooccurrenceAnalyzer(avoidance_df())
    ab = next(p for p in az.compute_avoiding_pairs('family')
              if {p['flavor_1'], p['flavor_2']} == {'A', 'B'})
    # 2x2: both=0, A only=40, B only=40, neither=20
    assert ab['p_value'] == pytest.approx(stats.fisher_exact([[0, 40], [40, 20]], alternative='less')[1])


def test_avoiding_pairs_sorted_and_summary_shape():
    az = FlavorCooccurrenceAnalyzer(avoidance_df())
    avoid = az.compute_avoiding_pairs('family')
    ratios = [p['ratio'] for p in avoid]
    assert ratios == sorted(ratios)
    combos = az.get_distinctive_flavor_combinations('family')
    assert set(combos) == {'surprising_pairs', 'avoiding_pairs'}
    assert combos['avoiding_pairs'] == avoid
    # degenerate inputs return the same dict shape
    empty = FlavorCooccurrenceAnalyzer(pd.DataFrame({'flavor_families': [[], []]}))
    assert empty.get_distinctive_flavor_combinations('family') == {'surprising_pairs': [], 'avoiding_pairs': []}
    summary = az.get_cooccurrence_summary('family')
    json.dumps(summary, allow_nan=False, default=str)


def test_cross_feature_tab_renders_new_and_old_avoiding_shapes():
    from page_apps.analytics import cross_feature_tab as cf
    fn = next(getattr(cf, n) for n in dir(cf) if 'cooccur' in n.lower() and n.startswith('_render'))
    az = FlavorCooccurrenceAnalyzer(avoidance_df())
    new = {'family_level': az.get_cooccurrence_summary('family'),
           'genus_level': {}, 'species_level': {}}
    fn(new)
    old = {'family_level': {'distinctive_combinations': {
        'surprising_pairs': [], 'avoiding_pairs': [{'flavor_1': 'A', 'flavor_2': 'B', 'pmi': -1.0,
                                                    'cooccurrence_count': 5}]},
        'top_pairs_by_count': []}}
    fn(old)
    fn({})


# ---- extractor ---------------------------------------------------------------------

def test_subregion_alias_applied_before_region_key():
    from analytics.db_access.coffee_data_extractor import CoffeeDataExtractor
    ex = CoffeeDataExtractor.__new__(CoffeeDataExtractor)
    attrs = pd.DataFrame({
        'coffee_id': [1, 2],
        'country_final': ['Brazil', 'Kenya'],
        'subregion_final': ['Sao Paolo', 'Nyeri'],
        'categorized_flavors': [None, None], 'process_type_final': ['Washed', 'Washed'],
        'varietal': [None, None],
        'cheapest_per_lb': [5.0, 6.0], 'average_per_lb': [5.0, 6.0], 'highest_per_lb': [5.0, 6.0],
    })
    sellers = pd.DataFrame({'coffee_id': [1, 2], 'coffee_name': ['a', 'b'], 'seller_id': [1, 1],
                            'seller_name': ['S1', 'S1'],
                            'first_observed': ['2026-01-01'] * 2, 'last_observed': ['2026-02-01'] * 2,
                            'is_active': [True, True]})
    m = ex.merge_and_prepare_data(attrs, sellers)
    assert m.loc[0, 'subregion_final'] == 'Sao Paulo'            # raw column agrees ...
    assert m.loc[0, 'region_key'] == 'Brazil_Sao Paulo'          # ... with the key
    cf = ex.prepare_cross_feature_format(m)
    assert cf.loc[0, 'subregion'] == 'Sao Paulo' and cf.loc[0, 'region'] == 'Brazil_Sao Paulo'


def test_extractor_old_formats_removed():
    from analytics.db_access.coffee_data_extractor import CoffeeDataExtractor
    for name in ('aggregate_by_country', 'aggregate_by_region', 'aggregate_by_seller',
                 'prepare_contingency_format', 'prepare_tfidf_format', 'prepare_hierarchical_format',
                 '_count_flavors_by_level', '_build_hierarchy_tree', '_find_parent_family'):
        assert not hasattr(CoffeeDataExtractor, name), name


# ---- generator ---------------------------------------------------------------------

def test_old_modules_and_methods_are_gone():
    for mod in ('statistical_analysis', 'tfidf_analysis', 'hierarchical_analysis',
                'integrated_analysis', 'test_analytics'):
        with pytest.raises(ImportError):
            importlib.import_module(f'analytics.processing.{mod}')
    from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator as G
    from analytics.frontend import cached_data_loader as cdl
    for name in ('_generate_unit_profiles_cache', '_generate_unit_profile', '_extract_statistical_findings',
                 '_extract_tfidf_findings', '_extract_hierarchical_findings', '_extract_consensus_findings',
                 '_generate_recommendations', '_get_top_flavors_for_entity',
                 '_generate_comparison_cache', '_generate_export_cache'):
        assert not hasattr(G, name), name
    assert not hasattr(cdl.CachedDataLoader, 'load_comparison_matrices')


def make_generator(tmp_path):
    from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator
    g = FrontendDataCacheGenerator(cache_dir=str(tmp_path))
    cf = synthetic_cross_feature()
    g.all_results = {'data': {'cross_feature_df': cf}}
    g.distinctiveness = g._compute_distinctiveness()
    return g, cf


def test_geographic_data_from_cross_feature_df(tmp_path):
    g, cf = make_generator(tmp_path)
    geo = {r['country']: r for r in g._extract_geographic_data()}
    assert set(geo) == {'Kenya', 'Brazil', 'Peru', 'Ethiopia', 'Fiji'}
    k = geo['Kenya']
    assert set(k) == {'country', 'total_coffees', 'flavor_families', 'sellers', 'regions'}
    assert k['total_coffees'] == 60 and k['sellers'] == 2 and k['regions'] == 2
    assert k['flavor_families'] == 2                      # Fruity + Sweet
    assert geo['Brazil']['total_coffees'] == 150
    json.dumps(list(geo.values()))


def test_metadata_version_and_single_cache_file(tmp_path):
    g, _ = make_generator(tmp_path)
    meta = g._generate_metadata()
    assert meta['version'] == '2.0'
    assert set(meta['total_units']) == {'country', 'region', 'seller'}
    g._save_cache_data({'metadata': meta, 'overview_data': {'x': np.int64(3)}})
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ['frontend_cache.json']               # no duplicate per-component files
    assert json.loads((tmp_path / 'frontend_cache.json').read_text())['overview_data'] == {'x': 3}


def test_rankings_cache_has_no_old_keys(tmp_path):
    g, cf = make_generator(tmp_path)
    cf = cf.assign(price_per_lb=5.0, is_active=False, lifespan_days=30.0)
    g.all_results['data']['cross_feature_df'] = cf
    r = g._generate_rankings_cache()
    for old in ('most_distinctive', 'most_specialized', 'most_diverse'):
        assert old not in r
    assert {'distinctive_profile', 'varied_profile', 'best_value', 'highest_priced'} <= set(r)
