import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_phase2_2b import synthetic_cross_feature   # noqa: E402

from analytics.frontend import cached_data_loader as cdl
from analytics.processing import distinctiveness_cache as dc
from analytics.processing.cooccurrence_analysis import FlavorCooccurrenceAnalyzer


def build_cache(include_new=True, include_legacy_rankings=True):
    cf = synthetic_cross_feature()
    cache = {
        'metadata': {'generated_at': '2099-01-01T00:00:00', 'version': '2.0', 'total_units': {}},
        'overview_data': {
            'dataset_stats': {'total_coffees': len(cf), 'countries_analyzed': 5,
                              'regions_analyzed': 9, 'sellers_analyzed': 4},
            'geographic_data': [{'country': 'Kenya', 'total_coffees': 60, 'flavor_families': 2,
                                 'sellers': 2, 'regions': 2}],
            'key_findings': [], 'top_units': [],
        },
        'flavor_hierarchies': {'families': ['Fruity', 'Sweet'],
                               'genera_by_family': {'Fruity': ['Berry']},
                               'species_by_genus': {'Berry': ['Blueberry']},
                               'all_genera': ['Berry'], 'all_species': ['Blueberry']},
        'data_completeness': {'has_data': False},
        'rankings_data': {'best_value': [{'entity_name': 'Kenya', 'unit_type': 'country',
                                          'score': 6.0, 'mean_price': 6.5, 'total_coffees': 60}],
                          'highest_priced': [], 'fastest_moving': [], 'longest_lasting': []},
    }
    if include_new:
        comps = dc.build_distinctiveness_components(dc.prepare_distinctiveness_input(cf))
        cache['distinctiveness_meta'] = comps['meta']
        cache['distinctiveness_profiles'] = comps['profiles']
        cache['distinctiveness_by_flavor'] = comps['by_flavor']
        cache['rankings_data'].update(comps['rankings'])
        cache['cooccurrence_data'] = FlavorCooccurrenceAnalyzer(cf).run_full_analysis()
        cache['overview_data']['key_findings'] = comps['meta']['key_findings']
    else:  # legacy shape: old components only
        cache['unit_profiles'] = {'country_Kenya': {'unit_name': 'Kenya', 'unit_type': 'country',
                                                    'overview': {'total_coffees': 60},
                                                    'tfidf_findings': {}, 'statistical_findings': {}}}
        cache['cooccurrence_data'] = {'family_level': {'top_pairs_by_count': []}}
    return cache


@pytest.fixture
def point_loader(tmp_path):
    def _point(cache):
        (tmp_path / 'frontend_cache.json').write_text(json.dumps(cache))
        cdl._read_cache_file.clear()
        cdl._loader = cdl.CachedDataLoader(cache_dir=str(tmp_path))
    yield _point
    cdl._loader = None
    cdl._read_cache_file.clear()


# ---- Explore by Origin ---------------------------------------------------------

def test_explore_new_cache(point_loader):
    from page_apps.analytics import flavor_origin_tab as fo
    point_loader(build_cache())
    fo.render_explore_section()
    for level in ('family', 'genus', 'species'):
        fo.display_unit_profile(cdl.load_unit_profile('Kenya', 'country'), level)
    fo.display_unit_profile(cdl.load_unit_profile('Brazil', 'country'), 'family')   # none stands out
    fo.display_unit_profile(cdl.load_unit_profile('Kenya_Nyeri', 'region') or
                            cdl.load_unit_profile('Kenya', 'country'), 'genus')


def test_explore_legacy_and_empty(point_loader):
    from page_apps.analytics import flavor_origin_tab as fo
    point_loader(build_cache(include_new=False))
    fo.render_explore_section()                      # no distinctiveness profiles: regenerate note
    fo.display_unit_profile({'unit_name': 'Kenya', 'unit_type': 'country', 'overview': {}}, 'genus')
    point_loader({})
    fo.render_explore_section()


def test_profile_support_text_and_region_labels(point_loader):
    from page_apps.analytics.common import support_text, evidence_frame, sentence_for
    point_loader(build_cache())
    p = cdl.load_unit_profile('Kenya', 'country')
    row = p['levels']['genus']['flavors'][0]
    assert 'sellers' in support_text(row) and '%' in support_text(row)
    ev = evidence_frame(p['levels']['genus']['flavors'], 'country')
    assert {'Sellers', 'Top seller share', 'q-value', 'Times as often'} <= set(ev.columns)
    # profile rows carry no unit name: it must come from the profile
    assert ' from Kenya list ' in sentence_for(row, 'country', 'Kenya')
    assert ' from  list ' not in sentence_for(row, 'country', 'Kenya')
    seller_rows = cdl.load_unit_profile('S1', 'seller')['levels']['genus']['flavors']
    assert 'Sellers' not in evidence_frame(seller_rows, 'seller').columns or not seller_rows
    regions = cdl.get_available_units('region')
    assert 'Kenya_Nyeri' in regions
    assert sentence_for({'unit': 'Kenya_Nyeri', 'share_in': .5, 'share_rest': .1, 'lift': 5.0,
                         'flavor': 'Berry'}, 'region').startswith('50% of coffees from Kenya / Nyeri')


# ---- Explore by Flavor + co-occurrence -------------------------------------------

def test_flavor_analysis_new(point_loader):
    from page_apps.analytics import flavor_origin_tab as fo
    point_loader(build_cache())
    fo.render_flavor_section()
    fo.display_flavor_analysis("All", "Berry", "All")
    fo.display_flavor_analysis("Fruity", "All", "All")
    fo.display_flavor_analysis("All", "All", "Blueberry")
    fo.display_flavor_analysis("All", "All", "All")          # nothing selected
    rows = cdl.load_flavor_unit_rows('country', 'genus', 'Berry')
    assert any(r['unit'] == 'Kenya' and r['distinctive'] for r in rows)
    comp = fo.get_cooccurring_flavors_for_flavor('Fruity', 'family')
    assert comp and {'p_b_given_a', 'p_b', 'cooccurrence_count'} <= set(comp[0])
    assert fo.get_cooccurring_flavors_for_flavor('NoSuchFlavor', 'family') == []


def test_flavor_analysis_legacy_and_empty(point_loader):
    from page_apps.analytics import flavor_origin_tab as fo
    point_loader(build_cache(include_new=False))
    assert fo.get_cooccurring_flavors_for_flavor('Fruity', 'family') is None   # legacy: no per-flavor table
    fo.display_flavor_analysis("All", "Berry", "All")
    point_loader({})
    fo.display_flavor_analysis("Fruity", "All", "All")
    assert cdl.load_flavor_unit_rows('country', 'genus', 'Berry') == []


# ---- Compare ----------------------------------------------------------------------

def test_compare_new_and_empty(point_loader):
    from page_apps.analytics import flavor_origin_tab as fo
    point_loader(build_cache())
    fo.render_compare_section()
    fo.display_comparison(['Kenya', 'Brazil'], 'country', 'genus')
    fo.display_comparison(['Kenya', 'Brazil', 'Peru'], 'country', 'family')
    fo.display_comparison(['Kenya_Nyeri', 'Kenya_Kiambu'], 'region', 'genus')
    fo.display_comparison(['Nope1', 'Nope2'], 'country', 'genus')        # nothing found
    point_loader(build_cache(include_new=False))
    fo.render_compare_section()
    fo.display_comparison(['Kenya', 'Brazil'], 'country', 'genus')


# ---- Rankings ---------------------------------------------------------------------

def test_rankings_new_and_legacy(point_loader):
    from page_apps.analytics import rankings_tab as rk
    point_loader(build_cache())
    rk.render_rankings_tab()
    cats = rk.ranking_categories()
    assert 'Highest Volume' not in cats
    assert 'Fastest Moving (Shortest Lifespan)' in cats and 'Longest Lasting' in cats   # shown with Turnover
    assert cats[:2] == ['Most distinctive flavor profile', 'Most varied flavor profile']
    df = rk.generate_profile_rankings('Most distinctive flavor profile', 'country', 'genus', 20)
    assert list(df['unit']) and (df['n_coffees'] >= 20).all()
    assert list(df['score']) == sorted(df['score'], reverse=True)
    # Slider can go down to the analysis floor of 10
    low = rk.generate_profile_rankings('Most distinctive flavor profile', 'seller', 'family', 10)
    assert (low['n_coffees'] >= 10).all()
    assert rk.MIN_COFFEES_DEFAULT == 20 and rk.MIN_COFFEES_FLOOR == 10
    for t in ('country', 'region', 'seller'):
        d = rk.generate_profile_rankings('Most varied flavor profile', t, 'genus', 10)
        rk.display_profile_rankings(d, 'Most varied flavor profile', t, 'genus')
    assert not rk.generate_rankings('Best Value (Lowest Price)', 20).empty
    rk.display_rankings(rk.generate_rankings('Best Value (Lowest Price)', 20), 'Best Value (Lowest Price)')
    assert rk.generate_rankings('Highest Priced', 20).empty                # empty list shape
    rk.display_rankings(pd.DataFrame(), 'Highest Priced')

    point_loader(build_cache(include_new=False))
    assert rk.generate_profile_rankings('Most distinctive flavor profile', 'country', 'genus', 20) is None
    rk.render_rankings_tab()
    point_loader({})
    rk.render_rankings_tab()


def test_rankings_copy():
    from page_apps.analytics import rankings_tab as rk
    assert "An origin can rank high through many small shifts even if no single flavor stands out on its own." in rk.JSD_COPY
    assert rk.JSD_COPY.startswith("This measures how different the overall flavor mix is from the rest of the market.")
    assert rk.SELLER_CAPTION == "A seller's profile reflects both which origins they stock and how they write tasting notes."


# ---- Overview ---------------------------------------------------------------------

def test_overview_new_and_legacy(point_loader):
    import page_apps.analytics_page as ap
    point_loader(build_cache())
    ap.render_overview_tab()
    kf = cdl.load_distinctiveness_meta()['key_findings']
    title, lines, _ = ap.format_key_finding(kf[0])
    assert title == kf[0]['finding']
    assert any('sellers' in l and "one seller's tasting vocabulary" in l for l in lines)
    point_loader(build_cache(include_new=False))
    ap.render_overview_tab()
    point_loader({'overview_data': {}})
    ap.render_overview_tab()
    fb = ap._generate_fallback_findings({'geographic_data': []},
                                        {'total_coffees': 10, 'countries_analyzed': 3, 'regions_analyzed': 5})
    assert fb == ['**10 coffees** analyzed across **3 countries** and **5 regions**.']


def test_main_runs_and_turnover_shown(point_loader):
    import page_apps.analytics_page as ap
    from page_apps.analytics import config
    point_loader(build_cache())
    ap.main()
    assert ap.SHOW_TURNOVER_TAB is True and config.SHOW_TURNOVER_TAB is True


# ---- Phase 3 leftovers -------------------------------------------------------------

def test_no_flavor_parse_rate_in_pages():
    import inspect
    import page_apps.analytics_page as ap
    from page_apps.analytics import flavor_origin_tab, rankings_tab
    for mod in (ap, flavor_origin_tab, rankings_tab):
        assert 'flavor_parse_rate' not in inspect.getsource(mod)
        assert 'subregion_final' not in inspect.getsource(mod)


def test_process_varietal_tab_handles_fractional_counts():
    from page_apps.analytics import process_varietal_tab as pv
    from page_apps.analytics.common import fmt_n
    assert fmt_n(12.0) == '12' and fmt_n(12.5) == '12.5' and fmt_n(3) == '3'
    data = {'has_data': True,
            'heatmap_counts': {'rows': ['Kenya', 'Peru'], 'columns': ['SL28', 'Bourbon'],
                               'values': [[10.5, 3.0], [2.25, 8.0]]},
            'heatmap_pct': {'rows': ['Kenya', 'Peru'], 'columns': ['SL28', 'Bourbon'],
                            'values': [[.78, .22], [.22, .78]]},
            'chi_square': {}}
    pv._render_varietal_by_origin(data)
    flavor = {'varietal_family': {'has_data': True,
              'profiles': {'SL28': {'total_coffees': 10.5, 'n_coffees': 14,
                                    'flavors': [{'flavor': 'Fruity', 'count': 7.5, 'rate': .71}]}},
              'distinctive': {}}}
    pv._render_flavor_by_group(flavor, 'varietal', 'Varietal', 'v')
    by_region = {'by_region': {'has_data': True, 'stacked_data': {
        'groups': ['Kenya_Nyeri'], 'categories': ['Washed'], 'percentages': [[1.0]]}, 'chi_square': {}}}
    pv._render_process_by_origin(by_region)
