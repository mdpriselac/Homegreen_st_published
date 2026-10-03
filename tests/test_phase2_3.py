import contextlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from page_apps.analytics import common  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def rec(monkeypatch):
    """Capture everything the pages write (text) and every plotly figure title."""
    out = {'text': [], 'figs': [], 'tables': []}

    def grab(name):
        def fn(*a, **k):
            out['text'].append(str(a[0]) if a else '')
            return None
        return fn
    for name in ('markdown', 'caption', 'info', 'success', 'warning', 'write', 'subheader', 'header', 'error'):
        monkeypatch.setattr(st, name, grab(name))
    monkeypatch.setattr(st, 'metric', lambda label, value=None, *a, **k: out['text'].append(f"{label}: {value}"))
    monkeypatch.setattr(st, 'plotly_chart', lambda fig, *a, **k: out['figs'].append(fig))
    monkeypatch.setattr(st, 'dataframe', lambda df, *a, **k: out['tables'].append(df))
    monkeypatch.setattr(st, 'expander', lambda *a, **k: (out['text'].append('EXPANDER ' + str(a[0])),
                                                         contextlib.nullcontext())[1])
    out['all'] = lambda: "\n".join(out['text'])
    out['titles'] = lambda: [f.layout.title.text for f in out['figs']]
    return out


# ---- formatting helpers --------------------------------------------------------------

def test_fmt_helpers():
    assert common.fmt_p(0.0004997, 'monte_carlo', 2000) == "p < 0.001"
    assert common.fmt_p(0.012, 'monte_carlo', 2000) == "p = 0.012"
    assert common.fmt_p(1e-9) == "p < 0.001"
    assert common.fmt_p(None) == "n/a"
    assert common.fmt_q(0.02) == "0.020" and common.fmt_q(1e-9) == "1.0e-09" and common.fmt_q(None) == "n/a"
    assert common.effect_label(0.2, 'eta2_h') == 'large' and common.effect_label(0.2, 'cramers_v') == 'small'
    assert common.effect_metric_of({'effect_size_metric': 'eta2_h'}) == 'eta2_h'
    assert common.effect_metric_of({'effect_label': "Avg Cramer's V"}) == 'mean_cramers_v'
    assert common.effect_metric_of({'effect_label': 'Epsilon-squared'}) == 'eta2_h'
    note = common.collapsed_text({'rows': ['Haiti'], 'cols': ['Kurume'], 'dropped': ['cols:Other']}, 10)
    assert "fewer than 10 coffees" in note and "Haiti" in note and "Kurume" in note
    assert common.collapsed_text({'rows': [], 'cols': [], 'dropped': []}) is None
    assert "Epsilon" not in common.ETA2_NOTE and "Eta-squared (H)" in common.ETA2_NOTE


# ---- chi-square status display ---------------------------------------------------------

BASE = {'chi2': 120.5, 'p_value': 0.03, 'dof': 6, 'cramers_v': 0.31, 'n_observations': 500,
        'p_value_method': 'asymptotic', 'has_data': True, 'status': 'ok',
        'collapsed': {'rows': [], 'cols': [], 'dropped': []}, 'collapse_threshold': None,
        'is_significant': True, 'effect_size_metric': 'cramers_v', 'effect_interpretation': 'medium'}


def test_chi_square_ok(rec):
    common.render_chi_square(dict(BASE), "country and process")
    t = rec['all']()
    assert "p = 0.030" in t and "Monte Carlo" not in t and "merged" not in t


def test_chi_square_collapsed_shows_what_merged(rec):
    c = dict(BASE, status='collapsed', collapse_threshold=10,
             collapsed={'rows': ['Haiti', 'Kenya'], 'cols': ['Kurume'], 'dropped': []})
    common.render_chi_square(c, "x")
    t = rec['all']()
    assert "Haiti, Kenya" in t and "Kurume" in t and "fewer than 10 coffees" in t


def test_chi_square_monte_carlo_at_floor(rec):
    c = dict(BASE, status='monte_carlo', p_value=1 / 2001, p_value_method='monte_carlo', n_simulations=2000)
    common.render_chi_square(c, "x")
    t = rec['all']()
    assert "Monte Carlo p, 2,000 simulations" in t and "p < 0.001" in t and "p = 0.000" not in t


def test_chi_square_insufficient_has_no_p(rec):
    common.render_chi_square({'has_data': False, 'status': 'insufficient', 'insufficient_data': True}, "x")
    t = rec['all']()
    assert "nsufficient data" in t and "p =" not in t and "p <" not in t


def test_process_varietal_tab_uses_status_display(rec):
    from page_apps.analytics import process_varietal_tab as pv
    mc = dict(BASE, status='monte_carlo', p_value=1 / 2001, p_value_method='monte_carlo', n_simulations=2000)
    hm = {'rows': ['A', 'B'], 'columns': ['x', 'y'], 'values': [[1.5, 2.0], [3.0, 0.5]]}
    pv._render_varietal_by_origin({'has_data': True, 'heatmap_counts': hm, 'heatmap_pct': hm, 'chi_square': mc})
    assert "Monte Carlo p, 2,000 simulations" in rec['all']()
    pv._render_process_by_varietal({'has_data': True, 'heatmap_counts': hm, 'heatmap_pct': hm,
                                    'chi_square': {'has_data': False, 'status': 'insufficient'}})
    assert "nsufficient data" in rec['all']()
    st_data = {'by_region': {'has_data': True, 'stacked_data': {'groups': ['Kenya_Nyeri'], 'categories': ['W'],
                                                                'percentages': [[1.0]]},
                             'chi_square': dict(BASE, status='collapsed')},
               'by_country': {'has_data': False}}
    pv._render_process_by_origin(st_data)          # default radio = Country: not enough data path
    pv._render_process_by_origin({'by_country': st_data['by_region']})


# ---- cross-feature discovery ------------------------------------------------------------

ASSOC = [
    {'feature_a': 'Country', 'feature_b': 'Varietal', 'test': 'Chi-square', 'statistic': 5390.0,
     'p_value': 1 / 2001, 'effect_size': 0.496, 'effect_label': "Cramer's V", 'effect_size_metric': 'cramers_v',
     'test_status': 'monte_carlo', 'collapsed': {'rows': ['China'], 'cols': [], 'dropped': []},
     'is_significant': True, 'n_observations': 1095, 'q_value': 1 / 2001},
    {'feature_a': 'Country', 'feature_b': 'Flavor (family)', 'test': "Avg Cramer's V across flavors",
     'statistic': None, 'p_value': None, 'effect_size': 0.29, 'effect_label': "Avg Cramer's V",
     'effect_size_metric': 'mean_cramers_v', 'is_significant': None, 'n_observations': 2620,
     'n_flavors_tested': 8, 'n_flavors_total': 9},
    {'feature_a': 'Country', 'feature_b': 'Price', 'test': 'Kruskal-Wallis', 'statistic': 519.0,
     'p_value': 6e-90, 'effect_size': 0.198, 'effect_label': 'Eta-squared (H)', 'effect_size_metric': 'eta2_h',
     'is_significant': True, 'n_observations': 2496, 'q_value': 1.2e-89},
]


def test_discovery_separate_charts_q_and_notes(rec):
    from page_apps.analytics import cross_feature_tab as cf
    cf._render_discovery({'all_associations': ASSOC,
                          'insufficient_association_tests': [
                              {'feature_a': 'Process Method', 'feature_b': 'Varietal',
                               'test': 'Chi-square', 'status': 'insufficient data'}]})
    titles = rec['titles']()
    assert len(titles) == 3
    assert any("Cramér's V" in t and "Mean" not in t for t in titles)
    assert any("Eta-squared (H)" in t for t in titles)
    assert any("Mean Cramér's V" in t for t in titles)
    tbl = rec['tables'][0]
    assert 'q-value' in tbl.columns and 'Notes' in tbl.columns
    assert tbl.loc[tbl['Pair'] == 'Country x Flavor (family)', 'Notes'].iloc[0].startswith("8 of 9 flavors tested")
    assert tbl.loc[tbl['Pair'] == 'Country x Varietal', 'p'].iloc[0] == "p < 0.001"
    assert tbl.loc[tbl['Pair'] == 'Country x Flavor (family)', 'p'].iloc[0] == "n/a"
    t = rec['all']()
    assert "Not enough data to test: Process Method x Varietal" in t
    assert "psilon" not in t and "Eta-squared (H)" in t


def test_discovery_old_cache_without_metric_or_q(rec):
    from page_apps.analytics import cross_feature_tab as cf
    old = [{'feature_a': 'A', 'feature_b': 'B', 'test': 'Chi-square', 'p_value': 0.01, 'effect_size': 0.2,
            'effect_label': "Cramer's V", 'is_significant': True, 'n_observations': 100},
           {'feature_a': 'A', 'feature_b': 'C', 'test': 'Kruskal-Wallis', 'p_value': 0.2, 'effect_size': 0.01,
            'effect_label': 'Epsilon-squared', 'is_significant': False, 'n_observations': 100}]
    cf._render_discovery({'all_associations': old})
    assert len(rec['figs']) == 2
    cf._render_discovery({})
    cf._render_discovery({'all_associations': []})


# ---- price tab ---------------------------------------------------------------------------

def test_price_tab_eta_q_and_varietal_note(rec):
    from page_apps.analytics import price_tab as pt
    groups = [{'name': n, 'count': 20, 'mean': 10, 'median': 9 + i, 'std': 2, 'min': 3, 'max': 20,
               'q25': 7, 'q75': 12, 'p_value': 0.01, 'q_value': 0.02, 'is_significant': True}
              for i, n in enumerate(['SL28', 'Bourbon', 'Geisha'])]
    kw = {'has_data': True, 'h_statistic': 30.0, 'p_value': 1e-5, 'is_significant': True, 'n_groups': 3,
          'effect_size': 0.1, 'eta_squared_h': 0.1, 'effect_size_metric': 'eta2_h'}
    pt._render_price_by_category({'has_data': True, 'groups': groups, 'kruskal_wallis': kw,
                                  'category': 'Varietal'}, 'Varietal')
    t = rec['all']()
    assert "eta-squared (H)=0.100 [medium]" in t and "single-varietal coffees only" in t
    assert "psilon" not in t
    assert 'q-value' in rec['tables'][0].columns
    # older cache without q_value still renders
    for g in groups:
        g.pop('q_value')
    pt._render_price_by_category({'has_data': True, 'groups': groups, 'kruskal_wallis': kw}, 'Country')
    # flavor table: q-value shown
    fl = [{'flavor': 'Fruity', 'count_with': 50, 'count_without': 50, 'mean_price_with': 9,
           'mean_price_without': 8, 'median_price_with': 9, 'median_price_without': 8,
           'price_difference': 1.0, 'price_ratio': 1.1, 'p_value': 0.001, 'q_value': 0.01, 'is_significant': True}]
    pt._render_price_by_flavor({'by_flavor_family': {'has_data': True, 'flavors': fl}})
    assert 'q-value' in rec['tables'][-1].columns and 'P-Value' not in rec['tables'][-1].columns
    pt._render_price_by_flavor({'by_flavor_family': {'has_data': True,
                                                     'flavors': [{k: v for k, v in fl[0].items() if k != 'q_value'}]}})
    assert 'P-Value' in rec['tables'][-1].columns


# ---- interaction help text ----------------------------------------------------------------

def test_interaction_help_and_q(rec):
    from page_apps.analytics import cross_feature_tab as cf
    effect = {'feature_a_value': 'Kenya', 'feature_b_value': 'Washed', 'flavor': 'Fruity', 'observed_rate': .8,
              'expected_rate': .5, 'interaction_score': .3, 'sample_size': 20, 'effect_type': 'emergent',
              'is_significant': True, 'p_value': 0.001, 'q_value': 0.01}
    idata = {'has_data': True, 'valid_combinations': 3, 'top_emergent': [effect], 'top_suppressed': [],
             'combination_profiles': {}, 'feature_a_label': 'Origin', 'feature_b_label': 'Process'}
    cf._render_flavor_interactions({'origin_process_flavor_family': idata})
    t = rec['all']()
    assert "binomial test" in t and "single-varietal coffees only" in t and "q-value" in t
    assert 'q-value' in rec['tables'][0].columns   # detailed data table
    price = {'feature_a_value': 'Peru', 'feature_b_value': 'Geisha', 'sample_size': 15, 'median_price': 17.9,
             'expected_price': 27.1, 'price_premium': -9.2, 'effect_type': 'discount', 'is_significant': True,
             'p_value': 0.0005, 'q_value': 0.0027}
    pdata = {'has_data': True, 'combinations': [price], 'top_premiums': [], 'top_discounts': [price],
             'feature_a_label': 'Origin', 'feature_b_label': 'Flavor', 'global_median_price': 10.0,
             'valid_combinations': 1}
    cf._render_price_interactions({'origin_flavor_price': pdata})
    t = rec['all']()
    assert "permutation test" in t and "2,000 permutations" in t and "single-varietal" in t
    assert any('q=' in str(y) for f in rec['figs'] for y in f.data[0].y)       # q in chart labels
    assert 'q-value' in rec['tables'][-1].columns


# ---- turnover tab --------------------------------------------------------------------------

WINDOW = {'window_start': '2024-04-05', 'window_end': '2026-10-03', 'first_scrape': '2024-03-20',
          'initial_inventory_cutoff': '2024-04-05', 'tracking_start': '2025-06-18', 'seller_inactive_days': 60,
          'scrape_interval_days': 7.0,
          'duration_definition': 'Each sighting stands for one scrape interval I (median gap between scrapes).',
          'exclusions': {'total_coffees': 2740, 'invalid_or_missing_dates': 0,
                         'left_censored_initial_inventory': 503, 'expired_before_tracking': 872,
                         'eligible': 1365, 'eligible_expired': 1070, 'eligible_active_censored': 295}}


def turnover_data(median=78.0):
    grp = lambda n, med, **k: dict({'count': 30, 'events': 20, 'censored': 10, 'median_lifespan': med,
                                    'q25_lifespan': 20.0 if med else None, 'q75_lifespan': 90.0 if med else None,
                                    'p_value': 0.01, 'q_value': 0.02, 'is_significant': True}, **k)
    return {
        'observation_window': WINDOW,
        'lifespan_overview': {'has_data': True, 'total_expired': 1070, 'total_active': 295,
                              'median_days': median, 'q25_days': 30.0 if median else None, 'q75_days': 249.0,
                              'mean_days': 119.0, 'std_days': 160.0, 'min_days': 3.5, 'max_days': 913.5,
                              'mean_is_biased_note': 'x',
                              'km_curve': {'times': [7.0, 14.0, 30.0], 'survival': [0.9, 0.7, 0.4]},
                              'histogram': {'counts': [1], 'bin_edges': [0, 1], 'bin_labels': ['0-1']}},
        'by_seller': {'has_data': True, 'test': 'log-rank (seller vs all other eligible coffees), BH',
                      'sellers': [grp(1, 7.0, seller='Mill City'), grp(2, None, seller='Slow Seller'),
                                  grp(3, 120.0, seller='Other')],
                      'excluded_inactive_sellers': ['Copan', 'Lost Dutchman']},
        'by_origin': {'has_data': True, 'test': 'log-rank',
                      'countries': [grp(1, 24.0, country='Haiti'), grp(2, 60.0, country='Peru')]},
        'by_process': {'has_data': True, 'test': 'log-rank',
                       'processes': [grp(1, 24.0, process='Washed'), grp(2, 60.0, process='Natural')]},
        'by_price': {'has_data': True, 'correlation': -0.003, 'correlation_p_value': 0.9,
                     'correlation_note': 'Spearman on expired coffees only (censoring-biased)',
                     'p_value': 0.04, 'is_significant': True, 'test': 'log-rank, Budget vs Premium quartile',
                     'n_observations': 1324,
                     'quartile_stats': [{'quartile': 'Budget', 'count': 300, 'events': 200,
                                         'median_lifespan': 84.0, 'mean_price_per_lb': 6.8},
                                        {'quartile': 'Premium', 'count': 300, 'events': 100,
                                         'median_lifespan': None, 'mean_price_per_lb': 30.0}],
                     'scatter_data': {'prices': [5.0, 9.0], 'lifespans': [10.0, 200.0]}},
        'seller_summary': [{'seller': 'Mill City', 'total_coffees': 15, 'active': 0, 'expired': 15,
                            'turnover_rate': 1.0, 'median_lifespan_days': 7.0, 'unique_countries': 3}],
        'seasonal_patterns': {'has_data': True, 'appearances': [{'month': '2024-04', 'count': 10}],
                              'disappearances': [{'month': '2025-07', 'count': 5}],
                              'excluded_first_scrape_month': '2024-03', 'disappearances_from': '2025-06-18'},
    }


@pytest.fixture
def point_turnover(monkeypatch):
    from analytics.frontend import cached_data_loader as cdl

    def _point(data):
        monkeypatch.setattr(cdl, 'load_turnover_data', lambda: data)
    return _point


def test_turnover_tab_renders_survival_outputs(rec, point_turnover):
    from page_apps.analytics import turnover_tab as tt
    point_turnover(turnover_data())
    tt.render_turnover_tab()
    t = rec['all']()
    # short caption: two sentences with cutoff and typical listing
    cap = next(x for x in rec['text'] if x.startswith("Lifespans are estimated"))
    assert "weekly scrapes" in cap and "first listed after 2024-04-05" in cap
    assert 'at least this long' in cap and "Typical listing: 78 days." in cap
    assert cap.count('. ') + cap.count('.\n') <= 2 and len(cap) < 260
    # expander carries the method detail
    assert "EXPANDER How we measure this" in t
    for needle in ("2024-03-20", "1,365" if False else "1365", "503 from the initial inventory",
                   "872 removed before", "scrape interval", "Closed sellers", "Copan", "log-rank",
                   "Benjamini-Hochberg"):
        assert needle in t, needle
    # None median -> "not reached", never the string None / nan
    tbl = next(df for df in rec['tables'] if 'Seller' in df.columns and 'Median (days)' in df.columns)
    assert tbl.loc[tbl['Seller'] == 'Slow Seller', 'Median (days)'].iloc[0] == 'not reached'
    assert 'q-value' in tbl.columns and 'Removed' in tbl.columns and 'Still listed' in tbl.columns
    assert tbl.loc[tbl['Seller'] == 'Mill City', 'Middle half (days)'].iloc[0] == "20 to 90"
    assert "Not reached means fewer than half" in t
    # KM curve and no expired-only mean/std
    assert any("still listed after N days" in x for x in rec['titles']())
    assert "Mean Lifespan" not in t and "Std Dev" not in t and "Kruskal" not in t and "psilon" not in t
    # excluded sellers + seasonal captions
    assert "Left out: sellers with no listings seen in the last 60 days" in t
    assert "2024-03" in t and "from 2025-06-18" in t


def test_turnover_tab_median_not_reached_and_legacy(rec, point_turnover):
    from page_apps.analytics import turnover_tab as tt
    point_turnover(turnover_data(median=None))
    tt.render_turnover_tab()
    t = rec['all']()
    assert "not yet known (fewer than half have left yet)" in t
    assert "Typical listing (median): not reached" in t
    # legacy shape (old cache): no window, no km_curve, old keys: must not crash
    point_turnover({'lifespan_overview': {'has_data': True, 'total_expired': 5, 'total_active': 2,
                                          'median_days': 30.0, 'q25_days': 10.0, 'q75_days': 50.0},
                    'by_seller': {'has_data': True, 'sellers': [{'seller': 'S', 'count': 3,
                                                                 'median_lifespan': 30.0}]}})
    tt.render_turnover_tab()
    point_turnover({})
    tt.render_turnover_tab()
    point_turnover({'lifespan_overview': {'has_data': False}})
    tt.render_turnover_tab()


# ---- lifespan rankings (generator + page) ------------------------------------------------------

def test_lifespan_rankings_from_km_medians(tmp_path, monkeypatch, rec):
    from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator
    from page_apps.analytics import rankings_tab as rk
    from test_phase2_2b import synthetic_cross_feature
    g = FrontendDataCacheGenerator(cache_dir=str(tmp_path))
    cf = synthetic_cross_feature().assign(price_per_lb=5.0)
    g.all_results = {'data': {'cross_feature_df': cf}}
    g.distinctiveness = g._compute_distinctiveness()
    g._turnover = turnover_data()
    r = g._generate_rankings_cache()
    names = [(e['unit_type'], e['entity_name']) for e in r['fastest_moving']]
    assert ('seller', 'Slow Seller') not in names                    # median not reached: not ranked
    assert names[0] == ('seller', 'Mill City') and ('country', 'Peru') in names
    assert [e['score'] for e in r['fastest_moving']] == sorted(e['score'] for e in r['fastest_moving'])
    assert [e['score'] for e in r['longest_lasting']] == sorted((e['score'] for e in r['longest_lasting']), reverse=True)
    assert all({'q25_lifespan', 'q75_lifespan', 'total_coffees'} <= set(e) for e in r['fastest_moving'])
    # page: hidden by default, shown (both unit types) when the flag is on
    assert any('Fastest' in c for c in rk.ranking_categories())              # shown by default
    monkeypatch.setattr(rk, 'SHOW_TURNOVER_TAB', False)
    assert not any('Fastest' in c or 'Longest' in c for c in rk.ranking_categories())
    monkeypatch.setattr(rk, 'SHOW_TURNOVER_TAB', True)
    assert 'Fastest Moving (Shortest Lifespan)' in rk.ranking_categories()
    from analytics.frontend import cached_data_loader as cdl
    monkeypatch.setattr(rk, 'load_rankings_data', lambda: r)
    c = rk.generate_rankings('Fastest Moving (Shortest Lifespan)', 10, 'seller')
    assert set(c['unit_type']) == {'seller'} and c.iloc[0]['entity_name'] == 'Mill City'
    rk.display_rankings(c, 'Fastest Moving (Shortest Lifespan)')
    assert 'Median Lifespan (days)' in rec['tables'][-1].columns and 'Middle half (days)' in rec['tables'][-1].columns
    country = rk.generate_rankings('Longest Lasting', 10, 'country')
    assert country.iloc[0]['entity_name'] == 'Peru'
    rk.render_rankings_tab()                                         # lifespan category not selected: ok
    assert rk.generate_rankings('Fastest Moving (Shortest Lifespan)', 10, 'country') is not None


# ---- co-occurrence 'Other' exclusion ---------------------------------------------------------------

def test_cooccurrence_excludes_family_other():
    from analytics.processing.cooccurrence_analysis import FlavorCooccurrenceAnalyzer
    rows = [{'flavor_families': ['Fruity', 'Other'], 'flavor_genera': ['Other', 'Berry'], 'flavor_species': []}
            for _ in range(30)]
    rows += [{'flavor_families': ['Fruity', 'Sweet'], 'flavor_genera': ['Other', 'Berry'], 'flavor_species': []}
             for _ in range(30)]
    az = FlavorCooccurrenceAnalyzer(pd.DataFrame(rows))
    fam = az.conditional_by_flavor('family')
    assert 'Other' not in fam and all(c['flavor'] != 'Other' for c in fam['Fruity'])
    assert 'Other' not in {p for pair in az.compute_pmi('family')[['flavor_1', 'flavor_2']].values for p in pair}
    assert 'Other' in az.conditional_by_flavor('genus')               # genus-level 'Other' is meaningful


# ---- cleanup ------------------------------------------------------------------------------------------

def test_cleanup_files_gone_and_no_deprecation_warning():
    for rel in ('generate_cache.py', 'test_analytics_report.py', 'analytics/db_access/debug_flavors.py'):
        assert not (ROOT / rel).exists(), rel
    assert (ROOT / 'generate_cache_standalone.py').exists()
    import importlib
    import analytics.db_access.coffee_data_extractor as ex
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        importlib.reload(ex)
    assert not [x for x in w if 'no_silent_downcasting' in str(x.message)]
    assert not hasattr(ex.get_analytics_data, 'clear')               # no st.cache_data wrapper
    assert not hasattr(ex.CoffeeDataExtractor.extract_raw_data, 'clear')


# ---- 2.3b: environment overrides ---------------------------------------------------------------

def test_loader_cache_dir_from_environment(tmp_path, monkeypatch):
    import json
    from analytics.frontend import cached_data_loader as cdl
    (tmp_path / 'frontend_cache.json').write_text(json.dumps({'overview_data': {'ok': 1}}))
    monkeypatch.setenv('ANALYTICS_CACHE_DIR', str(tmp_path))
    cdl._read_cache_file.clear()
    loader = cdl.CachedDataLoader()
    assert loader.cache_dir == tmp_path
    assert loader.load_overview_data() == {'ok': 1}
    other = tmp_path / 'explicit'
    assert cdl.CachedDataLoader(cache_dir=str(other)).cache_dir == other      # explicit argument wins
    monkeypatch.delenv('ANALYTICS_CACHE_DIR')
    assert str(cdl.CachedDataLoader().cache_dir).endswith('analytics/data/frontend_cache')   # default
    monkeypatch.setenv('ANALYTICS_CACHE_DIR', '')
    assert str(cdl.CachedDataLoader().cache_dir).endswith('analytics/data/frontend_cache')   # empty = unset
    cdl._read_cache_file.clear()


@pytest.mark.parametrize('value,expected', [(None, True), ('', True), ('0', False), ('1', True),
                                            ('true', True), ('YES', True), ('no', False), ('off', False),
                                            ('False', False)])
def test_turnover_flag_env_override(monkeypatch, value, expected):
    import importlib
    from page_apps.analytics import config
    if value is None:
        monkeypatch.delenv('ANALYTICS_SHOW_TURNOVER', raising=False)
    else:
        monkeypatch.setenv('ANALYTICS_SHOW_TURNOVER', value)
    try:
        assert importlib.reload(config).SHOW_TURNOVER_TAB is expected
    finally:
        monkeypatch.delenv('ANALYTICS_SHOW_TURNOVER', raising=False)
        importlib.reload(config)


def test_seller_support_copy_uses_constants():
    from analytics.processing import distinctiveness as d
    assert f"{d.MAX_SELLER_SHARE * 100:.0f}%" in common.SELLER_SUPPORT_NOTE
    assert f"at least {d.MIN_SELLERS} sellers" in common.SELLER_SUPPORT_NOTE
    assert "two thirds" not in common.SELLER_SUPPORT_NOTE


# ---- 2.3c: copy fixes -------------------------------------------------------------------------

def test_plural_helpers_and_no_naive_plurals_in_pages():
    import re
    from analytics.constants import unit_plural
    assert [unit_plural(t) for t in ('country', 'region', 'seller')] == ['countries', 'regions', 'sellers']
    assert [common.category_plural(c) for c in ('Country', 'Process Method', 'Varietal')] == \
        ['countries', 'process methods', 'varietals']
    bad = re.compile(r"\{[^{}]*(unit_type|entity_type|category_label[^{}]*)\}s\b")
    for path in (ROOT / 'page_apps').rglob('*.py'):
        assert not bad.search(path.read_text()), path


def test_explore_footnote_pluralizes_and_example_from_cache(rec, tmp_path, monkeypatch):
    import json
    from analytics.frontend import cached_data_loader as cdl
    from page_apps.analytics import flavor_origin_tab as fo
    from test_phase2_2c import build_cache
    cache = build_cache()
    (tmp_path / 'frontend_cache.json').write_text(json.dumps(cache))
    cdl._read_cache_file.clear()
    monkeypatch.setattr(cdl, '_loader', cdl.CachedDataLoader(cache_dir=str(tmp_path)))
    for unit_type, plural in (('country', 'countries'), ('region', 'regions'), ('seller', 'sellers')):
        rec['text'].clear()
        profile = cdl.load_unit_profile(*{'country': ('Kenya', 'country'), 'region': ('Kenya_Nyeri', 'region'),
                                          'seller': ('S1', 'seller')}[unit_type])
        fo.display_unit_profile(profile, 'genus')
        text = rec['all']()
        assert f"Only {plural} with at least" in text and f"{unit_type}s with at least" not in text.replace(plural, '')
    # intro uses the real top key finding (no made-up numbers)
    rec['text'].clear()
    fo.render_explore_section()
    top = cache['distinctiveness_meta']['key_findings'][0]['finding']
    assert f"For example: {top}." in rec['all']() and "Kenyan" not in rec['all']()
    # no key findings: generic wording without numbers
    cache['distinctiveness_meta']['key_findings'] = []
    (tmp_path / 'frontend_cache.json').write_text(json.dumps(cache))
    cdl._read_cache_file.clear()
    monkeypatch.setattr(cdl, '_loader', cdl.CachedDataLoader(cache_dir=str(tmp_path)))
    rec['text'].clear()
    fo.render_explore_section()
    assert "X% of a country's coffees" in rec['all']() and "For example" not in rec['all']()
    cdl._read_cache_file.clear()


def test_sidebar_units_label(rec, tmp_path, monkeypatch):
    import json
    from analytics.frontend import cached_data_loader as cdl
    from analytics.processing.distinctiveness import MIN_UNIT_COFFEES
    meta = {'generated_at': '2099-01-01T00:00:00', 'version': '2.0',
            'total_units': {'country': 31, 'region': 65, 'seller': 1}}
    (tmp_path / 'frontend_cache.json').write_text(json.dumps({'metadata': meta}))
    cdl._read_cache_file.clear()
    monkeypatch.setattr(cdl, '_loader', cdl.CachedDataLoader(cache_dir=str(tmp_path)))
    cdl.show_cache_status_widget()
    t = rec['all']()
    assert f"Profiles ({MIN_UNIT_COFFEES}+ coffees):** 31 countries · 65 regions · 1 seller" in t
    assert "Units cached" not in t
    cdl._read_cache_file.clear()


def test_pickers_default_to_largest_unit(point_loader_2c):
    from page_apps.analytics import flavor_origin_tab as fo
    from analytics.frontend import cached_data_loader as cdl
    units = cdl.get_available_units('country')
    assert units == sorted(units)                                   # list order stays alphabetical
    sizes = cdl.get_unit_sizes('country')
    biggest = max(sizes, key=lambda u: (sizes[u], ))
    assert units[fo.default_unit_index('country', units)] == biggest
    top2 = fo.top_units_by_size('country', units, 2)
    assert top2[0] == biggest and len(top2) == 2
    assert sizes[top2[0]] >= sizes[top2[1]] >= max(v for u, v in sizes.items() if u not in top2)
    assert fo.default_unit_index('country', []) == 0


@pytest.fixture
def point_loader_2c(tmp_path):
    import json
    from analytics.frontend import cached_data_loader as cdl
    from test_phase2_2c import build_cache
    (tmp_path / 'frontend_cache.json').write_text(json.dumps(build_cache()))
    cdl._read_cache_file.clear()
    cdl._loader = cdl.CachedDataLoader(cache_dir=str(tmp_path))
    yield
    cdl._loader = None
    cdl._read_cache_file.clear()
