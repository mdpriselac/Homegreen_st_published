import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.processing.stat_utils import (
    apply_bh, bh_adjust, cochran_ok, eta_squared_h, kaplan_meier,
    permutation_premium_test, validated_chi_square,
)
from analytics.processing.interaction_analysis import InteractionAnalyzer
from analytics.processing.cross_feature_analysis import CrossFeatureAnalyzer
from analytics.processing.turnover_analysis import SellerTurnoverAnalyzer
from analytics.processing.varietals import normalise_varietal


def test_jember_is_s795():
    assert normalise_varietal('Jember') == 'S795'
    assert normalise_varietal('Djember') == 'S795'


# ---- Benjamini-Hochberg --------------------------------------------------

def test_bh_matches_scipy_and_handles_missing():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert bh_adjust(p) == pytest.approx(list(stats.false_discovery_control(p, method='bh')))
    with_none = [0.01, None, 0.04, float('nan'), 0.03]
    q = bh_adjust(with_none)
    assert q[1] is None and q[3] is None
    assert [q[0], q[2], q[4]] == pytest.approx(
        list(stats.false_discovery_control([0.01, 0.04, 0.03], method='bh')))


def test_apply_bh_sets_q_and_significance_keeps_raw_p():
    rows = [{'p_value': 0.01}, {'p_value': 0.02}, {'p_value': 0.5}, {'p_value': None}]
    apply_bh(rows)
    assert rows[0]['p_value'] == 0.01
    assert rows[0]['q_value'] == pytest.approx(0.03)
    assert rows[0]['is_significant'] and rows[1]['is_significant']
    assert not rows[2]['is_significant']
    assert rows[3]['q_value'] is None and rows[3]['is_significant'] is False


# ---- effect size ----------------------------------------------------------

def test_eta_squared_h_formula():
    assert eta_squared_h(10.0, 100, 4) == pytest.approx((10 - 4 + 1) / (100 - 4))
    assert eta_squared_h(1.0, 3, 3) == 0.0


# ---- interaction permutation test ----------------------------------------

def _price_df(planted: bool, n_per=40, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for a, ea in (('A1', 0.0), ('A2', 3.0)):
        for b, eb in (('B1', 0.0), ('B2', 2.0)):
            for _ in range(n_per):
                price = 10 + ea + eb + rng.normal(0, 1.0)
                if planted and a == 'A1' and b == 'B1':
                    price += 6.0                         # extra beyond additivity
                rows.append({'country': a, 'process_type': b, 'price_per_lb': price,
                             'has_flavors': True, 'flavor_families': []})
    return pd.DataFrame(rows)


def _combo(res, a, b):
    return next(c for c in res['combinations']
                if c['feature_a_value'] == a and c['feature_b_value'] == b)


def test_planted_interaction_detected():
    res = InteractionAnalyzer(_price_df(True)).analyze_two_way_price_interactions(
        'country', 'process_type')
    c = _combo(res, 'A1', 'B1')
    assert c['is_significant'] and c['q_value'] < 0.05
    assert c['price_premium'] > 2


def test_additive_only_not_flagged_even_with_big_main_effects():
    res = InteractionAnalyzer(_price_df(False)).analyze_two_way_price_interactions(
        'country', 'process_type')
    assert not any(c['is_significant'] for c in res['combinations'])
    assert all(c['q_value'] is not None for c in res['combinations'])


def test_permutation_is_seeded_and_uses_2000_perms_min():
    df = _price_df(True)
    ma, mb = (df.country == 'A1').to_numpy(), (df.process_type == 'B1').to_numpy()
    y = df.price_per_lb.to_numpy()
    t1, p1 = permutation_premium_test(y, ma, mb, n_perm=2000, seed=3)
    t2, p2 = permutation_premium_test(y, ma, mb, n_perm=2000, seed=3)
    assert (t1, p1) == (t2, p2)
    assert p1 >= 1 / 2001            # resolution of a 2000-permutation test
    assert InteractionAnalyzer.N_PERMUTATIONS >= 2000


def test_observed_premium_matches_module_definition():
    df = _price_df(True)
    res = InteractionAnalyzer(df).analyze_two_way_price_interactions('country', 'process_type')
    c = _combo(res, 'A2', 'B2')
    ma, mb = (df.country == 'A2').to_numpy(), (df.process_type == 'B2').to_numpy()
    y = df.price_per_lb.to_numpy()
    obs, _ = permutation_premium_test(y, ma, mb, n_perm=50)
    assert obs == pytest.approx(c['price_premium'], abs=0.006)


# ---- chi-square guard -----------------------------------------------------

def test_cochran_rule():
    assert cochran_ok(np.full((2, 2), 6.0))
    assert not cochran_ok(np.array([[0.5, 20.0], [20.0, 20.0]]))      # one < 1
    assert not cochran_ok(np.array([[4.0, 4.0], [4.0, 40.0]]))        # < 80% >= 5


def test_valid_table_untouched():
    res = validated_chi_square(pd.DataFrame([[30, 10], [10, 30]]))
    assert res['status'] == 'ok' and res['has_data']


def test_sparse_table_collapses_rare_categories():
    t = pd.DataFrame([[40, 30, 1, 0], [30, 40, 0, 1], [35, 35, 1, 1], [1, 0, 2, 1], [0, 1, 1, 2]],
                     index=list('ABCDE'), columns=['w', 'x', 'y', 'z'])
    assert not cochran_ok(__import__('scipy.stats', fromlist=['x']).chi2_contingency(t)[3])
    res = validated_chi_square(t)
    assert res['status'] == 'collapsed' and res['has_data']
    assert set(res['collapsed']['rows']) == {'D', 'E'}
    assert set(res['collapsed']['cols']) == {'y', 'z'}
    assert res['p_value_method'] == 'asymptotic'


def test_still_sparse_after_collapse_uses_monte_carlo():
    t = pd.DataFrame([[3, 2, 2, 2, 2, 1], [2, 3, 1, 2, 2, 2], [2, 2, 3, 1, 2, 2],
                      [1, 2, 2, 3, 2, 2], [2, 1, 2, 2, 3, 2], [2, 2, 2, 2, 1, 3]])  # totals 12, expected 2
    res = validated_chi_square(t)
    assert res['status'] == 'monte_carlo' and res['has_data']
    assert res['p_value_method'] == 'monte_carlo' and res['n_simulations'] == 2000
    assert res['collapsed']['rows'] == [] and res['collapsed']['cols'] == []
    assert 0 < res['p_value'] <= 1 and res['cramers_v'] > 0
    assert validated_chi_square(t)['p_value'] == res['p_value']      # seeded


def test_monte_carlo_agrees_with_asymptotic_on_valid_table():
    from analytics.processing.stat_utils import monte_carlo_chi2_p
    t = pd.DataFrame([[40, 25, 20], [20, 35, 30], [25, 20, 45]])
    asym = stats.chi2_contingency(t)[1]
    mc = monte_carlo_chi2_p(t, n_sims=2000, seed=1)
    assert abs(mc - asym) < 0.02


def test_monte_carlo_detects_dependence_and_fixed_margins():
    t = pd.DataFrame([[9, 1, 0], [1, 9, 1], [0, 1, 9], [2, 2, 6]])
    res = validated_chi_square(t)
    assert res['status'] == 'monte_carlo' and res['p_value'] < 0.01


def test_insufficient_only_for_tiny_or_degenerate():
    assert validated_chi_square(pd.DataFrame([[10, 12]]))['status'] == 'insufficient'    # 1 row
    assert validated_chi_square(pd.DataFrame([[3, 2], [4, 1]]))['status'] == 'insufficient'  # n < 30


def test_hopeless_table_reports_insufficient_not_p():
    t = pd.DataFrame([[1, 0, 2], [0, 1, 1], [2, 1, 0]])
    res = validated_chi_square(t)
    assert res['has_data'] is False and res['status'] == 'insufficient'
    assert 'p_value' not in res


def test_cross_feature_reports_insufficient_for_sparse():
    rows = [{'country': f'C{i}', 'region': np.nan, 'process_type': 'Washed' if i % 2 else 'Natural',
             'varietals': [], 'price_per_lb': 5.0, 'has_flavors': False,
             'flavor_families': [], 'flavor_genera': [], 'flavor_species': []}
            for i in range(12)]
    res = CrossFeatureAnalyzer(pd.DataFrame(rows)).analyze_process_by_origin(min_coffees=1)
    assert res['by_country']['chi_square']['has_data'] is False


# ---- Kaplan-Meier ---------------------------------------------------------

def test_km_hand_computed_with_censoring():
    # times 2(event) 3(event) 3(censored) 5(event) 8(censored)
    km = kaplan_meier([2, 3, 3, 5, 8], [1, 1, 0, 1, 0])
    # S(2)=4/5=.8 ; S(3)=.8*(3/4)... at risk at 3 is 4 (3,3,5,8), 1 event -> .6 ; S(5)=.6*(1/2)=.3
    assert km['times'] == [2.0, 3.0, 5.0]
    assert km['survival'] == pytest.approx([0.8, 0.6, 0.3])
    assert km['median'] == 5.0
    assert km['q25'] == 3.0          # first time S <= 0.75
    ecdf = stats.ecdf(stats.CensoredData(uncensored=[2, 3, 5], right=[3, 8]))
    assert ecdf.sf.evaluate(km['times']) == pytest.approx(km['survival'])   # agrees with scipy


def test_km_median_undefined_when_survival_stays_high():
    km = kaplan_meier([5, 10, 20, 30], [1, 0, 0, 0])
    assert km['median'] is None


def test_km_ignoring_censoring_would_differ():
    km = kaplan_meier([1, 2, 3, 100, 100, 100], [1, 1, 1, 0, 0, 0])
    assert km['median'] == 3.0       # S=0.5 after 3 events of 6
    assert float(np.median([1, 2, 3])) == 2.0   # naive expired-only median is lower


# ---- turnover censoring ---------------------------------------------------

def _coffee(i, seller, first, last, active, country='Kenya', process='Washed', price=5.0):
    return {'coffee_id': i, 'seller': seller, 'first_observed': first, 'last_observed': last,
            'is_active': active, 'country': country, 'process_type': process,
            'price_per_lb': price}


def _turnover_df():
    rows = [
        _coffee(1, 'S1', '2024-03-20', '2025-08-01', False),    # initial inventory -> excluded
        _coffee(2, 'S1', '2024-06-01', '2025-01-01', False),    # expired pre-tracking -> excluded
        _coffee(3, 'S1', '2025-07-01', '2025-07-11', False),    # eligible, event, 10 d
        _coffee(4, 'S1', '2025-07-01', '2025-07-21', False),    # eligible, event, 20 d
        _coffee(5, 'S1', '2025-07-01', '2025-09-29', True),     # eligible, censored, 90 d
        _coffee(6, 'S1', '2025-08-05', '2025-09-30', True),     # eligible, censored, 56 d
        _coffee(7, 'Dead', '2025-06-20', '2025-07-10', False),  # seller gone: last seen >60d before end
        _coffee(8, 'Dead', '2025-06-20', '2025-07-01', False),
        _coffee(9, 'S1', '2025-02-01', None, True),             # missing date -> excluded
    ]
    return pd.DataFrame(rows)


def test_censoring_exclusions_and_window():
    t = SellerTurnoverAnalyzer(_turnover_df())
    ex = t.observation_window()['exclusions']
    assert ex['left_censored_initial_inventory'] == 1
    assert ex['expired_before_tracking'] == 1
    assert ex['invalid_or_missing_dates'] == 1
    assert ex['eligible'] == 6 and ex['eligible_expired'] == 4 and ex['eligible_active_censored'] == 2
    w = t.observation_window()
    assert w['window_end'] == '2025-09-30' and w['initial_inventory_cutoff'] == '2024-04-05'


def test_overview_uses_km_not_expired_only_median():
    t = SellerTurnoverAnalyzer(_turnover_df())
    o = t.compute_coffee_lifespans()
    # scrape interval I = 9.5 d (median gap of the synthetic scrape dates). Durations
    # (span + I for expired, span + I/2 for active), sorted:
    #   19.5e 20.5e 29.5e 29.5e 60.75c 94.75c
    # S(19.5)=5/6, S(20.5)=4/6, S(29.5)=4/6*(2/4)=1/3 -> median 29.5
    assert o['median_days'] == 29.5
    assert o['total_expired'] == 4 and o['total_active'] == 2


def test_out_of_business_sellers_excluded_from_turnover():
    t = SellerTurnoverAnalyzer(_turnover_df())
    assert t._inactive_sellers() == ['Dead']
    by_seller = t.analyze_lifespan_by_seller()
    assert by_seller['excluded_inactive_sellers'] == ['Dead']
    assert all(s['seller'] != 'Dead' for s in by_seller['sellers'])
    assert all(s['seller'] != 'Dead' for s in t.get_seller_turnover_summary())


def test_new_listings_exclude_first_scrape_month_and_inventory():
    df = _turnover_df()
    df.loc[len(df)] = _coffee(10, 'S1', '2024-03-25', '2025-07-30', True)   # first month
    df.loc[len(df)] = _coffee(11, 'S1', '2024-04-02', '2025-07-30', True)   # <= cutoff
    sp = SellerTurnoverAnalyzer(df).compute_seasonal_patterns()
    months = {a['month']: a['count'] for a in sp['appearances']}
    assert '2024-03' not in months and '2024-04' not in months
    assert months['2025-07'] == 3 and months['2025-08'] == 1 and months['2025-06'] == 2
    assert sp['excluded_first_scrape_month'] == '2024-03'
    # removals only counted when expiry is reliably tracked (>= 2025-06-18)
    removed = {d['month']: d['count'] for d in sp['disappearances']}
    assert '2025-01' not in removed


# ---- 4.2: region aliases, lifespan floor ----------------------------------

def test_region_aliases_merge_known_variants():
    from analytics.processing.data_hygiene import make_region_key, normalise_subregion
    assert make_region_key('Brazil', 'Sao Paolo') == 'Brazil_Sao Paulo'
    assert make_region_key('Brazil', 'sao paolo') == make_region_key('Brazil', 'Sao Paulo')
    assert normalise_subregion('Brazil', 'Cerrado, Sao Paolo') == 'Cerrado, Sao Paulo'
    assert make_region_key('Brazil', 'Sul de Minas') == make_region_key('Brazil', 'Sul De Minas')
    assert make_region_key('Papua New Guinea', 'Eastern highlands') == 'Papua New Guinea_Eastern Highlands'
    # aliases are country-scoped; unrelated values untouched
    assert make_region_key('Kenya', 'Sul de Minas') == 'Kenya_Sul de Minas'
    assert make_region_key('Brazil', 'Cerrado') == 'Brazil_Cerrado'
    assert pd.isna(make_region_key('Brazil', 'UNKNOWN'))


def test_scrape_grid_duration_definition():
    rows = []
    # weekly scrape cadence after tracking start
    for i, day in enumerate(pd.date_range('2025-07-01', periods=8, freq='7D')):
        rows.append(_coffee(100 + i, 'S1', day, day, False))                   # seen once, expired
    rows.append(_coffee(200, 'S1', '2025-07-01', '2025-08-26', True))          # active, 56 d span
    t = SellerTurnoverAnalyzer(pd.DataFrame(rows))
    w = t.observation_window()
    assert w['scrape_interval_days'] == 7.0
    assert 'duration_definition' in w and 'min_duration_floor_days' not in w
    assert 'floored_short_lifespans' not in w['exclusions']
    sdf = t.survival_df
    assert (sdf.loc[sdf['event'], 'duration'] == 7.0).all()                    # 0 + I
    assert sdf.loc[~sdf['event'], 'duration'].iloc[0] == 56 + 3.5              # span + I/2
    assert (t.df['duration_days'].dropna() >= 0).all()                         # raw column stays raw
    assert t.df.loc[t.df['coffee_id'] == 200, 'duration_days'].iloc[0] == 56
    assert t.compute_coffee_lifespans()['median_days'] == 7.0                  # "seen once"


def test_km_hand_computed_with_scrape_grid_durations():
    # I = 7 (weekly dates). span/state: 7d expired, 14d expired, 14d active, 28d expired
    # durations: 14e, 21e, 17.5c, 35e -> sorted 14e 17.5c 21e 35e
    # S(14)=3/4=.75 ; at 17.5 censored; S(21)=.75*(1/2)=.375 -> median 21
    rows = [
        _coffee(1, 'S1', '2025-07-01', '2025-07-08', False),
        _coffee(2, 'S1', '2025-07-01', '2025-07-15', False),
        _coffee(3, 'S1', '2025-07-01', '2025-07-15', True),
        _coffee(4, 'S1', '2025-07-01', '2025-07-29', False),
        _coffee(5, 'S1', '2025-07-22', '2025-07-22', True),   # extra dates to keep 7-day cadence
    ]
    df = pd.DataFrame(rows)
    t = SellerTurnoverAnalyzer(df)
    assert t.scrape_interval_days == 7.0
    km = kaplan_meier(*(t.survival_df.loc[t.survival_df.coffee_id.isin([1, 2, 3, 4]), c].to_numpy()
                        for c in ('duration', 'event')))
    assert sorted(km['times']) == [14.0, 21.0, 35.0]
    assert km['survival'] == pytest.approx([0.75, 0.375, 0.0])
    assert km['median'] == 21.0


def test_flavor_counts_are_exact_integers():
    assert int((57 / 100) * 100) == 56          # the old derivation rounded down
    rows = [{'country': 'A', 'process_type': 'W', 'price_per_lb': 5.0, 'has_flavors': True,
             'flavor_families': ['Fruity'] if i < 57 else ['Sweet']} for i in range(100)]
    df = pd.DataFrame(rows)
    ia = InteractionAnalyzer(df)
    counts, n = ia._compute_group_flavor_counts(df, 'flavor_families')
    assert n == 100 and counts == {'Fruity': 57, 'Sweet': 43}
    res = ia.analyze_two_way_flavor_interactions('country', 'process_type')
    prof = list(res['combination_profiles'].values())[0]
    assert {f['flavor']: f['count'] for f in prof['flavors']} == {'Fruity': 57, 'Sweet': 43}
