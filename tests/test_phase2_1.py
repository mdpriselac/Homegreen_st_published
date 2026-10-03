import json
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics.processing import distinctiveness as dist


def mk(rows):
    df = pd.DataFrame(rows)
    for lvl in dist.LEVELS:
        col = f'flavors_{lvl}'
        if col not in df:
            df[col] = [[] for _ in range(len(df))]
    for col in ['country', 'region_key']:
        if col not in df:
            df[col] = 'x'
    if 'seller' not in df:
        df['seller'] = [f'S{i % 4}' for i in range(len(df))]   # several sellers by default
    df['coffee_id'] = range(len(df))
    return df


def planted():
    """Kenya: 50 coffees, 24 berry (48%). Others: 450 coffees, 54 berry (12%).
    'common' is listed by exactly half of everything."""
    rows = []
    for i in range(50):
        fl = (['berry'] if i < 24 else []) + (['common'] if i % 2 == 0 else [])
        rows.append({'country': 'Kenya', 'flavors_family': fl})
    for o, cname in enumerate(['Brazil', 'Peru', 'Ethiopia']):
        for i in range(150):
            fl = (['berry'] if i < 18 else []) + (['common'] if i % 2 == 0 else [])
            rows.append({'country': cname, 'flavors_family': fl})
    return mk(rows)


def test_planted_flavor_found_with_correct_numbers():
    fl, un = dist.compute_distinctiveness(planted(), 'country', 'family')
    r = fl[(fl.unit == 'Kenya') & (fl.flavor == 'berry')].iloc[0]
    assert (r.a, r.b, r.c, r.d) == (24, 26, 54, 396)
    assert r.share_in == pytest.approx(0.48) and r.share_rest == pytest.approx(0.12)
    assert r.lift == pytest.approx(4.0)
    assert r.q < 0.05 and r.distinctive
    # matches scipy's Fisher exact (one-sided)
    assert r.p == pytest.approx(stats.fisher_exact([[24, 26], [54, 396]], alternative='greater')[1])
    assert r.log_odds > 0 and r.log_odds_se > 0
    u = un[un.unit == 'Kenya'].iloc[0]
    assert u.signature == ['berry'] and u.n_distinctive_flavors == 1


def test_significant_but_small_lift_not_distinctive():
    # Big: 1200 coffees, 36% list 'x'; rest 30% -> lift 1.2 but highly significant
    rows = [{'country': 'Big', 'flavors_family': ['x'] if i < 432 else []} for i in range(1200)]
    rows += [{'country': 'Other', 'flavors_family': ['x'] if i < 360 else []} for i in range(1200)]
    fl, un = dist.compute_distinctiveness(mk(rows), 'country', 'family')
    r = fl[(fl.unit == 'Big') & (fl.flavor == 'x')].iloc[0]
    assert r.q < 0.05 and r.lift == pytest.approx(1.2)
    assert not r.distinctive
    assert un.set_index('unit').loc['Big', 'signature'] == []
    fl2, _ = dist.compute_distinctiveness(mk(rows), 'country', 'family', min_lift=1.1)
    assert fl2[(fl2.unit == 'Big') & (fl2.flavor == 'x')].iloc[0].distinctive


def test_undefined_lift_counts_as_passing():
    rows = [{'country': 'A', 'flavors_family': ['solo'] if i < 8 else []} for i in range(20)]
    rows += [{'country': 'B', 'flavors_family': []} for i in range(100)]
    fl, un = dist.compute_distinctiveness(mk(rows), 'country', 'family')
    r = fl[(fl.unit == 'A') & (fl.flavor == 'solo')].iloc[0]
    assert np.isnan(r.lift) and r.share_rest == 0 and r.distinctive


def test_common_flavor_not_distinctive_and_no_padding():
    fl, un = dist.compute_distinctiveness(planted(), 'country', 'family')
    common = fl[fl.flavor == 'common']
    assert not common.distinctive.any()
    brazil = un[un.unit == 'Brazil'].iloc[0]
    assert brazil.signature == []           # "no flavor stands out", never padded


def test_small_units_excluded_but_kept_in_rest():
    df = planted()
    extra = mk([{'country': 'Tiny', 'flavors_family': ['berry']} for _ in range(9)])
    both = pd.concat([df, extra], ignore_index=True)
    fl, un = dist.compute_distinctiveness(both, 'country', 'family')
    assert 'Tiny' not in set(un.unit) and 'Tiny' not in set(fl.unit)
    k = fl[(fl.unit == 'Kenya') & (fl.flavor == 'berry')].iloc[0]
    assert k.c == 54 + 9                    # tiny unit's coffees are still in the rest


def test_min_flavor_coffees():
    rows = [{'country': 'A', 'flavors_family': ['rare'] if i < 2 else []} for i in range(20)]
    rows += [{'country': 'B', 'flavors_family': []} for i in range(20)]
    fl, _ = dist.compute_distinctiveness(mk(rows), 'country', 'family')
    assert fl.empty                          # a=2 < MIN_FLAVOR_COFFEES


def test_bh_within_family_of_tests():
    rng = np.random.default_rng(0)
    rows = [{'country': f'C{i % 6}', 'region_key': f'R{i % 4}',
             'flavors_family': [f'f{j}' for j in range(8) if rng.random() < 0.3]}
            for i in range(600)]
    df = mk(rows)
    fl, _ = dist.compute_distinctiveness(df, 'country', 'family')
    assert np.allclose(fl.q, stats.false_discovery_control(fl.p, method='bh'))
    fr, _ = dist.compute_distinctiveness(df, 'region', 'family')
    assert np.allclose(fr.q, stats.false_discovery_control(fr.p, method='bh'))  # separate family
    assert (fl.q >= fl.p - 1e-12).all()


def test_missing_unit_values_excluded_entirely():
    df = planted()
    missing = mk([{'country': None, 'flavors_family': ['berry']} for _ in range(100)] +
                 [{'country': np.nan, 'flavors_family': ['berry']} for _ in range(20)] +
                 [{'country': '', 'flavors_family': ['berry']} for _ in range(5)])
    fl, _ = dist.compute_distinctiveness(pd.concat([df, missing], ignore_index=True), 'country', 'family')
    k = fl[(fl.unit == 'Kenya') & (fl.flavor == 'berry')].iloc[0]
    assert (k.a, k.b, k.c, k.d) == (24, 26, 54, 396)   # unchanged by missing rows


def test_jsd_identical_profile_near_zero_and_shrinkage():
    # units with the same flavor pattern as the market -> JSD ~ 0
    rows = []
    for u in ['A', 'B', 'C']:
        for i in range(100):
            fl = (['p'] if i % 2 == 0 else []) + (['q'] if i % 5 == 0 else [])
            rows.append({'country': u, 'flavors_family': fl})
    _, un = dist.compute_distinctiveness(mk(rows), 'country', 'family')
    assert un.jsd.max() < 1e-3

    # extreme unit: all coffees list 'x', market rarely does. 10 vs 200 coffees, same shares
    base = [{'country': 'Base', 'flavors_family': ['x'] if i < 50 else ['y']} for i in range(1000)]
    small = [{'country': 'Small', 'flavors_family': ['x']} for _ in range(10)]
    large = [{'country': 'Large', 'flavors_family': ['x']} for _ in range(200)]
    _, un = dist.compute_distinctiveness(mk(base + small + large), 'country', 'family')
    j = un.set_index('unit').jsd
    assert 0 < j['Small'] < j['Large'] <= 1
    # shrinkage lowers the small unit's divergence much more than the large one's
    _, un0 = dist.compute_distinctiveness(mk(base + small + large), 'country', 'family', alpha=0)
    j0 = un0.set_index('unit').jsd
    assert (j0['Small'] - j['Small']) > 3 * (j0['Large'] - j['Large'])


def test_diversity_range_and_ordering():
    rows = []
    for i in range(100):
        rows.append({'country': 'Narrow', 'flavors_family': ['a']})
        rows.append({'country': 'Broad', 'flavors_family': [['a', 'b', 'c', 'd'][i % 4]]})
    _, un = dist.compute_distinctiveness(mk(rows), 'country', 'family')
    d = un.set_index('unit').diversity
    assert 0 <= d['Narrow'] < d['Broad'] <= 1


def test_duplicates_within_coffee_counted_once():
    rows = [{'country': 'A', 'flavors_family': ['x', 'x', 'x']} for _ in range(10)]
    rows += [{'country': 'B', 'flavors_family': []} for _ in range(10)]
    fl, _ = dist.compute_distinctiveness(mk(rows), 'country', 'family')
    r = fl[(fl.unit == 'A') & (fl.flavor == 'x')].iloc[0]
    assert (r.a, r.b) == (10, 0)


def test_records_json_friendly_and_sentence():
    fl, un = dist.compute_distinctiveness(planted(), 'country', 'family')
    s = json.dumps({'f': dist.to_records(fl), 'u': dist.to_records(un)}, allow_nan=False)
    assert 'NaN' not in s
    row = dist.to_records(fl[(fl.unit == 'Kenya') & (fl.flavor == 'berry')])[0]
    assert dist.format_distinctive_sentence(row) == \
        "48% of coffees from Kenya list berry vs 12% of other coffees (4.0x as often)"
    row['lift'] = None
    assert 'no other coffee lists it' in dist.format_distinctive_sentence(row, 'seller')


def test_performance_3000_coffees():
    rng = np.random.default_rng(3)
    n = 3000
    fam = [f'F{i}' for i in range(12)]
    gen = [f'G{i}' for i in range(60)]
    spe = [f'S{i}' for i in range(250)]
    pick = lambda vocab, k: list(rng.choice(vocab, size=rng.integers(0, k), replace=False))
    df = pd.DataFrame({
        'coffee_id': range(n),
        'country': rng.choice([f'C{i}' for i in range(30)], n),
        'region_key': rng.choice([f'R{i}' for i in range(400)], n),
        'seller': rng.choice([f'S{i}' for i in range(25)], n),
        'flavors_family': [pick(fam, 6) for _ in range(n)],
        'flavors_genus': [pick(gen, 8) for _ in range(n)],
        'flavors_species': [pick(spe, 10) for _ in range(n)],
    })
    t = time.perf_counter()
    out = dist.compute_all(df)
    dt = time.perf_counter() - t
    print(f"compute_all on 3000 coffees: {dt:.2f}s")
    assert len(out) == 9 and dt < 30


# ---- 2.2b' additions -------------------------------------------------------

def test_seller_rule_removes_single_seller_flavor_for_origins_only():
    rows = []
    # Haiti: 'burnt' listed by 10 coffees, all from seller Burman; 30 coffees total
    for i in range(30):
        rows.append({'country': 'Haiti', 'seller': 'Burman' if i < 15 else 'Other1',
                     'flavors_genus': ['Burnt'] if i < 10 else []})
    # market: 300 coffees from several sellers, no 'Burnt'
    for i in range(300):
        rows.append({'country': 'Brazil', 'seller': f'S{i % 5}', 'flavors_genus': []})
    df = mk(rows)
    fl, un = dist.compute_distinctiveness(df, 'country', 'genus')
    r = fl[(fl.unit == 'Haiti') & (fl.flavor == 'Burnt')].iloc[0]
    assert r.q < 0.05 and r.lift != r.lift or r.lift >= 1.5      # statistically distinctive ...
    assert r.n_sellers_supporting == 1 and r.top_seller == 'Burman' and r.top_seller_share == 1.0
    assert not r.seller_supported and not r.distinctive          # ... but one seller's vocabulary
    assert un.set_index('unit').loc['Haiti', 'signature'] == []
    # same data viewed as seller units: rule not applied
    df2 = df.copy(); df2['seller'] = df2['country'].map({'Haiti': 'SH', 'Brazil': 'SB'})
    fs, _ = dist.compute_distinctiveness(df2, 'seller', 'genus')
    rs = fs[(fs.unit == 'SH') & (fs.flavor == 'Burnt')].iloc[0]
    assert rs.distinctive and rs.seller_supported and np.isnan(rs.n_sellers_supporting)


def test_seller_rule_thresholds():
    def build(split):                      # split = coffees listing the flavor per seller
        rows = []
        for sname, n in split.items():
            rows += [{'country': 'T', 'seller': sname, 'flavors_genus': ['X']} for _ in range(n)]
        rows += [{'country': 'T', 'seller': 'Z', 'flavors_genus': []} for _ in range(20)]
        rows += [{'country': 'M', 'seller': f'M{i % 5}', 'flavors_genus': []} for i in range(300)]
        fl, _ = dist.compute_distinctiveness(mk(rows), 'country', 'genus')
        return fl[(fl.unit == 'T') & (fl.flavor == 'X')].iloc[0]
    assert dist.MAX_SELLER_SHARE == 0.75
    assert build({'A': 8, 'B': 4}).distinctive          # 0.667 passes
    assert build({'A': 9, 'B': 3}).distinctive          # exactly 0.75 (9 of 12) passes
    assert not build({'A': 10, 'B': 2}).distinctive     # 0.833 > 0.75 fails
    assert build({'A': 6, 'B': 6}).n_sellers_supporting == 2


def test_missing_seller_counts_as_one_unknown_seller():
    rows = [{'country': 'T', 'seller': None, 'flavors_genus': ['X']} for _ in range(10)]
    rows += [{'country': 'M', 'seller': f'M{i % 5}', 'flavors_genus': []} for i in range(300)]
    fl, _ = dist.compute_distinctiveness(mk(rows), 'country', 'genus')
    r = fl[(fl.unit == 'T')].iloc[0]
    assert r.n_sellers_supporting == 1 and r.top_seller is None and not r.distinctive


def test_other_family_excluded_but_genus_species_kept():
    rows = [{'country': 'A', 'flavors_family': ['Other', 'Fruity'],
             'flavors_genus': ['Papery/Musty']} for _ in range(20)]
    rows += [{'country': 'B', 'flavors_family': ['Fruity'], 'flavors_genus': []} for _ in range(200)]
    df = mk(rows)
    fam, _ = dist.compute_distinctiveness(df, 'country', 'family')
    assert 'Other' not in set(fam.flavor)
    gen, _ = dist.compute_distinctiveness(df, 'country', 'genus')
    assert 'Papery/Musty' in set(gen.flavor)
    assert dist.EXCLUDED_FLAVORS == {'family': {'Other'}}
    assert 'Other' not in dist.unit_share_vectors(df, 'country', 'family')['A']


def test_notes_per_coffee_style_does_not_move_jsd():
    # Same flavor proportions (x:y = 3:1), but unit W writes 4 notes per coffee and V writes 1.
    rows = []
    for i in range(100):
        rows.append({'country': 'W', 'flavors_family': ['x', 'x2', 'x3', 'y']})   # x-ish x3, y x1
        rows.append({'country': 'V', 'flavors_family': [['x', 'x2', 'x3', 'y'][i % 4]]})
    for i in range(400):
        rows.append({'country': 'Mkt', 'flavors_family': [['x', 'x2', 'x3', 'y'][i % 4]]})
    df = mk(rows)
    vec = dist.unit_share_vectors(df, 'country', 'family')
    assert dist.jsd_between(vec['W'], vec['V']) < 1e-3
    _, un = dist.compute_distinctiveness(df, 'country', 'family')
    j = un.set_index('unit').jsd
    assert j['W'] < 1e-3 and j['V'] < 1e-3
