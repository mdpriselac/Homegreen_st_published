import json

import numpy as np
import pandas as pd
import pytest

from analytics.constants import unit_label
from analytics.processing import distinctiveness as dist
from analytics.processing import distinctiveness_cache as dc
from analytics.processing.cooccurrence_analysis import FlavorCooccurrenceAnalyzer


def synthetic_cross_feature(seed=0):
    """Kenya (60 coffees, berry over-represented, 2 regions), Brazil/Peru/Ethiopia
    150 each, a tiny Fiji (4 coffees), and 20 coffees with missing country."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(country, sub, seller, fam, gen, spe):
        rows.append({'coffee_id': len(rows), 'country': country,
                     'region': f'{country}_{sub}' if country and sub else np.nan,
                     'seller': seller, 'flavor_families': fam,
                     'flavor_genera': gen, 'flavor_species': spe})

    for i in range(60):
        berry = i < 30
        add('Kenya', 'Nyeri' if i % 2 else 'Kiambu', 'S1' if i % 3 else 'S2',
            (['Fruity'] if berry else []) + (['Sweet'] if i % 2 == 0 else []),
            ['Berry'] if berry else [], ['Blueberry'] if berry and i < 20 else [])
    for c in ['Brazil', 'Peru', 'Ethiopia']:
        for i in range(150):
            fruity = i < 18
            add(c, 'Sub' + str(i % 3), 'S' + str(1 + i % 4),
                (['Fruity'] if fruity else []) + (['Sweet'] if i % 2 == 0 else []),
                ['Berry'] if fruity else [], [])
    for i in range(4):
        add('Fiji', 'X', 'S1', ['Fruity'], ['Berry'], [])
    for i in range(20):
        add(None, None, 'S1', ['Fruity'], [], [])
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def comps():
    cf = synthetic_cross_feature()
    return cf, dc.build_distinctiveness_components(dc.prepare_distinctiveness_input(cf))


def test_input_adapter_columns():
    df = dc.prepare_distinctiveness_input(synthetic_cross_feature())
    for col in ['coffee_id', 'country', 'region_key', 'seller',
                'flavors_family', 'flavors_genus', 'flavors_species']:
        assert col in df.columns


def test_profiles_schema_and_content(comps):
    cf, c = comps
    p = c['profiles']['country_Kenya']
    assert p['unit_type'] == 'country' and p['unit_name'] == 'Kenya' and p['label'] == 'Kenya'
    assert p['n_coffees'] == 60
    assert 'flavor_parse_rate' not in p['overview']
    assert p['overview']['n_regions'] == 2 and p['overview']['n_sellers'] == 2
    fam = p['levels']['family']
    assert fam['signature'] == ['Fruity'] and fam['n_distinctive'] == 1
    f = fam['flavors'][0]
    assert f['flavor'] == 'Fruity' and f['share_in'] == pytest.approx(0.5)
    assert f['lift'] > 1.5 and f['q'] < 0.05
    assert set(p['levels']) == {'family', 'genus', 'species'}
    assert set(p['family_shares']) == {'Fruity', 'Sweet'}
    assert 'country_Fiji' not in c['profiles']            # below MIN_UNIT_COFFEES
    assert 'country_None' not in c['profiles']


def test_region_label_and_keys(comps):
    _, c = comps
    key = 'region_Kenya_Nyeri'
    assert key in c['profiles']
    assert c['profiles'][key]['label'] == 'Kenya / Nyeri'
    assert unit_label('region', 'Costa Rica_Tarrazu') == 'Costa Rica / Tarrazu'
    assert unit_label('seller', 'A_B') == 'A_B'


def test_no_flavor_stands_out_not_padded(comps):
    _, c = comps
    brazil = c['profiles']['country_Brazil']['levels']['family']
    assert brazil['signature'] == [] and brazil['n_distinctive'] == 0


def test_rankings_per_type_sorted_with_min_n(comps):
    _, c = comps
    r = c['rankings']['distinctive_profile']['country']['family']
    scores = [x['score'] for x in r]
    assert scores == sorted(scores, reverse=True)
    assert r[0]['unit'] == 'Kenya'
    assert all(x['n_coffees'] >= 10 for x in r)
    assert {x['unit'] for x in c['rankings']['varied_profile']['country']['family']} == \
        {x['unit'] for x in r}
    assert set(c['rankings']['distinctive_profile']) == {'country', 'region', 'seller'}
    # region rows carry the "Country / Subregion" label
    reg = c['rankings']['varied_profile']['region']['family']
    assert all(' / ' in x['label'] for x in reg)


def test_by_flavor_columnar_and_consistent(comps):
    _, c = comps
    t = c['by_flavor']['country_family']
    cols = t['columns']
    rows = [dict(zip(cols, r)) for r in t['rows']]
    k = next(r for r in rows if r['unit'] == 'Kenya' and r['flavor'] == 'Fruity')
    assert (k['a'], k['b']) == (30, 30) and k['distinctive'] is True
    assert all(len(r) == len(cols) for r in t['rows'])


def test_key_findings_use_generator_keys(comps):
    _, c = comps
    kf = c['meta']['key_findings']
    assert kf and kf[0]['unit'] == 'Kenya'
    assert kf[0]['finding'].startswith('50% of coffees from Kenya list Berry')
    assert len({f['unit'] for f in kf}) == len(kf)


def test_meta_counts_and_params(comps):
    cf, c = comps
    m = c['meta']
    assert m['params']['min_lift'] == 1.5 and m['params']['min_unit_coffees'] == 10
    assert m['by_unit_type']['country']['n_coffees_missing'] == 20
    assert m['by_unit_type']['region']['n_coffees_missing'] == 20


def test_whole_output_is_json_safe(comps):
    _, c = comps
    json.dumps(c, allow_nan=False)


def test_pairwise_jsd_from_share_vectors(comps):
    _, c = comps
    k = c['profiles']['country_Kenya']['family_shares']
    b = c['profiles']['country_Brazil']['family_shares']
    p = c['profiles']['country_Peru']['family_shares']
    assert dist.jsd_between(k, k) == pytest.approx(0, abs=1e-12)
    assert dist.jsd_between(k, b) > dist.jsd_between(b, p) >= 0
    assert dist.jsd_between(k, b) == pytest.approx(dist.jsd_between(b, k))


def test_cooccurrence_conditional_by_flavor():
    rows = []
    # 100 coffees: A in 40, B in 30 (all with A), C in 50 (indep.)
    for i in range(100):
        fl = []
        if i < 40: fl.append('A')
        if i < 30: fl.append('B')
        if i % 2 == 0: fl.append('C')
        rows.append({'flavor_families': fl})
    az = FlavorCooccurrenceAnalyzer(pd.DataFrame(rows))
    by = az.conditional_by_flavor('family')
    assert set(by) == {'A', 'B', 'C'}                     # every flavor, not a global top-N
    b_given_a = next(x for x in by['A'] if x['flavor'] == 'B')
    assert b_given_a['p_b_given_a'] == pytest.approx(30 / 40)
    # base rate is over flavored coffees: 40 (A) + 30 (even, no A) = 70
    assert b_given_a['p_b'] == pytest.approx(30 / 70)
    assert b_given_a['ratio'] == pytest.approx((30 / 40) / (30 / 70))
    a_given_b = next(x for x in by['B'] if x['flavor'] == 'A')
    assert a_given_b['p_b_given_a'] == pytest.approx(1.0)
    assert 'by_flavor' in az.get_cooccurrence_summary('family')
    json.dumps(az.get_cooccurrence_summary('family')['by_flavor'])


def test_generator_wiring(tmp_path):
    from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator
    g = FrontendDataCacheGenerator(cache_dir=str(tmp_path))
    g.all_results = {'data': {'cross_feature_df': synthetic_cross_feature()}}
    g.distinctiveness = g._compute_distinctiveness()
    assert set(g.distinctiveness) == {'profiles', 'by_flavor', 'rankings', 'meta'}
    assert 'country_Kenya' in g.distinctiveness['profiles']
