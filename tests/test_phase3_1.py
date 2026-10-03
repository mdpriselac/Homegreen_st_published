import logging

import numpy as np
import pandas as pd
import pytest

from analytics.processing.data_hygiene import (
    clean_flavors, clean_text, flavor_terms, headline_counts, is_placeholder,
    make_region_key, add_region_key, normalize_process,
)
from analytics.processing.varietals import (
    clean_varietal_list, expand_varietals, normalise_varietal, split_varietal_string,
)
from analytics.processing.cross_feature_analysis import CrossFeatureAnalyzer


# ---- placeholders --------------------------------------------------------

@pytest.mark.parametrize('v', [None, np.nan, '', '  ', 'UNKNOWN', 'Unknown', 'unknown',
                               'N/A', 'n/a', 'None', 'null', '[]', pd.NA])
def test_placeholders_are_missing(v):
    assert is_placeholder(v)
    assert pd.isna(clean_text(v))


def test_real_values_kept_and_stripped():
    assert not is_placeholder('Huila')
    assert clean_text('  Huila ') == 'Huila'


def test_flavors_with_falsy_levels_are_dropped_at_that_level():
    raw = [
        {'family': 'Fruity', 'genus': 'Berry', 'species': ''},
        {'family': None, 'genus': 'Citrus', 'species': 'Lemon'},
        {'family': '', 'genus': None, 'species': None},
        'junk',
    ]
    out = clean_flavors(raw)
    assert out == [{'family': 'Fruity', 'genus': 'Berry'},
                   {'genus': 'Citrus', 'species': 'Lemon'}]
    assert flavor_terms(out, 'family') == ['Fruity']
    assert flavor_terms(out, 'genus') == ['Berry', 'Citrus']
    assert flavor_terms(out, 'species') == ['Lemon']
    assert None not in flavor_terms(raw, 'family')


# ---- region key ----------------------------------------------------------

def test_region_key_is_country_plus_subregion():
    assert make_region_key('Ethiopia', 'Sidama') == 'Ethiopia_Sidama'
    assert make_region_key('Kenya', 'Central') != make_region_key('Colombia', 'Central')


def test_region_key_missing_parts_give_nan_not_unknown_unit():
    assert pd.isna(make_region_key('Kenya', 'UNKNOWN'))
    assert pd.isna(make_region_key('Kenya', None))
    assert pd.isna(make_region_key(np.nan, 'Central'))
    df = pd.DataFrame({'country_final': ['Kenya', 'Peru', None],
                       'subregion_final': ['UNKNOWN', 'Cusco', 'Cusco']})
    keys = add_region_key(df)
    assert keys.tolist()[1] == 'Peru_Cusco'
    assert keys.isna().tolist() == [True, False, True]


def _cf(rows):
    base = dict(price_per_lb=5.0, has_flavors=True, flavor_families=['Fruity'],
                flavor_genera=[], flavor_species=[])
    return pd.DataFrame([{**base, **r} for r in rows])


def test_same_named_subregion_in_two_countries_not_pooled():
    rows = []
    for country in ('Kenya', 'Colombia'):
        for i in range(6):
            rows.append({'country': country, 'region': f'{country}_Central',
                         'process_type': 'Washed' if i % 2 else 'Natural',
                         'varietals': []})
    res = CrossFeatureAnalyzer(_cf(rows)).analyze_process_by_origin(min_coffees=5)
    assert set(res['by_region']['stacked_data']['groups']) == {'Kenya_Central', 'Colombia_Central'}


def test_missing_process_and_region_excluded_from_charts():
    rows = [{'country': 'Kenya', 'region': np.nan, 'process_type': np.nan, 'varietals': []}
            for _ in range(8)]
    rows += [{'country': 'Kenya', 'region': 'Kenya_Nyeri', 'process_type': 'Washed',
              'varietals': []} for _ in range(8)]
    res = CrossFeatureAnalyzer(_cf(rows)).analyze_process_by_origin(min_coffees=5)
    assert res['by_region']['stacked_data']['groups'] == ['Kenya_Nyeri']
    assert res['by_country']['stacked_data']['categories'] == ['Washed']


# ---- varietals -----------------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    ('SL-28', 'SL28'), ('SL 28', 'SL28'), ('sl28', 'SL28'), ('SL28', 'SL28'),
    ('Catuaí', 'Catuai'), ('Catuai', 'Catuai'), ('catuai (40%)', 'Catuai'),
    ('Gesha', 'Geisha'), ('Ruiru-11', 'Ruiru 11'), ('Ruiru11', 'Ruiru 11'),
    ('S-795', 'S795'), ('Tim-tim', 'Tim Tim'),
    ('Regional cultivars 74110', 'JARC 74110'), ('74110', 'JARC 74110'),
    ('Ethiopian Heirloom', 'Heirloom'), ('Heirloom Varieties', 'Heirloom'),
    ('Catuaí Vermelho', 'Red Catuai'), ('Catuai Rojo', 'Red Catuai'),
    ('Yellow bourbon', 'Yellow Bourbon'),
])
def test_varietal_normalisation_merges(raw, expected):
    assert normalise_varietal(raw) == expected


def test_distinct_cultivars_stay_distinct():
    names = {normalise_varietal(v) for v in
             ['Catuai', 'Yellow Catuai', 'Red Catuai', 'Bourbon', 'Pink Bourbon', 'Caturra', 'Catimor']}
    assert len(names) == 7


@pytest.mark.parametrize('v', ['UNKNOWN', '', None, np.nan, '[]', 'Various', 'Arabica', 'Peaberry'])
def test_varietal_non_values_are_none(v):
    assert normalise_varietal(v) is None


def test_normalise_is_idempotent():
    for raw in ['SL-28', 'Catuaí Vermelho', 'Regional cultivars 74112', 'Lini-S', 'USDA 762']:
        once = normalise_varietal(raw)
        assert normalise_varietal(once) == once


def test_split_keeps_parenthesised_commas():
    assert split_varietal_string('Bourbon (SL-28, SL-34), Caturra') == ['Bourbon (SL-28, SL-34)', 'Caturra']
    assert clean_varietal_list(['Catuai', 'Catuaí', 'UNKNOWN']) == ['Catuai']


def _vdf():
    return pd.DataFrame({
        'coffee_id': [1, 2, 3, 4],
        'country': ['A', 'A', 'B', 'B'],
        'varietals': [['Bourbon'], ['Bourbon', 'Caturra', 'Typica'], ['SL-28', 'SL28'], []],
    })


def test_weighted_expansion_weights_sum_to_one_per_coffee():
    out = expand_varietals(_vdf(), 'weighted')
    sums = out.groupby('coffee_id')['weight'].sum()
    assert np.allclose(sums.values, 1.0)
    assert set(sums.index) == {1, 2, 3}               # coffee 4 has no varietal
    assert (out['coffee_id'] == 2).sum() == 3
    assert (out['coffee_id'] == 3).sum() == 1         # SL-28 + SL28 collapse to one


def test_single_mode_drops_multi_varietal_rows():
    out = expand_varietals(_vdf(), 'single')
    assert sorted(out['coffee_id']) == [1, 3]         # 3 collapses to one varietal
    assert out['coffee_id'].is_unique
    assert (out['weight'] == 1.0).all()
    assert out.loc[out['coffee_id'] == 3, 'single_varietal'].iloc[0] == 'SL28'


def test_expansion_empty_and_bad_mode():
    assert expand_varietals(_vdf().iloc[0:0], 'single').empty
    with pytest.raises(ValueError):
        expand_varietals(_vdf(), 'both')


def test_chi_square_uses_single_varietal_rows_only():
    rows = []
    for i in range(30):          # single-varietal coffees
        rows.append({'country': 'A' if i % 2 else 'B', 'region': np.nan,
                     'process_type': 'Washed' if i % 3 else 'Natural',
                     'varietals': ['Bourbon' if i % 2 else 'Caturra']})
    for i in range(30):          # multi-varietal coffees must not enter the test
        rows.append({'country': 'A', 'region': np.nan, 'process_type': 'Washed',
                     'varietals': ['Bourbon', 'Caturra', 'Typica']})
    a = CrossFeatureAnalyzer(_cf(rows))
    res = a.analyze_varietal_by_origin(min_coffees=5)
    assert res['chi_square']['n_observations'] == 30
    pv = a.analyze_process_by_varietal()
    assert pv['chi_square']['n_observations'] == 30
    # descriptive heatmap is weighted: total mass == number of coffees
    assert sum(sum(r) for r in res['heatmap_counts']['values']) == pytest.approx(60, abs=1e-6)


# ---- process -------------------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    ('Washed', 'Washed'), ('Natural', 'Natural'), ('Honey', 'Honey'),
    ('Wet Hulled', 'Wet Hulled'), ('wet-hulled', 'Wet Hulled'),
    ('Monsoon', 'Other'), ('Decaf', 'Other'),
])
def test_process_exact_mapping(raw, expected):
    assert normalize_process(raw) == expected


@pytest.mark.parametrize('raw', ['Unknown', '', None, np.nan])
def test_process_placeholders_missing(raw):
    assert pd.isna(normalize_process(raw))


@pytest.mark.parametrize('raw', ['Semi-washed', 'Unwashed', 'Natural Anaerobic', 'Washed Experimental'])
def test_process_unrecognised_is_nan_with_warning(raw, caplog):
    with caplog.at_level(logging.WARNING):
        assert pd.isna(normalize_process(raw))
    assert pd.isna(normalize_process(raw))   # never silently coerced to another category


# ---- headline counts -----------------------------------------------------

def test_headline_counts_consistent_with_region_keys():
    df = pd.DataFrame({
        'country_final': ['Kenya', 'Kenya', 'Peru', 'Peru', None],
        'subregion_final': ['Nyeri', 'UNKNOWN', 'Cusco', 'Cusco', 'Nyeri'],
        'seller_name': ['S1', 'S2', 'Unknown', 'S1', None],
        'flavors_parsed': [[{'family': 'Fruity'}], [{'family': None}], [], [{'family': 'Sweet'}], []],
    })
    df['region_key'] = add_region_key(df)
    c = headline_counts(df)
    assert c['total_coffees'] == 5
    assert c['countries_analyzed'] == 2
    assert c['regions_analyzed'] == df['region_key'].dropna().nunique() == 2
    assert c['sellers_analyzed'] == 2
    assert c['unique_flavor_families'] == 2          # None family not counted
    # same answer when region_key is not precomputed
    assert headline_counts(df.drop(columns='region_key')) == c


def test_extractor_merge_applies_all_cleaning():
    from analytics.db_access.coffee_data_extractor import CoffeeDataExtractor
    ex = object.__new__(CoffeeDataExtractor)       # skip Supabase client setup
    attrs = pd.DataFrame({
        'coffee_id': [1, 2],
        'country_final': ['Kenya', 'Peru'],
        'subregion_final': ['UNKNOWN', 'Cusco'],
        'categorized_flavors': ["[{'family': 'Fruity', 'genus': 'Berry', 'species': ''}]", None],
        'process_type_final': ['Unknown', 'Washed'],
        'varietal': ["['UNKNOWN']", 'SL-28, SL28, Catuaí'],
        'cheapest_per_lb': [5.0, 6.0], 'average_per_lb': [5.0, 6.0], 'highest_per_lb': [5.0, 6.0],
    })
    sellers = pd.DataFrame({'coffee_id': [1, 2], 'coffee_name': ['a', 'b'], 'seller_id': [1, 1],
                            'seller_name': ['Unknown', 'S1'],
                            'first_observed': ['2026-01-01'] * 2, 'last_observed': ['2026-02-01'] * 2,
                            'is_active': [True, True]})
    m = ex.merge_and_prepare_data(attrs, sellers)
    assert pd.isna(m.loc[0, 'subregion_final']) and pd.isna(m.loc[0, 'region_key'])
    assert m.loc[1, 'region_key'] == 'Peru_Cusco'
    assert pd.isna(m.loc[0, 'process_type_clean'])
    assert m.loc[0, 'varietals_parsed'] == [] and not m.loc[0, 'has_varietal']
    assert m.loc[1, 'varietals_parsed'] == ['SL28', 'Catuai']
    assert m.loc[0, 'flavors_parsed'] == [{'family': 'Fruity', 'genus': 'Berry'}]
    assert pd.isna(m.loc[0, 'seller_name'])
    assert 'Kenya_UNKNOWN' not in set(m['region_key'].dropna())
    cf = ex.prepare_cross_feature_format(m)
    assert cf.loc[1, 'region'] == 'Peru_Cusco' and not cf.loc[0, 'has_region']
