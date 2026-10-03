import json
import math

import numpy as np
import pandas as pd
import pytest

from analytics.frontend.flags import as_bool, significant_mask
from analytics.frontend.data_cache_generator import FrontendDataCacheGenerator


@pytest.fixture
def gen(tmp_path):
    return FrontendDataCacheGenerator(cache_dir=str(tmp_path))


def roundtrip(gen, obj):
    # default=str mimics the cache writer: any leftover numpy type would become a string
    return json.loads(json.dumps(gen._sanitize_for_json(obj), default=str))


def test_numpy_scalars_become_native(gen):
    out = roundtrip(gen, {'b': np.bool_(False), 't': np.bool_(True),
                          'i': np.int64(3), 'f': np.float32(1.5)})
    assert out == {'b': False, 't': True, 'i': 3, 'f': 1.5}
    assert out['b'] is False and out['t'] is True
    assert isinstance(out['i'], int)


def test_nan_inf_none(gen):
    out = roundtrip(gen, [float('nan'), np.float64('nan'), np.inf, None, pd.NA])
    assert out == [None] * 5


def test_nested_and_dataframe(gen):
    df = pd.DataFrame({'is_significant': np.array([True, False]), 'p': [0.01, np.nan]})
    out = roundtrip(gen, {'a': [{'x': (np.bool_(True), np.int32(2))}], 'df': df,
                          'arr': np.array([1, 2])})
    assert out['a'] == [{'x': [True, 2]}]
    assert out['df'] == [{'is_significant': True, 'p': 0.01},
                         {'is_significant': False, 'p': None}]
    assert out['arr'] == [1, 2]


def test_to_dict_records_booleans_are_real_bools(gen):
    df = pd.DataFrame({'is_significant': [True, False]})
    s = json.dumps(gen._sanitize_for_json(df.to_dict('records')), default=str)
    assert '"True"' not in s and '"False"' not in s
    assert 'true' in s and 'false' in s


@pytest.mark.parametrize('val,exp', [
    (True, True), (False, False), ('True', True), ('False', False),
    ('true', True), ('false', False), (None, False), (float('nan'), False),
    (np.bool_(True), True), (np.bool_(False), False), (1, True), (0, False),
])
def test_as_bool(val, exp):
    assert as_bool(val) is exp


def test_significant_mask_mixed():
    s = pd.Series([True, 'True', 'False', False, None])
    assert significant_mask(s).tolist() == [True, True, False, False, False]
    assert significant_mask(pd.Series(['False', 'False'])).sum() == 0
