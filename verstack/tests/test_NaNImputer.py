import pytest
import sys
sys.path.append('../')
from common import generate_data
from NaNImputer import NaNImputer

# test overall DateParser not being broken
def test_NaNImputer():
    '''Test if NaNImputer will impute all NaN values'''
    df = generate_data()
    module = NaNImputer()
    transformed = module.impute(df)
    result = sum(transformed.isnull().sum()) == 0
    assert result