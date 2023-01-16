import pytest
import sys
sys.path.append('../../')
from common import generate_data
from verstack import DateParser

# test overall DateParser not being broken
def test_DateParser():
    df = generate_data()
    module = DateParser(country='Russia', payday=[15,30])    
    transformed = module.fit_transform(df)
    result = len(module.datetime_cols)==2
    assert result
