import numpy as np
from common import generate_data
from categoric_encoders.OneHotEncoder import OneHotEncoder


def test_OneHotEncoder():
    '''Test if OneHotEncoder will transform one cat_col into multiple numeric cols'''
    df = generate_data()
    # ---------------------
    module = OneHotEncoder()
    module = OneHotEncoder()
    df_train = module.fit_transform(df, 'x')
    result_created_cols = np.all([col in df_train for col in ['a','b','c','d',np.nan]])    
    assert result_created_cols
