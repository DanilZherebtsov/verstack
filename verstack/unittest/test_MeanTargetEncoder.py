import pytest
import sys
sys.path.append('../../')
from common import generate_data
from verstack import MeanTargetEncoder

def test_MeanTargetEncoder():
    df = generate_data()
    # ---------------------
    module = MeanTargetEncoder()
    df_train = module.fit_transform(df, 'x', 'y')
    df_test = module.transform(df)
    inverse_df_test = module.inverse_transform(df_test)

    result_train_transform = df_train['x'].dtype==float
    result_test_transform = df_test['x'].dtype==float
    result_test_inverse_transform = inverse_df_test['x'].dtype=='O'
    result_returned_to_categorical = inverse_df_test['x'].dropna().dtype == 'O'

    assert result_train_transform
    assert result_test_transform
    assert result_test_inverse_transform
    assert result_returned_to_categorical
