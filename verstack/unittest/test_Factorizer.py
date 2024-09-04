import pytest
import sys

sys.path.append("../")
import numpy as np
from common import generate_data
from categoric_encoders.Factorizer import Factorizer


def test_Factorizer():
    """Test if Factorizer will make cat_col into numeric then back to categorical"""
    df = generate_data()
    # ---------------------
    module = Factorizer()
    df_train = module.fit_transform(df, "x")
    df_test = module.transform(df)

    inverse_df_train = module.inverse_transform(df_train)
    inverse_df_test = module.inverse_transform(df_test)

    result_became_numeric = df_train["x"].dtype == "int"
    result_transform = np.all(df_train["x"] == df_test["x"])
    result_inverse_transform = np.all(
        inverse_df_train["x"].dropna() == df["x"].dropna()
    )
    result_returned_to_categorical = inverse_df_train["x"].dtype == "O"

    assert result_became_numeric
    assert result_transform
    assert result_inverse_transform
    assert result_returned_to_categorical
