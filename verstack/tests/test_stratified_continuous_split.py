import pytest
import sys
sys.path.append("../")
import pandas as pd
from stratified_continuous_split import scsplit



def test_scsplit():
    df = pd.read_parquet("boston_train.parquet")
    train, test = scsplit(df, stratify=df["medv"], test_size=0.5)
    percent_diff_in_mean_of_column_used_for_stratification = (
        train["medv"].mean() - test["medv"].mean()
    ) / train["medv"].mean()
    assert percent_diff_in_mean_of_column_used_for_stratification < 0.05
