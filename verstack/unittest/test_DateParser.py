import pytest
import sys
import numpy as np
import pandas as pd

sys.path.append("../")
from DateParser import DateParser

datasets = {
    1: {
        "train": "unittest/dateparser_train_1.parquet",
        "test": "unittest/dateparser_test_1.parquet",
    },
    2: {
        "train": "unittest/dateparser_train_2.parquet",
        "test": "unittest/dateparser_test_2.parquet",
    },
    3: {
        "train": "unittest/dateparser_train_3.parquet",
        "test": "unittest/dateparser_test_3.parquet",
    },
}


# test overall DateParser not being broken
def test_DateParser():
    result = []
    module = DateParser()
    for dataset in datasets:

        train = pd.read_parquet(datasets[dataset]["train"])
        test = pd.read_parquet(datasets[dataset]["test"])
        transformed_train = module.fit_transform(train)
        transformed_test = module.transform(test)
        result.append(
            np.all(transformed_train.columns == transformed_test.columns)
        )
    assert result
