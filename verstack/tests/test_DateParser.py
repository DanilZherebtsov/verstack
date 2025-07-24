import numpy as np
import pandas as pd
import os

from DateParser import DateParser

# Get the directory where this test file is located
test_dir = os.path.dirname(__file__)

datasets = {
    1: {
        "train": os.path.join(test_dir, "dateparser_train_1.parquet"),
        "test": os.path.join(test_dir, "dateparser_test_1.parquet"),
    },
    2: {
        "train": os.path.join(test_dir, "dateparser_train_2.parquet"),
        "test": os.path.join(test_dir, "dateparser_test_2.parquet"),
    },
    3: {
        "train": os.path.join(test_dir, "dateparser_train_3.parquet"),
        "test": os.path.join(test_dir, "dateparser_test_3.parquet"),
    },
    4: {
        "train": os.path.join(test_dir, "dateparser_train_4.parquet"),
        "test": os.path.join(test_dir, "dateparser_test_4.parquet"),
    },
}

# test overall DateParser not being broken
def test_DateParser():
    result = []
    for dataset in datasets:
        module = DateParser()
        train = pd.read_parquet(datasets[dataset]["train"])
        test = pd.read_parquet(datasets[dataset]["test"])
        transformed_train = module.fit_transform(train)
        transformed_test = module.transform(test)
        result.append(
            np.all(transformed_train.columns == transformed_test.columns)
        )
    assert result