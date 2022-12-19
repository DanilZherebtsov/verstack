import pytest
import sys
sys.path.append('../../')
from common import generate_data
from verstack import PandasOptimizer

def test_PandasOptimizer():
    df = generate_data()
    module = PandasOptimizer()
    optimized_df = module.optimize_memory_usage(df)
    result_optimized_smaller_than_original = module.optimized_data_size_mb < module.original_data_size_mb
    assert result_optimized_smaller_than_original
