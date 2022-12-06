import pytest
import sys
sys.path.append('../../')
import numpy as np
from verstack import ThreshTuner

# just check workability
def test_ThreshTuner():
    np.random.seed(42)
    pred = np.random.uniform(0,1,500)
    true = np.random.randint(0,2,500)
    
    tuner = ThreshTuner(n_thresholds=10, min_threshold=0.4, max_threshold=0.7)
    tuner.fit(true, pred)
    result = tuner.best_predict_ratio()['threshold'].values[0] == 0.5333333333333333
    assert result