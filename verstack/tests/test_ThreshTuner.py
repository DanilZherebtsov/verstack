import sys
sys.path.append('../')
import numpy as np
from ThreshTuner import ThreshTuner

# just check workability
def test_ThreshTuner():
    np.random.seed(42)
    pred = np.linspace(0,1,500)
    true = np.zeros(500)
    true[:200] = 1
    
    tuner = ThreshTuner(n_thresholds=10, min_threshold=0.4, max_threshold=0.7)
    tuner.fit(true, pred)
    result_with_optional_parameters = tuner.best_predict_ratio()['threshold'].values[0] == 0.6

    tuner = ThreshTuner()
    tuner.fit(true, pred)
    result_with_default_parameters = tuner.best_predict_ratio()['fraction_of_1'].values[0] == 0.4

    true = [0,0,0,0,0,0,0,0,0,0]
    pred = [0.1, 0.2, 0.23, 0.11, 0.34, 0.346, 0.27, 0.18, 0.29, 0.22]
    tuner = ThreshTuner()
    tuner.fit(true, pred)
    result_with_1_unique_label = tuner.best_score()['balanced_accuracy_score'].values[0] == 1.0

    assert result_with_optional_parameters
    assert result_with_default_parameters
    assert result_with_1_unique_label