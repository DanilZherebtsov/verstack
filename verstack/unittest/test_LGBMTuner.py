import pytest
import sys
sys.path.append('../../')
from common import generate_data
from verstack import LGBMTuner

#TODO: extend tests including gpu

def test_LGBMTuner():
    '''Test if LGBMTuner will fit and save optimized params'''
    df = generate_data(processed=True)
    
    # test passing custom_lgbm_params argument
    custom_params = {'zero_as_missing': True}
    module = LGBMTuner(metric='rmse', trials=20, visualization=False, refit=True, custom_lgbm_params = custom_params)
    # test changing the grid
    module.grid['max_data_in_leaf'] = {'choice' : [40, 50, 70]}
    X = df.drop('y', axis = 1)
    y = df['y']
    module.fit(X, y)
    result_trained_model = module.fitted_model is not None
    result_saved_optimized_params = module.best_params is not None
    assert result_trained_model
    assert result_saved_optimized_params
    print(module.best_params)
