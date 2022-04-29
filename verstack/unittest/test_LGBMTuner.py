import unittest
import sys
sys.path.append('../../')
from common import generate_data

from verstack import LGBMTuner

# test overall LGBMTuner not being broken

class TestDateLGBMTuner(unittest.TestCase):
    
    def test_LGBMTuner(self):
        
        df = generate_data(processed=True)
        module = LGBMTuner(metric='rmse', trials=20, visualization=False, refit=True)
        X = df.drop('y', axis = 1)
        y = df['y']
        module.fit(X, y)
        result_trained_model = module.fitted_model is not None
        result_saved_optimized_params = module.best_params is not None
        self.assertTrue(result_trained_model)
        self.assertTrue(result_saved_optimized_params)
        
if __name__ == '__main__':
    unittest.main()