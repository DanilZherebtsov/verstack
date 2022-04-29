import unittest
import sys
sys.path.append('../../')
from common import generate_data

from verstack import Stacker

# test overall LGBMTuner not being broken

class TestDateLGBMTuner(unittest.TestCase):
    
    def test_LGBMTuner(self):
        
        df = generate_data(processed=True)
        module = Stacker('regression', auto=True, epochs=5, gridsearch_iterations=4)
        X = df.drop('y', axis = 1)
        y = df['y']
        X_train = module.fit_transform(X, y)
        result_layers_with_feats_present = len(module.stacked_features)>0
        self.assertTrue(result_layers_with_feats_present)
        
if __name__ == '__main__':
    unittest.main()