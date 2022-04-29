import unittest
import sys
sys.path.append('../../')
import numpy as np
from common import generate_data

from verstack import OneHotEncoder

class TestOneHotEncoder(unittest.TestCase):
    
    def test_OneHotEncoder(self):
        df = generate_data()
        # ---------------------
        module = OneHotEncoder()
        module = OneHotEncoder()
        df_train = module.fit_transform(df, 'x')
        result_created_cols = np.all([col in df_train for col in ['a','b','c','d',np.nan]])
        
        self.assertTrue(result_created_cols)
                
if __name__ == '__main__':
    unittest.main()