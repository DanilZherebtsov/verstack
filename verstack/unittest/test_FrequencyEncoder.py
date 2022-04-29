import unittest
import sys
sys.path.append('../../')
import numpy as np
from common import generate_data

from verstack import FrequencyEncoder

class TestFrequencyEncoder(unittest.TestCase):
    
    def test_FrequencyEncoder(self):
        df = generate_data()
        # ---------------------
        module = FrequencyEncoder()
        df_train = module.fit_transform(df, 'x')
        df_test = module.transform(df)

        inverse_df_train = module.inverse_transform(df_train)
        inverse_df_test = module.inverse_transform(df_test)

        result_transform = np.all(df_train['x'] == df_test['x'])
        result_inverse_transform = np.all(inverse_df_train['x'].dropna() == df['x'].dropna())

        self.assertTrue(result_transform)
        self.assertTrue(result_inverse_transform)
        
if __name__ == '__main__':
    unittest.main()