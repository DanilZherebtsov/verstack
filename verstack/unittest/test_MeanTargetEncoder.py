import unittest
import sys
sys.path.append('../../')
from common import generate_data

from verstack import MeanTargetEncoder

class TestMeanTargetEncoder(unittest.TestCase):
    
    def test_MeanTargetEncoder(self):
        df = generate_data()
        # ---------------------
        module = MeanTargetEncoder()
        df_train = module.fit_transform(df, 'x', 'y')
        df_test = module.transform(df)
        inverse_df_test = module.inverse_transform(df_test)

        result_train_transform = df_train['x'].dtype==float
        result_test_transform = df_test['x'].dtype==float
        result_test_inverse_transform = inverse_df_test['x'].dtype=='O'

        self.assertTrue(result_train_transform)
        self.assertTrue(result_test_transform)
        self.assertTrue(result_test_inverse_transform)
        
if __name__ == '__main__':
    unittest.main()