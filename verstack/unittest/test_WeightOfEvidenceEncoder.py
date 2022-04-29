import unittest
import sys
sys.path.append('../../')
from common import generate_data

from verstack import WeightOfEvidenceEncoder

class TestWeightOfEvidenceEncoder(unittest.TestCase):
    
    def test_WeightOfEvidenceEncoder(self):
        df = generate_data()
        # ---------------------
        module = WeightOfEvidenceEncoder()
        df_train = module.fit_transform(df, 'x', 'y_binary')
        df_test = module.transform(df)
        inverse_df_train = module.inverse_transform(df_train)
        inverse_df_test = module.inverse_transform(df_test)

        result_train_transform = df_train['x'].dtype==float
        result_test_transform = df_test['x'].dtype==float
        result_train_inverse_transform = inverse_df_test['x'].dtype=='O'
        result_test_inverse_transform = inverse_df_test['x'].dtype=='O'

        self.assertTrue(result_train_transform)
        self.assertTrue(result_test_transform)
        self.assertTrue(result_train_inverse_transform)
        self.assertTrue(result_test_inverse_transform)
        
if __name__ == '__main__':
    unittest.main()