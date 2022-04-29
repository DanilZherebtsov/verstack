import unittest
import sys
sys.path.append('../../')
from common import generate_data

from verstack import NaNImputer

# test overall DateParser not being broken

class TestNaNImputer(unittest.TestCase):
    
    def test_NaNImputer(self):
        df = generate_data()
        module = NaNImputer()
        transformed = module.impute(df)
        result = sum(transformed.isnull().sum()) == 0
        self.assertTrue(result)
        
if __name__ == '__main__':
    unittest.main()