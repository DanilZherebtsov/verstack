import unittest
import sys
sys.path.append('../../')
from common import generate_data

from verstack import DateParser

# test overall DateParser not being broken

class TestDateParser(unittest.TestCase):
    
    def test_DateParser(self):
        df = generate_data()
        module = DateParser(country='Russia', payday=[15,30])    
        transformed = module.fit_transform(df)
        result = len(module.datetime_cols)==1
        self.assertTrue(result)
        
if __name__ == '__main__':
    unittest.main()