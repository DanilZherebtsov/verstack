import unittest
import sys
sys.path.append('../../')
import numpy as np
import time

from verstack import Multicore

iterable = np.random.randint(0, 100, 10000)

# test overall Multicore not broken

def func_to_parallel(x):
    print('\nsleeping 3 sec\n')
    time.sleep(3)
    return(x**2)

class TestMulticore(unittest.TestCase):
    
    
    def test_Multicore(self):
        worker = Multicore(workers = 2)
        multicore_result = worker.execute(func_to_parallel, iterable)
        manual_result = iterable**2
        result = np.all(multicore_result==manual_result)
        self.assertTrue(result)
        
if __name__ == '__main__':
    unittest.main()