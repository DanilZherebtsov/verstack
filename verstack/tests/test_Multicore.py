import pytest
import sys
sys.path.append('../')
import numpy as np
import time
from Multicore import Multicore

iterable = np.random.randint(0, 100, 10000)

# test overall Multicore not broken
def func_to_parallel(x):
    print('\nsleeping 3 sec\n')
    time.sleep(3)
    return(x**2)

def test_Multicore():
    '''Test if Multicore will achieve the same computation as manual computation'''
    worker = Multicore(workers = 2)
    multicore_result = worker.execute(func_to_parallel, iterable)
    manual_result = iterable**2
    result = np.all(multicore_result==manual_result)
    assert result
