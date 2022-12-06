import pytest
import sys
sys.path.append('../../')
from verstack.tools import Printer


def test_Printer():
    '''Test if Printer will print a message'''
    result = True
    try:
        printer = Printer(verbose=True)
        printer.print('Message printed successfully', order=0)
    except:
        result=False
    assert result
