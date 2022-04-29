import unittest
import sys
sys.path.append('../../')

from verstack.tools import Printer

class TestPrinter(unittest.TestCase):
        
    def test_Printer(self):

        result = None
        try:
            printer = Printer(verbose=True)
            printer.print('Message printed successfully', order=0)
        except:
            result='error'
        
        self.assertIsNone(result)
        
if __name__ == '__main__':
    unittest.main()