def timer(func):
    '''
    Decorator to print func execution time

    Parameters
    ----------
    func : function to decorate

    Returns
    -------
    wrapped func: function execution time result

    '''
    import time
    from functools import wraps
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = round(end-start,5)
        if elapsed < 60:
            print(f"\nTime elapsed for {func.__name__} execution: {elapsed} seconds")
        elif 60 < elapsed < 3600:
            minutes = int(elapsed/60)
            seconds = round(elapsed%60,3)
            print(f"\nTime elapsed for {func.__name__} execution: {minutes} min {seconds} sec")
        else:
            hours = int(elapsed // 60 // 60)
            minutes = int(elapsed //60 % 60)
            seconds = int(elapsed % 60)
            print(f"\nTime elapsed for function {func.__name__} execution: {hours} hour(s) {minutes} min {seconds} sec")
        return result
    return wrapped

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:51:24 2022

@author: danil
"""

class Printer:
    
    __version__ = '0.1.1'
    
    def __init__(self, verbose=True):
        self.verbose=verbose

    def print(self, message=None, order=1, breakline=None, force_print=False):
        '''Output messages to the console based on seniority level (order).

        Logic:
            order=0 - program title print
            order=1 - major function title print
            order=2 - minor function title print
            order=3 - internal function first order results
            order=4 - internal function second order results
            order=5 - internal function third order results
            order='error' - error message print including traceback
        Parameters
        ----------
        message : str
            message to print
        order : int, optional
            order to tabulate the message print, can take values between 1 and 4. The default is 1.
        breakline : str, optional
            String symbol to print a breakline
        force_print : bool, optional
            If True will print message even if self.verbose == False. 
                Applicable for non-error important messages that need to be printed.

        Returns
        -------
        None.

        '''
        import traceback
        
        message_prefix = {
            1       :"\n * ",
            2       :"\n   - ",
            3       :"     . ",
            4       :"     .. ",
            5       :"     ... ",
            'error' : f"{traceback.format_exc()}\n! "
            }
        
        if order!='error' and not self.verbose and not force_print:
            return

        if not message:
            if breakline:
                print(f' {breakline*75}')
        else:
            if order == 0:
                print('\n')
                print('-'*75)
                print(f'{message}')
                print('-'*75)
            else:
                print(f'{message_prefix[order]}{message}')
                if breakline: 
                    print(f' {breakline*75}')