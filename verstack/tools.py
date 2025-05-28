#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:51:24 2022

@author: danil
"""


def timer(func):
    """
    Decorator that times the execution of a function and prints the duration
    in a user-friendly format.
    
    Format rules:
    - Under 1 minute: prints time in seconds
    - 1-60 minutes: prints time in minutes and seconds
    - Over 60 minutes: prints time in hours, minutes, and seconds
    """
    import time
    from functools import wraps    

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Determine verbosity
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        elif args:
            try:
                verbose = args[0].verbose
            except AttributeError:
                verbose = True
        else:
            verbose = True
            
        start_time = time.perf_counter()
        # Execute the function
        result = func(*args, **kwargs)
        
        # Calculate elapsed time
        elapsed_seconds = time.perf_counter() - start_time
        
        # Format based on duration
        if elapsed_seconds < 60:
            time_str = f"{elapsed_seconds:.4f} seconds"
        elif elapsed_seconds < 3600:
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            time_str = f"{minutes}m {seconds}s"  # More concise format
        else:
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            seconds = int(elapsed_seconds % 60)
            time_str = f"{hours}h {minutes}m {seconds}s"  # More concise format
            
        # Determine if this is a class method
        if verbose:
            if args and hasattr(args[0], '__class__') and not isinstance(args[0], (int, float, str, bool, list, dict, tuple)):
                # This is likely a class method
                class_name = args[0].__class__.__name__
                print(f"'{class_name}.{func.__name__}()' executed in {time_str}")
            else:
                # This is a regular function
                print(f"'{func.__name__}()' executed in {time_str}")
                
        return result
    
    return wrapper


class Printer:
    
    __version__ = '0.1.2'
    
    def __init__(self, verbose=True):
        self.verbose=verbose

    def print(self, 
              message=None, 
              order=1, 
              breakline=None, 
              force_print=False, 
              leading_blank_paragraph=False, 
              trailing_blank_paragraph=False):
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
        
        leading_blank = '\n' if leading_blank_paragraph else ''
        trailing_blank = '\n' if trailing_blank_paragraph else ''

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
                print(f'{leading_blank}{message_prefix[order]}{message}{trailing_blank}')
                if breakline: 
                    print(f' {breakline*75}')