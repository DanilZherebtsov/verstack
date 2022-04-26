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

def pretty_print(message=None, order=1, verbose=True, underline=None):
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
    verbose : bool, optional
        Flag to print or not print message.
    underline : str, optional
        String symbol to create an underline below the message

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
    
    if not verbose:
        return

    if not message:
        if underline:
            print(f' {underline*75}')
    else:
        if order == 0:
            print('\n')
            print('-'*75)
            print(f'{message}')
            print('-'*75)
        else:
            print(f'{message_prefix[order]}{message}')
            if underline: 
                print(f' {underline*75}')

def verbosity_decorator(func, verbose):
    '''Decorator for pretty_print to set verbosity globaly.

    Useful for large projects to inherit the verbosity level
    with a single setting'''
    from functools import wraps
    @wraps(func)
    def decorated(*args, **kwargs):
        kwargs['verbose'] = verbose
        func(*args, **kwargs)
    return decorated
