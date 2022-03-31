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
            print(f"\nTime elapsed for {func.__name__} execution: {elapsed} seconds\n")
        elif 60 < elapsed < 3600:
            minutes = int(elapsed/60)
            seconds = round(elapsed%60,3)
            print(f"\nTime elapsed for {func.__name__} execution: {minutes} min {seconds} sec\n")
        else:
            hours = int(elapsed // 60 // 60)
            minutes = int(elapsed //60 % 60)
            seconds = int(elapsed % 60)
            print(f"\nTime elapsed for function {func.__name__} execution: {hours} hour(s) {minutes} min {seconds} sec\n")
        return result
    return wrapped

def pretty_print(message, order = 1, verbose = True):
    '''Output messages to the console based on seniority level (order).

    Parameters
    ----------
    message : str
        message to print
    order : int, optional
        order to tabulate the message print, can take values between 1 and 4. The default is 1.
    verbose : bool, optional
        Flag to print or not print message.

    Returns
    -------
    None.

    '''
    if not verbose:
        return
    if order == 0:
        print('-'*70)
        print(f'{message}')
        print('-'*70)
    if order == 1:
        print(f'\n - {message}')
    if order == 2:
        print(f'   . {message}')
    if order == 3:
        print(f'   .. {message}')
    if order == 4:
        print(f'   ... {message}')
