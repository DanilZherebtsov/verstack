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