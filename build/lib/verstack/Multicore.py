import os
import gc
from tabnanny import verbose
import numpy as np
import pandas as pd
import concurrent.futures
from verstack.tools import timer, Printer
import operator

class Multicore():
    """
    Parallelize any function execution program.
    Use any function and iterable(s) to perform computation using all of the available cpu cores.

    """    
    __version__ = '0.1.3'
    
    def __init__(self,
                 workers=False,
                 multiple_iterables=False,
                 verbose=True):
        """
        Initialize class instance.

        Parameters
        ----------
        workers : int/bool, optional
            Number of workers. The default is False.
        multiple_iterables : bool, optional
            If func needs to iterate over multiple iterables, set to True. The default is False.
        verbose : bool, optional
            Enable function execution progress print to the console

        Returns
        -------
        None.

        """
        
        self.verbose = verbose
        self.printer = Printer(verbose=self.verbose)
        self.workers = workers
        self.multiple_iterables = multiple_iterables
        print(self.__repr__())

    # print init parameters when calling the class instance
    def __repr__(self):
        return f'Multicore(workers = {self.workers},\
            \n          multiple_iterables = {self.multiple_iterables},\
            \n          verbose = {self.verbose}'

    # Validate init arguments
    # =========================================================
    # workers
    workers = property(operator.attrgetter('_workers'))

    @workers.setter
    def workers(self, w):
        if w == False:
            self._workers = self._assign_workers(w)
        elif type(w) == int:
            if w <= os.cpu_count():
                self._workers = w
            else:
                raise Exception(f'Workers number: {w} is greater than available cores: {os.cpu_count()}')
        else:
            raise Exception('Workers value must be either False or a number of desired cpu cores for the job')
    # -------------------------------------------------------
    # multiple_iterables
    multiple_iterables = property(operator.attrgetter('_multiple_iterables'))

    @multiple_iterables.setter
    def multiple_iterables(self, mi):
        if type(mi) != bool : raise Exception('multiple_iterables must be bool (True/False)')
        self._multiple_iterables = mi
    # -------------------------------------------------------
    # verbose
    verbose = property(operator.attrgetter('_verbose'))

    @verbose.setter
    def verbose(self, val):
        if type(val) != bool : raise Exception('verbose must be bool (True/False)')
        self._verbose = val
    # =========================================================

    def _assign_workers(self, workers):
        """
        Define number of workers from user input or based on all available cpu cores.

        Parameters
        ----------
        workers : int/bool
            Number of workers if passed by user or False.

        Returns
        -------
        workers : int
            Number of workers passed by user or number of available cpu cores.

        """
        if workers:
            workers = workers
        else:
            try:
                workers = os.cpu_count()
            except NotImplementedError:
                workers = 1
        return workers

    @timer
    def execute(self, func, iterable, *args):
        """
        Parallelize function execution over iterables.
        
        Iterate over iterable(s) broken down into chunks according to the 
        number of defined/available cores. 
        
        If func needs more then one iterable, pass pass multiple iterables in a nested list.
        Iterables in a list must be in the same order as defined in func.
        
    
        Parameters
        ----------
        func : function.
            iteration function to execute in parallel
        iterable : list/pd.Series/pd.DataFrame/dictionary
            data to iterate over
        *args : 
            additional arguments to func. E.g. if func needs data from pd.DataFrame
            to perform calculations on the iterable, df should be passed into func *args
        multiple_iterables : bool, optional
            If multiple_iterables are passed, they should, set to True.  
            The default is False.
        workers : int, optional
            Number of parallel workers. 
            The default is False: will use all the available cores.
    
        Returns
        -------
        list
            Function output result. If multiple outputs are expected, nested list is returned.
    
        """
        
        self.printer.print(f'Initializing {self.workers} workers for {func.__name__} execution', order=1)
    
        if self.multiple_iterables:
            # create chunks
            chunks_dict = {}
            for ix, obj in enumerate(iterable):
                chunks_dict[ix] = np.array_split(obj, self.workers)            
            chunks_lst = list(chunks_dict.values())            
            del chunks_dict
            chunks = []        
            for lst_ix in range(len(chunks_lst)):
                if lst_ix <= len(chunks_lst)-1:
                    for sublist_ix in range(len(chunks_lst[lst_ix])):
                        chunks.append([chunks_lst[lst_ix][sublist_ix] for lst_ix in range(len(chunks_lst))])
            del chunks_lst
            gc.collect()
            # -----
            with concurrent.futures.ProcessPoolExecutor(max_workers = self.workers) as executor:       
                results = [(ix, executor.submit(func, *chunk)) for ix, chunk in enumerate(chunks)]
        else:
            # -----
            # create chunks
            chunks = np.array_split(iterable, self.workers)
            # -----
            with concurrent.futures.ProcessPoolExecutor(max_workers = self.workers) as executor:       
                results = [(ix, executor.submit(func, *args, chunk)) for ix, chunk in enumerate(chunks)]
        del chunks
        gc.collect()
        # ------------------------------------------------------------------------
        # extract imputed cols from multiprocessing results
        data_from_results = []    
        for f in results:
            data_from_results.append(f[1].result())
        del results
        gc.collect()
        # ------------------------------------------------------------------------
        # join the results
        if isinstance(data_from_results[0], tuple):
            result = {}
            for cnt in range(len(data_from_results[0])):
                lst = [x[cnt] for x in data_from_results]
                flat_lst = [item for sublist in lst for item in sublist]
                result['output_'+(str(cnt))] = flat_lst
            return [val for val in result.values()]
        else: # if array or list
            if isinstance(data_from_results[0], np.ndarray):
                result = np.concatenate(data_from_results)   
            elif isinstance(data_from_results[0], list):
                result = [item for sublist in data_from_results for item in sublist]
            elif isinstance(data_from_results[0], pd.Series):
                result = pd.concat(data_from_results)
            elif isinstance(data_from_results[0], dict):
                result = {}
                for dictionary in data_from_results:
                    result.update(dictionary)
            else:
                raise TypeError(f"Invalid return type for function '{func.__name__}'."
                        " Only list, dict, Numpy ndarray and Panda's Series are supported!")
            return result

