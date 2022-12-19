import gc
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from verstack.tools import Printer

INT_LIMITS = [
    ['int8',  (np.iinfo(np.int8).min, np.iinfo(np.int8).max)],
    ['int16', (np.iinfo(np.int16).min, np.iinfo(np.int16).max)],
    ['int32', (np.iinfo(np.int32).min, np.iinfo(np.int32).max)],
    ['int64', (np.iinfo(np.int64).min, np.iinfo(np.int64).max)]
    ]

FLOAT_LIMITS = [
    ['float16', (np.finfo(np.float16).min, np.finfo(np.float16).max)],
    ['float32', (np.finfo(np.float32).min, np.finfo(np.float32).max)],
    ['float64', (np.finfo(np.float64).min, np.finfo(np.float64).max)]
    ]

class PandasOptimizer:

    __version__ = '0.0.1'
    
    '''Class for memory usage optimization reading file with pandas.read_csv.'''
    def __init__(self, **kwargs):
        '''Init PandasReadOptimized object.
        
        Args:
            pd_read_func (function): pandas.read_csv or pandas.read_excel
                default: pandas.read_csv
            sep (str): separator for csv file
                default: ','
            usecols (list): list of columns to read
                default: None
            encoding (str): encoding for csv file
                default: 'utf-8'
            chunksize (int): chunk size for iterating over large dataframe
                default: 100000
                
        Returns:
            PandasReadOptimized object
            
        '''
        self.pd_read_func = kwargs.get('pd_read_func', pd.read_csv)
        self.sep = kwargs.get('sep', ',')
        self.delimiter = kwargs.get('delimiter', None)
        self.usecols = kwargs.get('usecols', None)
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.chunksize = kwargs.get('chunksize', 100000)
        self.optimized_dtypes = {}
        self.original_data_size_mb = 0
        self.optimized_data_size_mb = 0
        self.optimized_to_original_ratio = 0
        self.verbose = kwargs.get('verbose', True)
        self.pandas_kwargs = self._populate_pandas_kwargs()
        self.printer = Printer(verbose = self.verbose)

    def __repr__(self):
        return f'PandasReadOptimized(sep = {self.sep}\
            \n                    usecols = {self.usecols}\
            \n                    encoding = {self.encoding}\
            \n                    chunksize = {self.chunksize}\
            \n                    verbose = {self.verbose})'

    # Validate init arguments
    # =========================================================================
    # pd_read_func
    @property
    def pd_read_func(self):
        return self._pd_read_func

    @pd_read_func.setter
    def pd_read_func(self, value):
        if not callable(value):
            raise TypeError('pd_read_func must be a function (pd.read_csv or pd.read_excel)')
        self._pd_read_func = value

    # -------------------------------------------------------------------------
    # sep
    @property
    def sep(self):
        return self._sep

    @sep.setter
    def sep(self, sep):
        if not isinstance(sep, str):
            raise TypeError('sep must be a string')
        self._sep = sep    
    # -------------------------------------------------------------------------
    # usecols
    @property
    def usecols(self):
        return self._usecols

    @usecols.setter
    def usecols(self, usecols):
        if usecols is not None:
            if not isinstance(usecols, list):
                raise TypeError('usecols must be a list of strings')
        self._usecols = usecols
    # -------------------------------------------------------------------------
    # encoding
    @property
    def encoding(self):
        return self._encoding
    
    @encoding.setter
    def encoding(self, encoding):
        if not isinstance(encoding, str):
            raise TypeError('encoding must be a string')
        self._encoding = encoding
    # -------------------------------------------------------------------------
    # chunksize
    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, chunksize):
        if not isinstance(chunksize, int):
            raise TypeError('chunksize must be an integer')
        self._chunksize = chunksize
    # -------------------------------------------------------------------------
    # verbose
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be a boolean')
        self._verbose = verbose
    # =========================================================================

    def _populate_pandas_kwargs(self):
        '''Populate pandas kwargs for reading file based on init arguments.
        
        Args:
            None
        
        Returns:
            pandas_kwargs (dict): dictionary with pandas kwargs

        '''
        pandas_kwargs = {}
        pandas_kwargs['sep'] = self.sep
        pandas_kwargs['delimiter'] = self.delimiter
        pandas_kwargs['usecols'] = self.usecols
        pandas_kwargs['encoding'] = self.encoding
        return pandas_kwargs

    def _yield_blocks(self, files, size=65536):
        '''Yield blocks of file for reading'''
        while True:
            b = files.read(size)
            if not b: break
            yield b

    def _get_nrows(self, path):
        '''Get number of rows in dataframe'''
        with open(path, "r", encoding = self.encoding, errors = 'ignore') as f:
            nrows = (sum(bl.count("\n") for bl in self._yield_blocks(f)))
        return nrows

    def _get_header(self, path):
        '''Get header of dataframe'''
        header = self.pd_read_func(path, sep = self.sep, nrows = 0).columns.tolist()
        return header

    def get_shape(self, path_or_df):
        '''Get shape of dataframe without reading it into memory, (not used in optimization).
        
        Args:
            path (str or pd.DataFrame): path to file or pd.DataFrame object
        
        Returns:
            shape (tuple): shape of dataframe
            
        '''
        if isinstance(path_or_df, pd.DataFrame):
            shape = path_or_df.shape
        else:
            header = self._get_header(path_or_df)
            nrows = self._get_nrows(path_or_df)
            shape = (nrows, len(header))
        return shape

    def _discover_col_dtype(self, actual_min, actual_max, limits_list):
        '''Discover data type of column.
        
        Args:
            actual_min (int): min value of column
            actual_max (int): max value of column
            limits_list (list): list of limits for data types (INT_LIMITS or FLOAT_LIMITS)
            
        Returns:
            type_name (str): name of data type
            
        '''
        for type_definition in limits_list:
            type_name = type_definition[0]
            allowed_min = type_definition[1][0]
            allowed_max = type_definition[1][1]
            within_type_range = (actual_min > allowed_min, actual_max < allowed_max)
            if np.all(within_type_range):
                return type_name
    
    def _discover_chunk_dtypes(self, chunk):
        '''Discover data types of columns in chunk.
        
        Args:
            chunk (pandas.DataFrame): chunk of data
            
        Returns:
            None
            
        '''
        global INT_LIMITS, FLOAT_LIMITS
        for col in chunk.select_dtypes(exclude = ['O', bool]):
            actual_min = chunk[col].min()
            actual_max = chunk[col].max()
            col_type = chunk[col].dtype
            if col_type == 'int':
                type_name = self._discover_col_dtype(actual_min, actual_max, INT_LIMITS)
            else:
                type_name = self._discover_col_dtype(actual_min, actual_max, FLOAT_LIMITS)
            self.optimized_dtypes[col] = type_name

    def discover_dtypes(self, path_or_df):
        '''Discover data types of all columns in dataframe.

        Populate self.optimized_dtypes with data types of columns.
        
        Args:
            path_or_df (str or pandas.DataFrame): path to file or pd.DataFrame object
            
        Returns:
            dict: dictionary with data types of columns
            
        '''
        if isinstance(path_or_df, pd.DataFrame):
            self.optimized_dtypes = {}
            self._discover_chunk_dtypes(path_or_df)
            return self.optimized_dtypes
        else:
            self.original_data_size_mb = 0
            kwargs = self.pandas_kwargs.copy()
            kwargs['chunksize'] = self.chunksize
            with pd.read_csv(path_or_df, **kwargs) as reader:
                if self.verbose:
                    for chunk in tqdm(reader):
                        original_chunksize_mb = sys.getsizeof(chunk)/1024/1024
                        self.original_data_size_mb += original_chunksize_mb
                        self._discover_chunk_dtypes(chunk)
                        gc.collect()
                else:
                    for chunk in reader:
                        original_chunksize_mb = sys.getsizeof(chunk)/1024/1024
                        self.original_data_size_mb += original_chunksize_mb
                        self._discover_chunk_dtypes(chunk)
                        gc.collect()

        return self.optimized_dtypes

    def _optimize_df(self, df):
        '''Optimize dataframe by converting columns to optimal data types.
        
        Args:
            df (pandas.DataFrame): dataframe to optimize
            
        Returns:
            df (pandas.DataFrame): optimized dataframe
            
        '''
        self._discover_chunk_dtypes(df)
        try:
            df = df.astype(self.optimized_dtypes)
        except KeyError as e:
            self.printer.print('Columns names mismatch, initialize a fresh instance of PandasOptimizer', order='error')
            raise
        return df

    def _get_optimized_data_size_and_ratio(self, df):
        '''Get optimized data size and ratio of original data size.
        
        Args:
            df (pandas.DataFrame)
            
        Returns:
            None
        '''
        self.optimized_data_size_mb = sys.getsizeof(df)/1024/1024
        self.optimized_to_original_ratio = self.optimized_data_size_mb/self.original_data_size_mb

    def _print_number_of_iterations(self, path):
        '''Print number of iterations.'''
        if not self.optimized_dtypes:
            nrows = self._get_nrows(path)
            if nrows < self.chunksize:
                self.printer.print('Learning optimized data types by processing 1 chunk', order=1)
            else:
                self.printer.print(f'Learning optimized data types by processing {int(nrows/self.chunksize)+1} chunks (iterations)', order=1)

    def _print_optimization_results(self):
        '''Print optimization results.'''
        self.printer.print(f'Original data size: {np.round(self.original_data_size_mb,2)} MB', order=2)
        self.printer.print(f'Optimized data size: {np.round(self.optimized_data_size_mb,2)} MB', order=2)
        self.printer.print(f'Optimized data percentage of origianl data: {np.round(self.optimized_to_original_ratio*100,2)}%', order=2)
        
    def optimize_memory_usage(self, path_or_df):
        '''Read dataframe & optimized data types or optimize existing dataframe.
        
        Args:
            path_or_df (str or pandas.DataFrame): path to file or pd.DataFrame object
            
        Returns:
            df (pandas.DataFrame): dataframe with optimized data types
            
        '''
        if isinstance(path_or_df, pd.DataFrame):
            self.original_data_size_mb = sys.getsizeof(path_or_df)/1024/1024
            df = self._optimize_df(path_or_df)
            self._get_optimized_data_size_and_ratio(df)
            self._print_optimization_results()
        else:
            self._print_number_of_iterations(path_or_df)
            if not self.optimized_dtypes:
                self.discover_dtypes(path_or_df)
            self.printer.print('Loading data with optimized dtypes...')
            df = pd.read_csv(path_or_df, dtype = self.optimized_dtypes, **self.pandas_kwargs)
            self._get_optimized_data_size_and_ratio(df)
            self._print_optimization_results()
        
        return df
