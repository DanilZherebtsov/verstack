import pandas as pd
import numpy as np
from verstack.categoric_encoders.args_validators import is_not_bool_na_sentinel

class Factorizer():
    '''
    Assing numeric labels to categoric column (binary/multiclass).

    Can make transformation without transforming original NaN values, or assign any user defined \
        string/number to replace original NaN values.
    
    '''
    __version__ = '0.1.3'

    def __init__(self, na_sentinel = -1):
        '''
        Initialize Factorizer instance.
        
        Parameters
        ----------
        na_sentinel : int/str/np.nan, optional
            Will replace NaN by a passed value. Pass np.nan if need to keep NaN untouched.
            The default is -1.
            
        Returns
        -------
        None.
        '''

        self._pattern = None
        self._colname = None
        self._na_sentinel = is_not_bool_na_sentinel(na_sentinel)
        self._transformed_col_dtype = None # save for transform()

    def __repr__(self): # what an object will say when you call/print it
        return 'verstack.categoric_encoders.Factorizer'

    # ----------------------------------------------------------------------------------------
    # define getters
    @property 
    def pattern(self): 
        return self._pattern

    @property 
    def colname(self): 
        return self._colname

    @property 
    def na_sentinel(self): 
        return self._na_sentinel
    # ----------------------------------------------------------------------------------------
    # no setters: only configurable attribute na_sentinel is defined at init...

    def fit_transform(self, df, colname):
        '''
        Fit encoder, transform column in df, save attributes for transform(/inverse_transform().

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the colname to transform.
        colname : str
            Column name in df to be transformed.

        Returns
        -------
        transformed_df : pd.DataFrame
            Data with the column transformed.
        '''
        
        self._colname = colname
        # try:
        #     data = pd.Series(df[colname])
        # except TypeError:
        #     print('Acceptable arguments for fit_transform(pd.DataFrame, str(colname))')
        #     return
        data = pd.Series(df[colname])
        pattern = {}
        # use pandas default na_sentinel == -1
        labels, uniques = pd.factorize(data)
        pattern = dict(zip(uniques, set(labels)))
        # change na_sentinel to Factorizer options (including np.nan that is not allowed in pandas)

        if -1 in labels:
            labels = [x if x!= -1 else self._na_sentinel for x in labels]
            nan_dict = {np.nan : self._na_sentinel}
            pattern = {**pattern, **nan_dict}
        self._pattern = pattern
        transformed_df = df.copy()
        transformed_df[colname] = labels
        self._transformed_col_dtype = transformed_df[colname].dtype
        return transformed_df

    def transform(self, df):
        '''
        Factorize data column saved in self._colname using saved patterns
        Unseen categories will be represented as self.na_sentinel.

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the column which had been passed at fit_transform().

        Returns
        -------
        transformed_df : pd.DataFrame
            Data containing the transformed column.

        '''
        data = pd.Series(df[self._colname])
        result = data.map(self._pattern).tolist()
        # convert back to int because mapping will make all float if nan was present
        result = [int(x) if x==x else x for x in result]
        result = [x if x==x else self.na_sentinel for x in result]
        transformed_df = df.copy()
        transformed_df[self._colname] = result
        # align column type to that of fit_transform(). May be necessary if train had NaNs and test does not.
        try: # try because unseen categories (if any) will appear as NaN
            transformed_df[self._colname] = transformed_df[self._colname].astype(self._transformed_col_dtype)
        except:
            pass
        return transformed_df
    
    def inverse_transform(self, df):
        '''
        Return transformed column in df to original values.

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the column which had been passed at fit_transform().

        Returns
        -------
        transformed_df : pd.DataFrame
            Data containing the transformed column.
        '''
        
        inverse_pattern = {val: key for key, val in self._pattern.items()}
        data = pd.Series(df[self._colname])
        try:
            result = np.vectorize(inverse_pattern.get)(data)
        except ValueError:
            result = np.vectorize(inverse_pattern.get, otypes='O')(data)
        result = [x if x not in ['nan'] else np.nan for x in result]
        transformed_df = df.copy()
        transformed_df[self._colname] = result
        return transformed_df
