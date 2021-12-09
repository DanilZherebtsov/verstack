import pandas as pd
import numpy as np
from verstack.categoric_encoders.args_validators import is_bool_na_sentinel, assert_fit_transform_args, assert_transform_args

class OneHotEncoder():
    '''
    Encoder to represent categoric (multiclass) variable as a set of 
    binary variables one for each class.
    
    '''
    __version__ = '0.1.2'
    
    def __init__(self, na_sentinel = True):
        '''
        Initialize OneHotEncoder instance.
        
        Parameters
        ----------
        na_sentinel : bool, optional
            If True - create a separate column for np.nan values 
            else will not create a column for np.nan values. The default is True.
            
        Returns
        -------
        None.
        '''

        self._categories = None # save for inverse_transform()
        self._colname = None
        self._na_sentinel = is_bool_na_sentinel(na_sentinel)
        self.__one_hot_cols_sequence = None # save for transform()
        self.__original_fit_transform_cols_sequence = None # save for inverse_transform()
        self.__original_transform_cols_sequence = None # save for inverse_transform()
        self.__prefix = None

    def __repr__(self): # what an object will say when you call/print it
        return 'verstack.categoric_encoders.OneHotEncoder'

    # ----------------------------------------------------------------------------------------
    # define getters
    @property 
    def categories (self): 
        return self._categories

    @property 
    def colname(self): 
        return self._colname

    @property 
    def na_sentinel(self): 
        return self._na_sentinel
    # ----------------------------------------------------------------------------------------


    def _create_output_df(self, df, one_hot_df):
        '''Common method for concatenating the one_hot_df to original df'''
        resulting_df = pd.concat([df, one_hot_df], axis = 1)
        # drop original encoded column
        resulting_df.drop(self._colname, axis = 1, inplace = True)
        return resulting_df
    
    def fit_transform(self, df, colname, prefix = None):
        '''
        Fit encoder, transform column in df, save attributes for transform(/inverse_transform().

        Change na_sentinel to False if df[colname].isnull().sum() == 0

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the colname to transform.
        colname : str
            Column name in df to be transformed.
        prefix : str/int/float/bool/none, opnional.
            String to append DataFrame column names.
            The default is None.

        Returns
        -------
        transformed_df : pd.DataFrame
            Data with the column transformed.
        '''
        
        assert_fit_transform_args(df, colname)
        
        # change na_sentinel if col does not include NaNs
        if df[colname].isnull().sum() == 0:
            self._na_sentinel = False

        self._colname = colname
        self.__original_fit_transform_cols_sequence = df.columns.tolist()
        self.__prefix = prefix
        data = pd.Series(df[colname])
        one_hot_df = pd.get_dummies(df[colname], prefix = self.__prefix, dummy_na = self._na_sentinel)
        self._categories = df[colname].unique().tolist()
        if not self._na_sentinel:
            self._categories = [x for x in self._categories if x == x]
        self.__one_hot_cols_sequence = one_hot_df.columns.tolist()
        return self._create_output_df(df, one_hot_df)

    def _align_to_fit_transform(self, transformed):
        '''
        Correct columns order of transformed df to the columns order recorded
        during fit_transform

        Parameters
        ----------
        transformed : pd.DataFrame
            Dataframe after transform.

        Returns
        -------
        transformed : pd.DataFrame
            Dataframe after columns order correction.

        '''
        if transformed.columns.tolist() == self.__one_hot_cols_sequence:
            return transformed
        else:
            for col in self.__one_hot_cols_sequence:
                if col not in transformed:
                    print(f'added empty one_hot column {col}')
                    transformed[col] = 0
            transformed = transformed[self.__one_hot_cols_sequence]
        return transformed

    def transform(self, df):
        '''
        One-hot-encode data column saved in self.colname using saved patterns

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the column which had been passed at fit_transform().

        Returns
        -------
        transformed_df : pd.DataFrame
            Data containing a set of one-hot-encoded columns, without the original column.

        '''

        assert_transform_args(df)
        self.__original_transform_cols_sequence = df.columns.tolist()        
        one_hot_df = pd.get_dummies(df[self._colname], prefix=self.__prefix, dummy_na = self._na_sentinel)
        one_hot_df = self._align_to_fit_transform(one_hot_df)
        return self._create_output_df(df, one_hot_df)

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
        
        assert_transform_args(df)
        try: # check if the one-hot-encoded columns are present in df
            assert np.all([col in df for col in self.__one_hot_cols_sequence]) 
        except AssertionError:
            print('Error: Passed data does not contain the ohe-hot-encoded columns sequence')
            return df        
        inverse_transform_map = dict(zip(self.__one_hot_cols_sequence, self._categories))
        inverse_transformed_col = df[self.__one_hot_cols_sequence].replace(0, np.nan).idxmax(axis=1)
        inverse_transformed_col = inverse_transformed_col.map(inverse_transform_map)
        resulting_df = df.drop(self.__one_hot_cols_sequence, axis = 1)
        resulting_df[self._colname] = inverse_transformed_col
        try:
            resulting_df = resulting_df[self.__original_fit_transform_cols_sequence]
        except KeyError:
            resulting_df = resulting_df[self.__original_transform_cols_sequence]
        return resulting_df
