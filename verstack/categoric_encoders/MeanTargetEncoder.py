import pandas as pd
import numpy as np
from verstack.categoric_encoders.args_validators import is_bool_na_sentinel, assert_fit_transform_args, assert_transform_args

class MeanTargetEncoder():
    '''
    Encode train fold (set) cat cols by mean target calculated on different folds.

    To avoid target leakage train set encoding is performed by braking data into 5 folds & 
    encoding categories of each fold with their respective target mean values calculated on the other 4 folds.
    This will introduce minor noize to train data encoding (at fit_transform()) as a normalization technique. 
    Test set (transform()) is encoded without normalization.
    '''

    __version__ = '0.1.1'
    
    def __init__(self, save_inverse_transform = False, na_sentinel = True):
        '''
        Initialize MeanTargetEncoder instance.
        
        Parameters
        ----------
        save_inverse_transform : bool, optional
            If True - save a pattern for inverse_transform. The default is False.
        na_sentinel : bool, optional
            If True - fill NaN values by target mean, else keep NaN untransformed.
            The default is True.
            
        Returns
        -------
        None.
        '''

        self._pattern = None
        self._colname = None 
        self._na_sentinel = is_bool_na_sentinel(na_sentinel)
        self._save_inverse_transform = save_inverse_transform        
        self.__global_mean = None
        self.__inverse_transform_dict = {}

    def __repr__(self): # what an object will say when you call/print it
        return 'verstack.categoric_encoders.MeanTargetEncoder'
        
    @property 
    def pattern(self): 
        return self._pattern

    @property 
    def colname(self): 
        return self._colname

    @property 
    def na_sentinel(self): 
        return self._na_sentinel

    @property 
    def save_inverse_transform(self): 
        return self._save_inverse_transform

    def fit_transform(self, df, colname, targetname):
        '''
        Fit encoder, transform column in df, save attributes for transform(/inverse_transform().

        Parameters
        ----------
        df : pd.DataFrame
            Data containing the colname to transform.
        colname : str
            Column name in df to be transformed.
        targetname : str
            column name for extracting the mean values for each colname category.

        Returns
        -------
        transformed_df : pd.DataFrame
            Data with the column transformed.
        '''
        assert_fit_transform_args(df, colname, targetname)
        from sklearn.model_selection import KFold
        
        self._colname = colname
        self._pattern = dict(df.groupby(colname)[targetname].mean()) # save pattern (without smoothing) for transform()
        if self._na_sentinel:
            self._pattern[np.nan] = df[targetname].mean() 
        self.__global_mean = df[targetname].mean() # record global mean for NaN encoding
        encoded_df = df.copy()
        kf = KFold(n_splits = 5, shuffle=True, random_state=5)
        
        self.rev = {}
        for nfold, (tr_ix, val_ix) in enumerate(kf.split(df)):
            X_tr, X_val = df.loc[tr_ix, :], df.loc[val_ix, :]
            means = X_val.loc[:,colname].map(X_tr.groupby(colname)[targetname].mean())
            X_val.loc[:,colname] = means
            encoded_df.loc[val_ix, :] = X_val
            
            if self._save_inverse_transform: # save indexes and pattern for inverse_transform()
                self.__inverse_transform_dict[nfold] = {}
                self.__inverse_transform_dict[nfold]['val_ix'] = val_ix
                self.__inverse_transform_dict[nfold]['means'] = {val:key for key, val in dict(X_tr.groupby(colname)[targetname].mean()).items()}


        if self._na_sentinel:
            encoded_df[colname].fillna(self.__global_mean, inplace=True)
        return encoded_df

    def transform(self, df):
        '''
        Mean-target-encode data column saved in self._colname using saved patterns

        Parameters
        ----------
        df : pd.DataFrame
            Data containing colname that needs encoding by initial class instance.

        Returns
        -------
        df : pd.DataFrame
            Data containing the transformed column.
        '''
        assert_transform_args(df)
        
        encoded_df = df.copy()
        # save NaN indexes before encoding. To split the incoming NaNs and emerged NaNs due to new categories
        incoming_nan_indexes = df[self._colname].isnull()[df[self._colname].isnull()].index
        encoded_df[self._colname] = encoded_df[self._colname].map(self._pattern)
        # fill the unseen categories with self.__global_mean
        encoded_df[~encoded_df.index.isin(incoming_nan_indexes)][self._colname].fillna(self.__global_mean, inplace = True)
        # fill the incoming NaNs with self.__global_mean
        if self._na_sentinel:
            encoded_df[self._colname].fillna(self.__global_mean, inplace = True)
        return encoded_df
        
    def _inverse_transform_without_smoothing(self, df):
        '''Inverse transform the data encoded without noize (test set).'''
        print('\nInverse transforming without noize')
        inverse_encoded = df.copy()
        inverse_pattern = {val:key for key, val in self._pattern.items()}
        inverse_encoded[self._colname] = inverse_encoded[self._colname].map(inverse_pattern)
        return inverse_encoded
        
    def _inverse_transform_with_somoothing(self, df):
        '''Inverse transform the data encoded with noize (train set).'''
        inverse_encoded = df.copy()
        temp = df[self._colname].copy()
        for key in self.__inverse_transform_dict.keys():
            means = self.__inverse_transform_dict[key]['means']
            val_ix = self.__inverse_transform_dict[key]['val_ix']
            temp.loc[val_ix] = temp.loc[val_ix].map(means)
        inverse_encoded[self._colname] = temp
        return inverse_encoded
        
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

        # check for inverse_transform option without normalization
        assert_transform_args(df)
        encoded_vals = list(set(df[self._colname]))
        if not self._na_sentinel:
            encoded_vals = [val for val in encoded_vals if val == val]
        if np.all([v in list(self._pattern.values()) for v in encoded_vals]):
            return self._inverse_transform_without_smoothing(df)
        else:
            if not self._save_inverse_transform:
                print('\n"InstanceError: save_inverse_transform == True" option had not been enabled during initial data encoding.')
                return df
            else:
                print('\nInverse transforming with smoothing')
                return self._inverse_transform_with_somoothing(df)