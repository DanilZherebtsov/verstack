import pandas as pd
import numpy as np
from verstack.categoric_encoders.args_validators import is_bool_na_sentinel, assert_fit_transform_args, assert_transform_args, assert_binary_target
from verstack.tools import Printer

class WeightOfEvidenceEncoder():
    '''
    Encoder to represent categoric variables by Weight of evidence in regards to the binary target variable.
    If encoded value is negative - it represents a category that is more heavily enclided to the negative target class (0).
    
    When fit_transform() is used on a train set, variable is encoded with adding minor noize to reduce the risk of overfitting.
    When transform() unseen categories are encoded by 0 WoE.
    '''

    __version__ = '0.1.1'    
    
    def __init__(self, na_sentinel = True, **kwargs):
        '''
        Initialize WeightOfEvidence instance.
        
        Parameters
        ----------
        na_sentinel : bool, optional
            If True - encode NaN by 0 WoE, else keep NaN untransformed.
            The default is True.
        kwargs : category_encoders.woe.WOEEncoder parameters can be found at:
            https://contrib.scikit-learn.org/category_encoders/woe.html

        Returns
        -------
        None.
        '''
        self.printer = Printer(verbose=True)
        self._na_sentinel = is_bool_na_sentinel(na_sentinel)
        self._colname = None
        self._params = {'randomized':True, 'random_state':42, 'handle_missing':'return_nan' if not na_sentinel else 'value'} if not kwargs else kwargs
        if 'random_state' not in self._params.keys():
            self._params['random_state'] = 42
        if 'randomized' not in self._params.keys():
            self.randomized = True
        self.__generic_encoder = None # category_encoders.woe.WOEEncoder placeholder
        self.__pattern = None # for inverse_transform without noize

    def __repr__(self): # what an object will say when you call/print it
        return 'verstack.categoric_encoders.WeightOfEvidenceEncoder'
        
    @property 
    def colname(self): 
        return self._colname

    @property 
    def na_sentinel(self): 
        return self._na_sentinel

    @property 
    def params(self): 
        return self._params

    def fit_transform(self, df, colname, targetname):
        '''
        Fit encoder, transform column in df, save attributes for transform(/inverse_transform().
                                                                           
        Variable is encoded with adding minor noize to reduce the risk of overfitting.

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
        
        assert_fit_transform_args(df, colname)
        assert_binary_target(df, targetname)
        encoded_df = df.copy()
        self._colname = colname
        from category_encoders import WOEEncoder
        generic_encoder = WOEEncoder(**self._params)
        encoded_column = generic_encoder.fit_transform(df[colname], df[targetname])
        self.__generic_encoder = generic_encoder
        encoded_df[self._colname] = encoded_column
        
        # save inverse_transform pattern for test set (without noize)
        woe_vals_no_noize = self.transform(df)[self._colname].unique()
        original_vals = df[self._colname].unique()
        self.__pattern = dict(zip(woe_vals_no_noize, original_vals))

        return encoded_df        
        
    def transform(self, df):
        '''
        WOE Encode data column saved in self._colname using saved patterns.
        Unseen categories are encoded by 0 WoE.
        
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
        encoded_column = self.__generic_encoder.transform(df[self._colname])
        encoded_df[self._colname] = encoded_column
        return encoded_df

    def _check_random_state(self, seed):
        """Turn seed into a np.random.RandomState instance (used for inverse_transform()).
    
        Parameters
        ----------
        seed : None, int or instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.
        """
        import numbers
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, numbers.Integral):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)

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
        # first check for inverse_transform without noize
        if np.all([x in list(self.__pattern.keys()) for x in df[self._colname].unique()]):
            self.printer.print('Inverse transforming without noize', order=3)
            inverse_encoded_col = df[self._colname].map(self.__pattern)
            inverse_encoded_df = df.copy()
            inverse_encoded_df[self._colname] = inverse_encoded_col
            return inverse_encoded_df
        else:
            inverse_encoded_df = df.copy()
            random_state_generator = self._check_random_state(self.__generic_encoder.random_state)
            inverse_encoded_col = df[self._colname]  / random_state_generator.normal(1., self.__generic_encoder.sigma, df[self._colname].shape[0])
            
            ordinal_string_mapping = self.__generic_encoder.ordinal_encoder.mapping[0]['mapping']
            ordinal_string_mapping = {val: key for key, val in ordinal_string_mapping.items()}
            ordinal_woe_mapping = self.__generic_encoder.mapping[self._colname]
            final_mapping = {val_ord_str: ordinal_woe_mapping[key_ord_str] for key_ord_str, val_ord_str in ordinal_string_mapping.items()}
            final_mapping = {val: key for key, val in final_mapping.items()}
            # decrease the float depth for mapping (some division results in inverse_encoded_col might have slight differences)
            inverse_encoded_df[self._colname] = inverse_encoded_col.astype('float32').map({np.float32(key):val for key, val in final_mapping.items()})
            return inverse_encoded_df

