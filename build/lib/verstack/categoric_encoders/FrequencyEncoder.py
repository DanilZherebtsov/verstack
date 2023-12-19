#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:35:49 2022

@author: danil
"""

import pandas as pd
import numpy as np
from verstack.categoric_encoders.args_validators import is_bool_na_sentinel, assert_fit_transform_args, assert_transform_args
from verstack.tools import Printer

class FrequencyEncoder():
    '''
    Encoder to represent categoric variable classes' frequency across the dataset.
    
    Can handle missing values - encode NaN by NaN frequency or leave NaN values untransformed.
    Resulting frequencies are normalized as a percentage.
    
    '''
    __version__ = '0.1.1'
    
    def __init__(self, na_sentinel = True):
        '''
        Initialize FrequencyEncoder instance.
        
        Parameters
        ----------
        na_sentinel : bool, optional
            If True - encode NaN by NaN frequency, else keep NaN untransformed.
            The default is True.
            
        Returns
        -------
        None.
        '''

        self._pattern = None # save for transform
        self._colname = None
        self._na_sentinel = is_bool_na_sentinel(na_sentinel)

    def __repr__(self): # what an object will say when you call/print it
        return 'verstack.categoric_encoders.FrequencyEncoder'

    @property 
    def pattern(self): 
        return self._pattern

    @property 
    def colname(self): 
        return self._colname

    @property 
    def na_sentinel(self): 
        return self._na_sentinel
        
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
                
        assert_fit_transform_args(df, colname)
        self._colname = colname
        encoded_df = df.copy()
        encoding = df.groupby(self._colname).size()
        if self._na_sentinel:
            na_count = encoded_df[colname].isnull().sum()
            encoding[np.nan] = na_count
        else:
            pass

        encoding[encoding.duplicated()] += .1
        # represent encoding as a percent from dataset length
        encoding = encoding/len(df)
        self._pattern = dict(zip(encoding.index, encoding))
        encoded_df[colname] = df[colname].map(encoding)
        return encoded_df

    def _check_data_for_transformation_colname_content(self, df):
        '''Assert self._colname to be in df'''
        if not self._colname or not self._colname in df:
            raise KeyError('Column to transform not found. Check get_colname() and df contents')
        else:
            pass

    def transform(self, df):
        '''
        Frequency encode data column saved in self._colname using saved patterns.
        
        Fill unknown categories with most common frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Data containing colname that needs encoding by initial class instance.

        Returns
        -------
        df : pd.DataFrame
            Data containing the transformed column.
        '''
        printer = Printer(verbose=True)

        assert_transform_args(df)
        self._check_data_for_transformation_colname_content(df)
        encoded_df = df.copy()
        encoded_df[self._colname] = encoded_df[self._colname].map(self._pattern)
        if self._na_sentinel:
            if encoded_df[self._colname].isnull().sum() > 0:
                printer.print('Filling unknown categories with most common frequency', order=3)
                encoded_df[self._colname].fillna(max(self._pattern.values()), inplace = True)
        return encoded_df      

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
        self._check_data_for_transformation_colname_content(df)
        inverse_transformed = df.copy()
        inverse_pattern = {val: key for key, val in self._pattern.items()}
        inverse_transformed[self._colname] = inverse_transformed[self._colname].map(inverse_pattern)
        return inverse_transformed