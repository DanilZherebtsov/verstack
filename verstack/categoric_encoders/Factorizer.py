import pandas as pd
import numpy as np
from verstack.categoric_encoders.args_validators import is_not_bool_na_sentinel

class Factorizer():
    '''
    Assing numeric labels to categoric column (binary/multiclass).

    Can make transformation without transforming original NaN values, or assign any user defined \
        string/number to replace original NaN values.
    
    '''
    __version__ = '0.1.2'

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
        Unseen categories will be represented as NaN.

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:15:18 2022

@author: danil
"""

# !!!!!! for independent usage ADD DATA TRANSFORMATION IF INCLUDES OBJECT COLS, THEN MOVE UP BEFORE 
    # BINARIZER AND CAT ENCODER IN NOVAML


import pandas as pd
#from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
df = pd.read_csv('/Users/danil/OneDrive/Bell/datasets/_traditional/house_prices (rmsle,Id,SalePrice)/house_prices_train.csv')
#from verstack import Factorizer
import numpy as np


params_classification = {"task": "train",
                         "learning_rate": 0.05,
                         "num_leaves": 128,
                         "colsample_bytree": 0.7,
                         "subsample": 0.7,
                         "bagging_freq": 1,
                         "max_depth": -1,
                         "verbosity": -1,
                         "reg_alpha": 1,
                         "reg_lambda": 0.0,
                         "min_split_gain": 0.0,
                         "zero_as_missing": False,
                         "max_bin": 255,
                         "min_data_in_bin": 3,
                         "n_estimators": 100,
                         "random_state": 42}

params_regression = {"learning_rate": 0.05,
                     "num_leaves": 32,
                     "colsample_bytree": 0.9,
                     "subsample": 0.9,
                     "verbosity": -1,
                     "n_estimators": 100,
                     "random_state": 42}

class NaNImputer:
    
    def __init__(self):
        self.cols_to_impute = None
        self.transformers = []

    def _dont_need_impute(self, df):
        if not np.any(df.isnull()):
            return True

    def _get_cols_to_impute(self, df):
        cols = [col for col in df if np.any(df[col].isnull())]
        types = [str(df[col].dropna().dtype) for col in df if np.any(df[col].isnull())]
        self._cols_to_impute = dict(zip(cols, types))

    def _drop_nan_cols_with_constant_values(self, df):
        for col in df:
            if np.any(df[col].isnull()):
                if df[col].dropna().nunique() == 1:
                    print('.droped column {col} with NaN and a constant nonNaN value')
                    df.drop(col, axis = 1, inplace = True)
        return df

    def _get_high_corr_feats(self, df, col):
        """Get the n most important features for a given column based on
        binary corellations with the target col.

        Args:
            data_prepared (pandas.DataFrame):
                data prepared for modeling with all columns, including the target column
            col (str):
                target column name for which important features must be found
        Returns:
            top_feats (list):
                list of strings with the top (self.n_feats) number of features

        """
        import math
        corellations = df.drop(col, axis = 1).apply(lambda x: x.corr(df[col]))
        for i in corellations.index:
            corellations[i] = math.fabs(corellations[i])
            corellations = corellations.sort_values(ascending = False)
        feats = list(corellations[:10].index)
        return feats

    def _get_subset(self, df, col):
        feats = self._get_high_corr_feats(df, col)
        nan_idx = df[df[col].isnull()].index
        X_train = df.loc[~df.index.isin(nan_idx), feats]
        y_train = df.loc[~df.index.isin(nan_idx), col]
        X_test = df.loc[nan_idx, feats]
        return X_train, y_train, X_test

    def _process_df(self, df):
        for col in df.select_dtypes(include = 'O'):
            enc = Factorizer(na_sentinel=np.nan)
            df = enc.fit_transform(df, col)
            self.transformers.append(enc)
        return df

    def _is_proper_float(self, x):
        if x%1 == 0:
            return False
        else:
            return True
    
    def _define_objective(self, y):
        # ------------------------------------------
        if self._cols_to_impute[y.name] == 'bool':
            objective = 'binary'
        # ------------------------------------------
        if self._cols_to_impute[y.name] == 'object':
            if y.nunique() == 2:
                objective = 'binary'
            else:
                objective = 'multiclass'
        # ------------------------------------------
        if 'float' in self._cols_to_impute[y.name]:
            if y.nunique() == 2:
               objective = 'binary'
            else:
                # if decimal any decimal values are different than 0
                if np.any(y.dropna().apply(self._is_proper_float)): 
                    objective = 'regression'
                else:
                    
                    if y.nunique() < 30:
                        objective = 'multiclass'
                    else:
                        objective = 'regression'
        # ------------------------------------------
        if 'int' in self._cols_to_impute[y.name]:
            if y.nunique() == 2:
                objective = 'binary'
            else:
                if y.nunique() < 30:
                    objective = 'multiclass'
                else:
                    objective = 'regression'
        return objective
                                  
    def _select_params(self, objective, y):
        if objective == 'regression':
            params = params_regression.copy()
        else:
            params = params_classification.copy()

        params['objective'] = objective

        if objective == 'multiclass':
            params['num_classes'] = y.nunique()
        return params

    def _train_predict(self, params, X_train, y_train, X_test):
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain)
        pred = model.predict(X_test)
        # convert predictions into classes
        if params['objective'] == 'multiclass':
            pred = np.argmax(pred, axis = 1)
        if params['objective'] == 'binary':
            pred = (pred > 0.5).astype(int)
        return pred
    
    def _inverse_transform(self, df):
        for enc in self.transformers:
            df = enc.inverse_transform(df)
        return df 
    
    def impute(self, df):
        if self._dont_need_impute(df):
            print('no missing data')
            return df
    
        # ------------------------------
        df = self._drop_nan_cols_with_constant_values(df)
        # ------------------------------
        self._get_cols_to_impute(df)
        # ------------------------------
        df = self._process_df(df)
        # ------------------------------
        
        for col in self._cols_to_impute.keys():
            # ------------------------------
            # ------------------------------
            X_train_col, y_train_col, X_test_col = self._get_subset(df, col)            
            # ------------------------------
            objective = self._define_objective(y_train_col)
            # ------------------------------
            params = self._select_params(objective, y_train_col)
            # ------------------------------            
            pred = self._train_predict(params, X_train_col, y_train_col, X_test_col)
            df[col].loc[X_test_col.index] = pred
            print(f'.NaN imputed in {col}')

#        df = self._inverse_transform(df)
        
        return df
    
imp = NaNImputer()
cols = ['MasVnrType', 'SalePrice']
df = df[cols]

print(sum(df.isnull().sum()))
df = imp.impute(df)
print(sum(df.isnull().sum()))




for enc in imp.transformers:
    col = enc.colname
    # if col == 'MasVnrType':
    #     break
    pattern = enc.pattern
    print(f'COL: {col}')
    print(f'PATTERN: {pattern}')
    print(f'UNENCODED UNIQUE: {df[col].unique()}')
    print('-'*50)
    df = enc.inverse_transform(df)
    print(f'ENCODED UNIQUE: {df[col].unique()}')
    print(f'NUM NAN IN COL AFTER INVERSE: {df[col].isnull().sum()}')
    print(f'NUM NAN IN DF AFTER INVERSE: {sum(df.isnull().sum())}')
    print('='*50)
    print('\n')