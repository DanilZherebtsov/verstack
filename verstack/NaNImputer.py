import os
import gc
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from verstack import Factorizer
from verstack.tools import timer, Printer

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
                         "random_state": 42,
                         "n_jobs": -1}

params_regression = {"learning_rate": 0.05,
                     "num_leaves": 32,
                     "colsample_bytree": 0.9,
                     "subsample": 0.9,
                     "verbosity": -1,
                     "n_estimators": 100,
                     "random_state": 42,
                     "n_jobs": -1}


class NaNImputer:
    
    __version__ = '2.0.0'

    def __init__(self, train_sample_size = 30000, verbose = True):
        self._verbose = verbose
        self.printer = Printer(verbose=self.verbose)
        self._cols_to_impute = {}
        self._transformers = []
        self._train_sample_size = train_sample_size
        self._to_drop = []
        self._do_not_consider = []
        self._cols_constants = []
        self._fill_constants = {}
        '''
        NaNImputer class instance
        
        Parameters
        ----------
        train_sample_size : int, default = 30000
            Number of rows to use for training the NaNImputer model.
            If the dataset is smaller than train_sample_size, the whole dataset will be used.
        verbose : bool, default = True
            If True, prints information about the dataset, NaNImputer settings and
            the imputation progress.

        '''
    def __repr__(self):
        return f'NaNImputer(verbose: {self._verbose}\
            \n           train_sample_size: {self._train_sample_size}'
    
    # Validate init arguments
    # =========================================================================
    # cols_to_impute
    @property
    def cols_to_impute(self):
        return self._cols_to_impute
    # -------------------------------------------------------------------------
    # transformers
    @property
    def transformers(self):
        return self._transformers
    # -------------------------------------------------------------------------
    # train_sample_size
    @property
    def train_sample_size(self):
        return self._train_sample_size

    @train_sample_size.setter
    def train_sample_size(self, value):
        if type(value) != int : raise TypeError('train_sample_size must be an int')
        self._train_sample_size = value
    # -------------------------------------------------------------------------
    # verbose
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if type(value) != bool : raise TypeError('verbose must be a bool')
        self._verbose = value
    # -------------------------------------------------------------------------
    # to_drop
    @property
    def to_drop(self):
        return self._to_drop
    # -------------------------------------------------------------------------
    # do_not_consider
    @property
    def do_not_consider(self):
        return self._do_not_consider
    # -------------------------------------------------------------------------

    def _dont_need_impute(self, df):
        if not np.any(df.isnull()):
            return True

    def _print_data_dims(self, df):
        """Print initial information on the dataframe."""
        data_size = np.round(df.memory_usage().sum()/(1024*1024),2)
        if np.any(df.isnull()):
            nan_cols_num = df.isnull().sum().astype('bool').value_counts()[True]
        else:
            nan_cols_num = 0
        self.printer.print('Dataset dimensions:', order=3)
        self.printer.print(f'rows:         {df.shape[0]}', order=4)
        self.printer.print(f'columns:      {df.shape[1]}', order=4)
        self.printer.print(f'mb in memory: {data_size}', order=4)
        self.printer.print(f'NaN cols num: {nan_cols_num}', order=4)

    def _get_cols_to_impute(self, df):
        '''Find cols with NaN & their dtypes, save to class instance'''
        cols = [col for col in df if np.any(df[col].isnull())]
        types = [str(df[col].dropna().dtype) for col in df if np.any(df[col].isnull())]
        for ix, col in enumerate(cols):
            self._cols_to_impute[col] = {}
            self._cols_to_impute[col]['type'] = types[ix]

    def _drop_or_keep(self, df, col, train, message, order):
        '''Drop column if self.train == True else if col in self._to_drop and drop, else do not drop.'''
        if train:
            df.drop(col, axis = 1, inplace = True)
            self.printer.print(message, order)
            self._to_drop.append(col)
            return df
        else:
            if col in self._to_drop:
                df.drop(col, axis = 1, inplace = True)
                self.printer.print(message, order)
            return df     

    def _drop_hopeless_nan_cols(self, df, train):
        '''Drop cols with over 50% NaN or cols with constant nonNaN value'''
        self.printer.print('Drop hopeless NaN cols', order=2)
        for col in df:
            if np.any(df[col].isnull()):
                # drop nan cols with constant known values
                if df[col].dropna().nunique() == 1:
                    df = self._drop_or_keep(df, col, train, 
                                            message = f'droped column {col} with NaNs and a constant non-NaN value', 
                                            order = 2)
                # drop nan cols with over 50% missing
                elif df[col].isnull().sum() > int(len(df)*0.5):
                    df = self._drop_or_keep(df, col, train, 
                                            message = f'droped column {col} with NaNs and a constant non-NaN value', 
                                            order = 2)
        return df

    def _fill_object_nan_cols_with_string(self, df):
        """Fill missing values in text column with 'Missing_data' string value.
        Applicable to object type columns with over 500 unique values (considered as text).
        """
        object_nan_cols = [col for col in df.select_dtypes(include = 'O') if np.any(df[col].isnull())]
        for col in object_nan_cols:
            if df[col].nunique() > 200:
                df[col].fillna('Missing_data', inplace = True)
                self._do_not_consider.append(col)
                self.printer.print(f'Missing values in {col} replaced by "Missing_data" string', order=3)
        return df

    def _factorize_col(self, df, col):
        '''Factorize selected col, save encoder instance to NaNImputer instance'''
        enc = Factorizer(na_sentinel=np.nan)
        df = enc.fit_transform(df, col)
        self._transformers.append(enc)
        return df

    def _process_df(self, df):
        '''Encode all object cols and some numeric (subject to classification) cols by Factorizer'''
        self.printer.print('Processing whole data for imputation', order = 2)
        # first process all object type cols
        object_cols = df.select_dtypes(include = 'O').columns
        num_cols_to_process = len(object_cols) + len([col for col in self._cols_to_impute if col not in object_cols])
        cnt = 0
        for col in object_cols:
            if col not in self._do_not_consider:
                df = self._factorize_col(df, col)
                cnt+=1
                if cnt % 10 == 0:
                    self.printer.print(f'Processed {cnt} cols; {num_cols_to_process - cnt} to go', order = 3)
        # then process num cols that may be used as targets for classification
            # factorise, because class numbers order and sequence may be not from [0: n]
        for col in self._cols_to_impute:
            if col not in object_cols:
                objective = self._define_objective(df[col])
                if objective in ['binary', 'multiclass']:
                    df = self._factorize_col(df, col)
                    cnt+=1
                    if cnt % 10 == 0:
                        self.printer.print(f'Processed {cnt} cols; {num_cols_to_process - cnt} to go', order = 3)
        return df

    def _is_proper_float(self, x):
        '''Define if value is a float or an int represented as a float'''
        if x%1 == 0:
            return False
        else:
            return True
    
    def _define_objective(self, y):
        # --------------------------------------------------
        if self._cols_to_impute[y.name]['type'] == 'bool':
            objective = 'binary'
        # --------------------------------------------------
        if self._cols_to_impute[y.name]['type'] == 'object':
            if y.nunique() == 2:
                objective = 'binary'
            else:
                objective = 'multiclass'
        # -------------------------------------------------
        if 'float' in self._cols_to_impute[y.name]['type']:
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
        # -------------------------------------------------
        if 'int' in self._cols_to_impute[y.name]['type']:
            if y.nunique() == 2:
                objective = 'binary'
            else:
                if y.nunique() < 30:
                    objective = 'multiclass'
                else:
                    objective = 'regression'
        return objective
                                  
    def _train_predict(self, params, X_train, y_train, X_test):
        '''Train model on provided splits, output prediction'''
        dtrain = lgb.Dataset(np.array(X_train), label=y_train)
        model = lgb.train(params, dtrain)
        pred = model.predict(np.array(X_test))
        return pred

    def _decode_prediction(self, pred, y_name, encoder):
        '''Apply Factorizer.inverse_transform to encoded predictions'''
        pred = pd.DataFrame(pred, columns = [y_name])
        pred = encoder.inverse_transform(pred)
        pred = np.array(pred[y_name])
        return pred

    def _train_predict_decode(self, params, X_train, y_train, X_test):
        '''Apply _train_predict, if necessary encode/decode labels, transform probabilities into classes'''
        need_decoding = False
        try:
            pred = self._train_predict(params, X_train, y_train, X_test)
        except:
            enc = Factorizer()
            y_train_encoded = enc.fit_transform(pd.DataFrame(y_train), y_train.name)
            pred = self._train_predict(params, X_train, y_train_encoded, X_test)
            need_decoding = True
        # convert predictions into classes
        if params['objective'] == 'multiclass':
            pred = np.argmax(pred, axis = 1)
            if need_decoding:
                pred = self._decode_prediction(pred, y_train.name, enc)
        if params['objective'] == 'binary':
            pred = (pred > 0.5).astype(int)
        return pred
    
    def _select_params(self, objective, y):
        '''Select lightgbm parameters based on objective'''
        if objective == 'regression':
            params = params_regression.copy()
        else:
            params = params_classification.copy()
        params['objective'] = objective
        if objective == 'multiclass':
            params['num_classes'] = y.nunique()
        return params        

    def _sample_data(self, df, col):
        '''Stratified or random sample of data (self._train_sample_size samples)'''
        objective = self._define_objective(df[col].dropna())
        nan_ix = df[df[col].isnull()].index

        df_non_nan = df[~df.index.isin(nan_ix)]
        sample_size = self._train_sample_size
        if objective in ['binary', 'multiclass']:
            if sample_size < len(df_non_nan):
                train = df_non_nan.groupby(col, group_keys=False).apply(lambda x: x.sample(min(len(x), sample_size)))
                train_ix = train.index
            else:
                train_ix = df_non_nan.index
        else:
            if sample_size < len(df_non_nan):
                train_ix = df_non_nan.sample(n = sample_size).index
            else:
                train_ix = df_non_nan.index
        return train_ix, nan_ix

    def _get_high_corr_feats(self, df, col):
        '''Get the 10 most important features for a given column based on the\
            binary corellations with the target col. Do not consider cols which NaN values\
                were filled with 'Missing_data'
        '''                
        exclude_from_df = self._do_not_consider
        # catch object or datetime cols to exclude from corellations calculation
        not_supported = df.drop(col, axis = 1).select_dtypes(include = ['O', 'datetime']).columns.tolist()
        corellations = df.drop(exclude_from_df + [col] + not_supported, axis = 1).apply(lambda x: x.corr(df[col]))
        for i in corellations.index:
            corellations[i] = math.fabs(corellations[i])
            corellations = corellations.sort_values(ascending = False)
        feats = list(corellations[:10].index)
        return feats

    def _get_splits(self, df, col):
        '''Split df into train/test based on NaN indexes'''
        feats = self._get_high_corr_feats(df, col)
        train_ix, nan_ix = self._sample_data(df, col)
        
        X_train = df.loc[train_ix, feats]
        y_train = df.loc[train_ix, col]
        X_test = df.loc[nan_ix, feats]
        return X_train, y_train, X_test        
    
    def _predict_nan_in_col(self, df, col):
        '''Execute all functions to train the model and create prediction for a col in df'''
        try:
            objective = self._define_objective(df[col].dropna())
            X_train_col, y_train_col, X_test_col = self._get_splits(df, col) # already sampled and with 10 feats
            # clean up
            del df
            gc.collect()
            params = self._select_params(objective, y_train_col)
            pred = self._train_predict_decode(params, X_train_col, y_train_col, X_test_col)
            self.printer.print(f'Imputed ({objective:^10}) - {len(X_test_col):<8} NaN in {col}', order=3)
        except:
            self.printer.print(f'Error imputing col {col} ({objective})', order='error')
        return pred
       
    def _impute_single_core(self, df):
        '''Header function to trigger nan imputation in df'''
        self.printer.print(f'Imputing single core {len(self._cols_to_impute.keys())} cols', order = 2)
        for col in self._cols_to_impute.keys():
            pred = self._predict_nan_in_col(df, col)
            # insert preds into col nan_ix
            nan_ix = df[df[col].isnull()].index
            df.loc[nan_ix, col] = pred
        return df      

    def _inverse_transform(self, df):
        '''Apply Factorizer.inverse_transform to all encoded cols'''
        for enc in self._transformers:
            df = enc.inverse_transform(df)
        return df 

    def _clear_attributes_placeholders_for_test(self):
        '''Remove list of columns to impute and fitted transformers from instance attributes'''
        self._cols_to_impute = {}
        self._transformers = []        

    def _get_or_fill_constants(self, df, train):
        '''Get cols mean/mode from train or fill test unimputable cols with constants'''
        if train:
            for col in df:
                if df[col].dtype in ['O', 'bool']:
                    self._fill_constants[col] = df[col].mode().values[0]
                else:
                    self._fill_constants[col] = df[col].mean()
        else:
            # fill all NaNs or a column with single unique
            for col in df:
                if np.any(df[col].isnull()):
                    if df[col].nunique() in [0, 1]:
                        df[col].fillna(self._fill_constants[col], inplace = True)
        return df

    @timer
    def impute(self, data, train = True):
        '''
        Main function to execute NaN imputation in df.
        
        Parameters
        ----------
        data : pd.DataFrame
            data for NaN imputation.
        train : bool, optional
            flag for train / test set imputation. 
            If False will not drop excessive column when imputing NaN; 
            used for test set NaN imputation in order to output the same shaped imputed_df. 
            The default is True.
        
        Returns
        -------
        df : pd.DataFrame
            data with all NaN cols imputed.
            
        '''
        if self._dont_need_impute(data):
            self.printer.print('no missing data', order=2)
            return data
        df = data.copy()
        self.printer.print(f'Initiating NaNImputer.impute', order = 1)
        if self.verbose:
            self._print_data_dims(df)        
        if not train:
            self._clear_attributes_placeholders_for_test()
        # ------------------------------
        df = self._drop_hopeless_nan_cols(df, train)
        # ------------------------------
        df = self._get_or_fill_constants(df, train)
        # ------------------------------
        df = self._fill_object_nan_cols_with_string(df)
        # ------------------------------
        self._get_cols_to_impute(df)
        # ------------------------------
        df = self._process_df(df)
        # ------------------------------
        df = self._impute_single_core(df)
        df = self._inverse_transform(df)
        self.printer.print(f'Missing values after imputation: {sum(df.isnull().sum())}', order = 2)
        return df