import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import timeit
import multiprocessing
import concurrent.futures
import operator

class NaNImputer():
    """Missing values (NaNs) imputation program.

    Performs various operations on a dataset including:
        - data temporary transformation (factorization), in order to fit
            models on top.
        - search and replace the incorrectly named missing values.
            E.g. ('No data'/'Missing'/'None'). Configurable.
        - select important features for each column to speed up the
            model training. Configurable.
        - select an appropriate model (based on XGBoost) and parameters.
            Configurable.
        - subset data by rows.
        - split into train/test, fit/predict missing values.
        - NaNs in pure text fields are filled with 'Missing_data'
            text. Configurable.
        - NaNs in columns with a single constant value are not imputed.
            Appropriate message is displayed.
        - can impute NaNs in all of the columns containing them or
            in user defined columns. Configurable.
        - imputation is performed in a multiprocessing mode utilizing
            all the available processor cores. Configurable.

        All the agrumets have default values aimed at best performance, please
        check the __init__ docstring for details.

    """
    __version__ = '0.1.1'

    def __init__(self,
                 conservative = False,
                 n_feats = 10,
                 nan_cols = None,
                 fix_string_nans = True,
                 multiprocessing_load = 3,
                 verbose = True,
                 fill_nans_in_pure_text = True,
                 drop_empty_cols = True,
                 drop_nan_cols_with_constant = True,
                 feature_selection = 'correlation'):
        """Initialize class instance.

        All arguments have default values and are initialized for the best performance.

        Args:
            conservative (bool, optional):
                Model complexity level used to impute missing values.
                If True: model will be set to less complex and much faster.
                Default = False
            n_feats (int, optional):
                Number of corellated independent features to be used for
                corresponding column (with NaN) model training and imputation.
                Default = 10
            nan_cols (list, optional):
                List of columns to impute missing values in.
                If None - all columns with missing values will be used.
                Default = None
            fix_string_nans (bool, optional):
                Find possible missing values in numeric columns that had been
                (mistakenly) encoded as strings, E.g. 'Missing'/'NaN'/'No data'
                and replace them with np.nan
                Default = True
            multiprocessing_load (int, optional):
                Levels of parallel multiprocessing computing.
                1 = single core
                2 = half of all available cores
                3 = all available cores
                Default = 3
            verbose (bool, optional):
                Print the imputation progress.
                Default = True
            fill_nans_in_pure_text (bool, optional):
                Fill the missing values in text fields by string 'Missing_data'
                Applicable for text fields (not categoric)
                Default = True
            drop_empty_cols (bool, optional):
                Drop columns with all NaNs
                Default = True
            drop_nan_cols_with_constant (bool, optional):
                Drop columns containing NaNs and all other constant values
                Default = True
            feature_selection (str, optional):
                Define algorithm to select most important feats for each
                column imputation. Options: "correlation"/"feature_importance"
                Default = "correlation"
        Returns:
            None.

        """
        self.encoding_map = None
        self.metadata = None
        self.nan_cols = nan_cols
        self.conservative = conservative
        self.n_feats = n_feats
        self.fix_string_nans = fix_string_nans
        self.verbose = verbose
        self.multiprocessing_load = multiprocessing_load
        self.num_workers = self._estimate_workers()
        self.fill_nans_in_pure_text = fill_nans_in_pure_text
        self.drop_empty_cols = drop_empty_cols
        self.drop_nan_cols_with_constant = drop_nan_cols_with_constant
        self.feature_selection = feature_selection
        self.droped_cols = []
        print(self.__repr__())

    # print init parameters when calling the class instance
    def __repr__(self):
        return f'NaNImputer(conservative = {self.conservative}, n_feats = {self.n_feats},\
            \n           fix_string_nans = {self.fix_string_nans}, verbose = {self.verbose},\
                \n           multiprocessing_load = {self.multiprocessing_load}, fill_nans_in_pure_text = {self.fill_nans_in_pure_text},\
                    \n           drop_empty_cols = {self.drop_empty_cols}, drop_nan_cols_with_constant = {self.drop_nan_cols_with_constant}\
                        \n           feature_selection = {self.feature_selection})'


    # Validate init arguments
    # =========================================================
    # nan_cols
    def _check_nan_cols_contents(self, n):
        elements_count = 0
        for element in n:
            if type(element) == str:
                elements_count += 1
        if elements_count == len(n):
            return True
        else:
            return False

    nan_cols = property(operator.attrgetter('_nan_cols'))

    @nan_cols.setter
    def nan_cols(self, n):
        if n == None:
            self._nan_cols = n
        if n != None and type(n) != list:
            raise Exception('nan_cols must be a list')
        elif n != None and not self._check_nan_cols_contents(n):
            raise Exception('nan_cols must be strings')
        else:
            self._nan_cols = n
    # -------------------------------------------------------
    # conservative
    conservative = property(operator.attrgetter('_conservative'))

    @conservative.setter
    def conservative(self, c):
        if type(c) != bool : raise Exception('conservative must be bool (True/False)')
        self._conservative = c
    # -------------------------------------------------------
    # n_feats
    n_feats = property(operator.attrgetter('_n_feats'))

    @n_feats.setter
    def n_feats(self, nf):
        if type(nf) != int : raise Exception('n_feats must be int')
        if type(nf) == int and nf < 1: raise Exception('n_feats must be a positive int greater than 0')
        self._n_feats = nf
    # -------------------------------------------------------
    # fix_string_nans
    fix_string_nans = property(operator.attrgetter('_fix_string_nans'))

    @fix_string_nans.setter
    def fix_string_nans(self, fsn):
        if type(fsn) != bool : raise Exception('fix_string_nans must be bool (True/False)')
        self._fix_string_nans = fsn
    # -------------------------------------------------------
    # vervbose
    verbose = property(operator.attrgetter('_verbose'))

    @verbose.setter
    def verbose(self, v):
        if type(v) != bool : raise Exception('vervbose must be bool (True/False)')
        self._verbose = v
    # -------------------------------------------------------
    # multiprocessing_load
    multiprocessing_load = property(operator.attrgetter('_multiprocessing_load'))

    @multiprocessing_load.setter
    def multiprocessing_load(self, mpl):
        if type(mpl) != int : raise Exception('multiprocessing_load must be int')
        if type(mpl) == int and mpl not in [1,2,3] : raise Exception('multiprocessing_load can take values 1, 2 or 3')
        self._multiprocessing_load = mpl
    # -------------------------------------------------------
    # fill_nans_in_pure_text
    fill_nans_in_pure_text = property(operator.attrgetter('_fill_nans_in_pure_text'))

    @fill_nans_in_pure_text.setter
    def fill_nans_in_pure_text(self, fnt):
        if type(fnt) != bool : raise Exception('fill_nans_in_pure_text must be bool (True/False)')
        self._fill_nans_in_pure_text = fnt
    # -------------------------------------------------------
    # drop_empty_cols
    drop_empty_cols = property(operator.attrgetter('_drop_empty_cols'))

    @drop_empty_cols.setter
    def drop_empty_cols(self, dec):
        if type(dec) != bool : raise Exception('drop_empty_cols must be bool (True/False)')
        self._drop_empty_cols = dec
    # -------------------------------------------------------
    # drop_nan_cols_with_constant
    drop_nan_cols_with_constant = property(operator.attrgetter('_drop_nan_cols_with_constant'))

    @drop_nan_cols_with_constant.setter
    def drop_nan_cols_with_constant(self, dnc):
        if type(dnc) != bool : raise Exception('drop_nan_cols_with_constant must be bool (True/False)')
        self._drop_nan_cols_with_constant = dnc
    # -------------------------------------------------------
    # feature_selection
    feature_selection = property(operator.attrgetter('_feature_selection'))

    @feature_selection.setter
    def feature_selection(self, fs):
        if type(fs) != str : raise Exception('feature_selection must ba of type(str)')
        if fs not in ['correlation', 'feature_importance'] : raise Exception('feature selection can be either "correlation" or "feature_importance"')
        self._feature_selection = fs

    # =======================================================
    def _print_data_dims(self, data):
        """Print initial information on the dataframe."""
        data_size = np.round(data.memory_usage().sum()/(1024*1024),2)
        if np.any(data.isnull()):
            nan_cols_num = data.isnull().sum().astype('bool').value_counts()[True]
        else:
            nan_cols_num = 0
        print('\nDataset dimensions:')
        print(f' - rows:         {data.shape[0]}')
        print(f' - columns:      {data.shape[1]}')
        print(f' - mb in memory: {data_size}')
        print(f' - NaN cols num: {nan_cols_num}')
        print('--------------------------')

    def _get_metadata(self, data):
        """Collect each column metadata from the original dataframe.

        To be used further for correct task type (regression/classification)
        selection and model parameters.

        Args:
            data (pandas.DataFrame): original dataframe with missing values.

        Returns:
            None.

        """
        self.metadata = {}
        for i in data:
            self.metadata[i] = {}
            self.metadata[i]['dtype'] = data[i].dtype
            self.metadata[i]['nunique'] = data[i].nunique()
        self.metadata['data_shape'] = data.shape

    def _estimate_workers(self):
        """Translate the init argument 'multiprocessing_load' into cpu_count.

        Returns:
            int: cpu_count for multiprocessing.

        """
        if self.multiprocessing_load == 3:
            return os.cpu_count() # this argument will use all available cores
        elif self.multiprocessing_load == 2:
            return int(os.cpu_count()/2)
        else:
            return 1

    def _correct_string_nans(self, data):
        """Replace the possible string values in numeric columns as a NaN.

        Example: 'Missing'/'None'/'NaN'/'No data'/etc

        In every 'object' type column try to represent most frequent value as
        type(int(value)). If success, find unique values in a column that can
        not be represented as type(int(value)) and replace them with np.nan

        Args:
            data (pandas.DataFrame):
                data for fixing string represented nans.

        Returns:
            data (pandas.DataFrame):
                data with string represented nans replaced by np.nan.

        """
        for col in data.select_dtypes(include = 'O'):
            most_frequent_val = data[col].value_counts().index[0]
            try:
                float(most_frequent_val)
                for unique in data[col].value_counts().index:
                    try:
                        float(unique)
                    except ValueError:
                        d = {unique:np.nan}
                        data = data.replace({col:d})
                data[col] = data[col].astype('float')
                print(f'Changed (fixed) column {col} to type float')
                print('Incorrectly represented values replaced by np.nan\n')
            except ValueError:
                continue
        return data

    def _get_objective_for_model(self, col):
        """Define objective for a model.

        Examine the column and define wether it is a regression or
        classification task and define the objective for the
        binary/multiclass/regression models.
        Logic:
            - for 'bool' and 'object' type columns possible objectives are
                'binary:logistic'/'multi:softprob'
            - for 'int' and 'float' type columns 'binary:logistic' if
                col.nunique() == 2, 'multi:softprob' if 2 col.nunique() < 70,
                else 'reg:squarederror'

        Args:
            col (str):
                column in name.

        Returns:
            objective (str):
                value for model definition:
                "binary:logistic"/"multi:softprob"/"reg:squarederror".

        """
        if self.metadata[col]['dtype'] in ['object','bool']:
            if self.metadata[col]['nunique'] == 2:
                objective = 'binary:logistic'
            else:
                objective = 'multi:softprob'
        else:
            if self.metadata[col]['nunique'] == 2:
                objective = 'binary:logistic'
            elif 2 < self.metadata[col]['nunique'] <= 70:
                objective = 'multi:softprob'
            else:
                objective = 'reg:squarederror'
        return objective

    def _select_model_complex(self, objective): # self, change conservative = False to conservative (passed from impute function)
        """Select a complex model for a given column: XGBRegressor orXGBClassifier.

        Args:
            objective (str):
                objective for model definition:
                "binary:logistic"/"multi:softprob"/"reg:squarederror"

        Returns:
            model (class instance):
                XGBRegressor or XGBClassifier model with predefined parameters

        """
        params = {
             'objective'         : objective,
             'colsample_bytree'  : 0.7,
             'subsample'         : 0.7,
             'random_state'      : 42,
             'verbosity'         : 1,
             'n_jobs'            : -1
             }
        if objective in ['binary:logistic', 'multi:softprob']:
            model = XGBClassifier(**params)
        else:
            model = XGBRegressor(**params)
        return model

    def _select_model_conservative(self, objective): # self, change conservative = False to conservative (passed from impute function)
        """Select an appropriate model for a given column: XGBRegressor orXGBClassifier.

        Args:
            objective (str):
                objective for model definition:
                "binary:logistic"/"multi:softprob"/"reg:squarederror"
            conservative (bool):
                model complexity parameter. Default = False
        Returns:
            model (class instance):
                XGBRegressor or XGBClassifier model with predefined parameters
        """
        # this speeds up the multiclass model
        if objective == 'multi:softprob':
            objective = 'reg:squarederror'
        params = {
            'objective'         : objective,
            'colsample_bytree'  : 0.5,
            'subsample'         : 0.5,
            'learning_rate'     : 0.5,
            'n_estimators'      : 100,
            'tree_method'       : 'hist',
            'random_state'      : 42,
            'verbosity'         : 1,
            'n_jobs'            : -1
            }

        if objective in ['binary:logistic', 'multi:softprob']:
            model = XGBClassifier(**params)
        else:
            model = XGBRegressor(**params)
        return model


    def _subset_columns(self, data_prepared, col):
        """Subset data to a limited number of relevant columns for missing
        data imputation in the target column.

        Args:
            data_prepared (pandas.DataFrame):
                data prepared for modeling
            col (str):
                column name to be used as a target for the model
        Returns:
            model (class instance):
                XGBRegressor or XGBClassifier model with predefined parameters

        """
        if self.metadata['data_shape'][1] >= 30:
            if self.feature_selection == 'correlation':
                top_feats = self._get_high_corr_feats(data_prepared, col)
            else:
                top_feats = self._get_important_feats(data_prepared, col)
            top_feats.append(col)
            subset = data_prepared[top_feats]
#            print('Ncols reduced to 10')
        else:
            subset = data_prepared
        return subset

    def _prepare_data(self, data):
        """Transform all non numeric columns to numeric.

        Perform factorization of each column and save a dict(encoding_map) to reverse transform
        the values back to the original format after missing values imputation.
        Updates the self.encoding_map dictionary.

        Args:
            data (pandas.DataFrame):
                data to be transformed

        Returns:
            data_prepared (pandas.DataFrame):
                data transformed to all numeric

        """
        if self.fix_string_nans:
            data = self._correct_string_nans(data)
        self._get_metadata(data)
        if not self.nan_cols:
            self.nan_cols = data.columns[data.isnull().any()].tolist()

        self.encoding_map = {}
        for col in data.select_dtypes(include = 'O'):
            cat_codes = data[col].dropna().unique().tolist()
            num_codes = [ix[0] for ix in enumerate(cat_codes)]
            self.encoding_map[col] = dict(zip(cat_codes, num_codes))
            data[col] = data[col].map(self.encoding_map[col])

        return data

    def _get_high_corr_feats(self, data_prepared, col):
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
        corellations = data_prepared.drop(col, axis = 1).apply(lambda x: x.corr(data_prepared[col]))
        for i in corellations.index:
            corellations[i] = math.fabs(corellations[i])
            corellations = corellations.sort_values(ascending = False)
        feats = list(corellations[:10].index)
        return feats

    def _get_important_feats(self, data_prepared, col):
        """Get the n most important features for a given column based on feature_importance.

        Build a conservative (simple) XGBoost model using all columns and up to
        5000 rows of data, extract feature importances.

        Args:
            data_prepared (pandas.DataFrame):
                data prepared for modeling with all columns, including the target column
            col (str):
                target column name for which important features must be found
        Returns:
            top_feats (list):
                list of strings with the top (self.n_feats) number of features

        """
        n_rows = 5000

        ix = data_prepared[col].dropna().index.tolist()
        temp_data = data_prepared.iloc[ix]
        if len(temp_data) > n_rows:
            # make sure that all of the column classes (if classification) will stay in subset after sample
            if self.metadata[col]['nunique'] <= 70:
                temp_data = temp_data.groupby(col, group_keys=False).apply(lambda x: x.sample(min(len(x), n_rows)))
            # or just sample (if regression)
            else:
                temp_data = temp_data.sample(n = n_rows)
        model = self._select_model_conservative(self._get_objective_for_model(col))
        model.fit(temp_data.drop(col, axis = 1), temp_data[col])
        imp_array = model.feature_importances_
        imp_by_name = dict(zip(temp_data.drop(col, axis = 1).columns, imp_array))
        sorted_imp = dict(sorted(imp_by_name.items(), key = lambda x: x[1],reverse = True))
        top_feats = list(sorted_imp.keys())[:self.n_feats]
        return top_feats

    def _fill_nans_with_string(self, data_prepared, col, col_nan_ix):
        """Fill missing values in text column with 'Missing_data' string value.

        Applicable to object type columns with over 500 unique values (considered as text).

        This function will fill the data_prepared (a factorized dataframe) with
        a new integer value in the col_nan_ix locations. As this function is
        executed within a multiprocess, it can not update the
        dict(self.encoding_map) instance attribute with new key,value pair
        for this column (E.g.{'Missing_data':999}).
        The dict(self.encoding_map) will be updated after the multiprocesses
        are finished by _update_encoding_map_with_fill_nans_with_string_result func.
        Actual insert of 'Missing_data' string value into the column will happen
        during final mapping of the data_prepared to the reversed(self.encoding_map)
        finalizing the impute() func.

        Args:
            data_prepared (pandas.DataFrame):
                factorized data including the column to fill in by 'Missing_data' value
            col (str):
                column to replace NaNs by 'Missing_data'
            col_nan_ix (list):
                list of indexes with NaNs in the column

        Returns:
            data_prepared[col] (pandas.Series):
                column with NaNs replaced by new encoding integer

        """
        new_code_for_category = max(list(self.encoding_map[col].values())) + 1
        data_prepared[col][col_nan_ix] = new_code_for_category
        data_prepared[col] = data_prepared[col].astype('int')
        print(f'Missing values in {col} replaced by "Missing_data" value')
        print('-'*50)
        return data_prepared[col]

    def _impute_nans_by_model(self, data_prepared, col, col_nan_ix, subset_train, subset_test):
        """Impute NaNs in column with machine learning.

            - Select an appropriate model
            - Use the self.conervative instance attribute for model selection
            - Fit/predict
            - Fill in the NaN indexes in col by predicted values

        Args:
            data_prepared (pandas.DataFrame):
                factorized data including the column to fill in by 'Missing_data' value
            col (str):
                column to impute NaNs
            col_nan_ix (list):
                list of NaNs indexes in column
            subset_train (pandas.DataFrame):
                training sample for NaNs imputation in column
            subset_test (pandas.DataFrame):
                sample with missing values in the 'col' to predict on

        Returns:
            data_prepared[col] (pandas.Series):
                column with filled/imputed NaNs

        """
        if self.conservative:
            model = self._select_model_conservative(self._get_objective_for_model(col))
        else:
            model = self._select_model_complex(self._get_objective_for_model(col))
        model.fit(subset_train.drop(col, axis = 1), subset_train[col])

        predicted_nans = model.predict(subset_test.drop(col, axis = 1))
        data_prepared[col][col_nan_ix] = predicted_nans
        if self.verbose:
            print(f'- {col+":":<30} imputed {len(col_nan_ix)} NaNs')
        return data_prepared[col]

    def _fill_or_impute(self, data_prepared, col):
        """Fill col NaNs with 'Missing_data' factorization code or impute by ML.

        Logic:
            - If original column dtype == 'O' and it has over 500 unique values:
                this column is considered as text (not categorical).
                NaNs in this column will be replaced with 'Missing_data'
                factorization code
            - Else: NaNs will be imputed by ML

        Before imputing NaNs by ML subset data_prepared subset data by limiting
        columns (self._subset_columns) and rows (up to 20000).

        Args:
            data_prepared (pandas.DataFrame):
                factorized data including the column to fill in by 'Missing_data' value
            col (str):
                column to fill/impute NaNs

        Returns:
            data_prepared[col] (pandas.Series):
                column with filled/imputed NaNs

        """
        subset = self._subset_columns(data_prepared, col)
        col_nan_ix = subset[subset[col].isnull()].index
        subset_test = subset[subset.index.isin(col_nan_ix)]
        subset_train = subset.drop(col_nan_ix, axis = 0)

        if len(subset_train) > 20000:
            # make sure that all of the column classes (if classification) will stay in subset after sample
            if self.metadata[col]['nunique'] <= 70:
                subset_train = subset_train.groupby(col, group_keys=False).apply(lambda x: x.sample(min(len(x), 20000)))
            # or random sample (if regression)
            else:
                subset_train = subset.sample(n = 20000)

        if self.metadata[col]['dtype'] == 'O' and self.metadata[col]['nunique'] > 500:
            if self.fill_nans_in_pure_text:
                data_prepared[col] = self._fill_nans_with_string(data_prepared, col, col_nan_ix)
        else:
            data_prepared[col] = self._impute_nans_by_model(data_prepared, col, col_nan_ix, subset_train, subset_test)

        return data_prepared[col]

    def _update_encoding_map_with_fill_nans_with_string_result(self, data_prepared, col):
        '''Update encoding_map with 'Missing_values' category if it had been
        introduced in the 'pure text' columns.

        Before mapping the data_prepared to encoding_map check if the imputed
        cols in data_prepared include extra values that are not present in
        the encoding_map values for this column.
        This could be the case if _fill_nans_with_string() had been
        utilized within one of the multoprocesses, which introduced the
        new category "Missing_data" and could not append the class instance
        attribute self.encoding_map for this column from a multiprocess.

        Args:
            data_prepared (pandas.DataFrame):
                imputed data, factorized, unreverted
            col (str):
                column name check.

        Returns:
            None

        '''
        encoding_map_reversed = {val:key for key, val in self.encoding_map[col].items()}
        diff = list(set(data_prepared[col].unique()) - set(np.array(list(encoding_map_reversed.keys()))))
        if len(diff) > 0:
            if not np.isnan(diff[0]):
                new_code_for_category = diff[0]
                self.encoding_map[col]['Missing_data'] = new_code_for_category

    def _drop_cols_with_all_nans(self, data_prepared):
        """Drop columns with all missing values.

        Remove such columns from self.nan_cols.

        Args:
            data_prepared (pandas.DataFrame):
                Data to drop empty columns

        Returns:
            data_prepared (pandas.DataFrame):
                Data without empty columns

        """
        for col in data_prepared:
            if np.all(data_prepared[col].isnull()):
                data_prepared.drop(col, axis = 1, inplace = True)
                self.nan_cols.remove(col)
                self.droped_cols.append(col)
                if self.verbose:
                    print(f'Droped column {col} with all NaNs')
                    print('-'*50)

        return data_prepared

    def _drop_nan_constant_cols(self, data_prepared):
        """Drop columns with all missing values and all other constant vals.

        Remove such columns from self.nan_cols.

        Args:
            data_prepared (pandas.DataFrame):
                Data to drop columns witn NaNs and all other constant vals

        Returns:
            data_prepared (pandas.DataFrame):
                Data without NaN cols and all other constant vals

        """
        for col in data_prepared:
            if np.any(data_prepared[col].isnull()):
                if data_prepared[col].nunique() == 1:
                    data_prepared.drop(col, axis = 1, inplace = True)
                    self.nan_cols.remove(col)
                    self.droped_cols.append(col)
                    if self.verbose:
                        print(f'Droped column {col} with NaNs and all other constants')
                        print('-'*50)
        return data_prepared

    def impute(self, data):
        """Impute missing values in dataset.

        Args:
            data (pandas.DataFrame):
                Data to impute missing values in

        Returns:
            data (pandas.DataFrame):
                Data with imputed missing values
            or
            data (pandas.DataFrame):
                original data if there are no missing values to impute

        """
        start = timeit.default_timer()
        # check for actual presence of nans
        if ( not self.nan_cols and not np.any(data.isnull()) ) or ( not not self.nan_cols and not np.any(data[self.nan_cols].isnull()) ):
            print('\nNo missing data to impute')
            return data
        else:
            if self.verbose:
                self._print_data_dims(data)
            data = self._prepare_data(data)

            # deal with NaN cols with constant and empty_cols
            if self.drop_nan_cols_with_constant:
                data = self._drop_nan_constant_cols(data)
#            self._skip_cols_with_constant()
            if self.drop_empty_cols:
                data = self._drop_cols_with_all_nans(data)

            # impute in multiprocessing mode
            if self.multiprocessing_load > 1:
                print(f'\nDeploy multiprocessing with {self.num_workers} parallel proceses\n')
                with concurrent.futures.ProcessPoolExecutor(max_workers = self.num_workers) as executor:
                    results = [executor.submit(self._fill_or_impute, data, col) for col in self.nan_cols]
                # extract imputed cols from multiprocessing results
                for f in concurrent.futures.as_completed(results):
                    try:
                        imputed_col = f.result()
                        data[imputed_col.name] = imputed_col
                    except:
                        continue

            # impute on a single core
            else:
                print('\nImpute sequentially on a single core\n')
                for col in self.nan_cols:
                    try:
                        data[col] = self._fill_or_impute(data, col)
                    except:
                        continue

            # map to revresed encoding_map dict
            for col in list(self.encoding_map.keys()):
                if col in data:
                    self._update_encoding_map_with_fill_nans_with_string_result(data, col)
                    data[col] = data[col].map({val:key for key, val in self.encoding_map[col].items()})
            stop = timeit.default_timer()
            if self.verbose:
                if len(self.droped_cols) > 0:
                    print(f'\nDroped {len(self.droped_cols)} columns with either all NaNs or with NaNs and all other constants')
                print(f'\nNaNs imputation time: {np.round((stop-start)/60,2)} minutes')
                print('-'*50)
            return data