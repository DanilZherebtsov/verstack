import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from verstack.tools import Printer
from verstack.tools import timer
# SUBSET DATA, AND MAYBE OTHERS MOVE TO nova.common.py script

class FeatureSelector:  

    __version__ = '0.0.8'

    def __init__(self, **kwargs):

        '''
        Initialize FeatureSelectorRFE instance.

        Parameters
        ----------
        objective : str, optional
            Training objective. Can take values like 'regression', any other value is considered as 'classification'. 
            The default is 'regression'.
        auto : bool, optional
            Verbose flag for short informative messages. The default is True.
            fit_transform has it's own 'verbosity' setting which controlls the 
            verbosity of sklearn.feature_selection.RFECV.
        allowed_score_gap : float, optional
            Auto mode setting: percent difference in CV score on features selected 
            by [RFE with linear model] and by [RFE with RF model]. If the the percent 
            difference in scores (E.g. 0.03) is lower than allowed_score_gap (E.g. 0.05),
            then a smaller final set of features is chosen even if the scoring result is
            marginally lower.
        final_scoring_model : class instance, optional
            Model for cross_val_score of the two sets of features that were indepently
            selected by RFECV with linear model and with RF model. The default is None.
            If None, Lightgbm model will be applied.
        default_model_linear : bool, optional
            Type of default model to use. If True, will deploy LogisticRegression/Ridge, if False will deploy RandomForest.
            The default is False.
        custom_model : class instance, optional
            model class instance that must contain train/predict/predict_proba metods. 
            If not passed, defalult model will be used. The default is None.
        subset_size_mb : int, optional
            Maximum size of data subset for feature selection experiments. 
            Speeds up the feature selection for the large datasets. The default is 20.
        verbose : bool, optional
            Verbose flag for short informative messages. The default is True.
            fit_transform has it's own 'verbosity' setting which controlls the 
            verbosity of sklearn.feature_selection.RFECV.
        random_state : int, optional
            Random state for reproducibility. The default is None.
        Returns
        -------
        None.

        '''       
        self.objective = kwargs.get('objective')
        self.auto = kwargs.get('auto', False)
        self.allowed_score_gap = kwargs.get('allowed_score_gap', 0.0)
        self.final_scoring_model = kwargs.get('final_scoring_model', None)
        self.default_model_linear = kwargs.get('default_model_linear', False)
        self.custom_model = kwargs.get('custom_model', None)
        self.subset_size_mb = kwargs.get('subset_size_mb', 20)
        self.verbose = kwargs.get('verbose', True)
        self.random_state = kwargs.get('random_state', None)
        self._model = self._initialise_model(self.custom_model, self.default_model_linear)
        self.selected_features = None
        self._score_diff_percent = None # placeholder for print results in the console 
        self.printer = Printer(verbose=self.verbose)

    def __repr__(self):
        return f'FeatureSelectorRFE(objective: {self.objective}\
            \n                   model: {self._model}\
            \n                   auto: {self.auto}\
            \n                   allowed_score_gap: {self.allowed_score_gap}\
            \n                   subset_size_mb: {self.subset_size_mb}\
            \n                   random_state: {self.random_state}\
            \n                   verbose: {self.verbose})'
        
    # Validate init arguments
    # =========================================================================
    # define cmmon functions for setters
    def _is_bool(self, val):
        return type(val) == bool
    
    def _is_float(self, val):
        return type(val) == float

    def _is_int(self, val):
        return type(val) == int
    # -------------------------------------------------------------------------
    # objective
    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        if type(value) != str : raise TypeError('objective must be a string: "regression" or "classification"')
        self._objective = value
    # -------------------------------------------------------------------------
    # default_model_linear
    @property
    def default_model_linear(self):
        return self._default_model_linear

    @default_model_linear.setter
    def default_model_linear(self, value):
        if not self._is_bool(value) != str : raise TypeError('default_model_linear must be a boolean value: True or False')
        self._default_model_linear = value
    # -------------------------------------------------------------------------
    # custom_model
    @property
    def custom_model(self):
        return self._custom_model
    
    @custom_model.setter
    def custom_model(self, value):
        if value is not None and not hasattr(value, '__dict__'):
            print(f'{value} is not a valid model, it must be a class instance, setting default model')
            self._custom_model = None
        else:
            self._custom_model = value
    # -------------------------------------------------------------------------
    # subset_size_mb
    @property
    def subset_size_mb(self):
        return self._subset_size_mb
    
    @subset_size_mb.setter
    def subset_size_mb(self, value):
        if not self._is_int(value):
            print(f'{value} is not a valid subset_size_mb argument, must be a positive integer, setting to 20')
            self._subset_size_mb = 20
        else:
            if value <= 0:
                print('subset_size_mb must be a positive integer, setting to 20')
                self._subset_size_mb = 20
            else:
                self._subset_size_mb = value
    # -------------------------------------------------------------------------
    # verbose
    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        if not self._is_bool(value):
            print(f'{value} is not a valid verbose argument, must be a bool, setting to True')
            self._verbose = True
        else:
            self._verbose = value
    # -------------------------------------------------------------------------
    # verbose
    @property
    def random_state(self):
        return self._random_state
    
    @random_state.setter
    def random_state(self, value):
        if not self._is_int(value) and value is not None:
            print(f'{value} is not a valid random_state argument, must be an int or None. Setting default random_state to None')
            self._random_state = None
        else:
            self._random_state = value
    # -------------------------------------------------------------------------
    # auto
    @property
    def auto(self):
        return self._auto
    
    @auto.setter
    def auto(self, value):
        if not self._is_bool(value):
            print(f'{value} is not a valid auto argument, must be a bool, setting to False')
            self._auto = True
        else:
            self._auto = value
    # -------------------------------------------------------------------------
    # final_scoring_model
    @property
    def final_scoring_model(self):
        return self._final_scoring_model
    
    @final_scoring_model.setter
    def final_scoring_model(self, value):
        if value is not None and not hasattr(value, '__dict__'):
            print(f'{value} is not a valid model, it must be a class instance, setting default model, setting to default LGBM')
        else:
            self._final_scoring_model = value
    # -------------------------------------------------------------------------
    # allowed_score_gap
    @property
    def allowed_score_gap(self):
        return self._allowed_score_gap 
    
    @allowed_score_gap.setter
    def allowed_score_gap(self, value):
        if not self._is_float(value):
            print(f'{value} is not a valid auto argument, must be a float, setting to 0.0')
            self._allowed_score_gap = 0.0
        else:
            self._allowed_score_gap = value
    # =========================================================================
    def _initialise_model(self, custom_model, default_model_linear):
        '''Select LogisticRegression/Ridge or RandomForestRegressor/Classifier 
        if no model instance is passed, or use passed model instance'''
        if custom_model is None:
            if default_model_linear:
                return self._get_linear_model()
            else:
                return self._get_randomforest_model()
        else:
            return custom_model
    
    @timer
    def fit_transform(self, X, y, **kwargs):
        '''
        Apply sklearn.feature_selection.RFECV using a model that was defined at init

        Parameters
        ----------
        X : pd.DataFrame
            train features.
        y : pd.Series
            train labels.
        kwargs : optional
            keyword arguments for sklearn.feature_selection.RFECV
            
        Returns
        -------
        pd.DataFrame
            selected train features.

        '''
        self._validate_pandas(X)
        self.printer.print('Initiating FeatureSelector', order=1)
        if self.auto:
            self.printer.print(f'Comparing LinearRegression and RandomForest for feature selection', order = 2)
            self._auto_linear_randomforest_selector(X, y, kwargs)
        else:
            self.printer.print(f'Running feature selection with {self._model}', order = 2)
            selector = self._get_selector(self._model, y, kwargs)
            selected_feats_flags = self._prepare_data_apply_selector(X, y, selector, scale_data = self.default_model_linear)
            self.selected_features = X.columns[selected_feats_flags]
        self.printer.print(f'Selected {len(self.selected_features)} features from {X.shape[1]}', order=3)
        return X[self.selected_features]

    def _validate_pandas(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Invalid argument for X, must be a pandas.DataFrame')

    def _prepare_data_apply_selector(self, X, y, selector, scale_data = False):
        X_subset, y_subset = self._subset_data(X, y)
        if scale_data:
            X_subset = self._scale_data(X_subset)
        try:
            X_subset, y_subset = self._transform_data_to_float_32(X_subset, y_subset)
            selector.fit(X_subset, y_subset)
        except BrokenPipeError:
            print('Experienced BrokenPipeError, restarting')
            selector.fit(X_subset, y_subset)
        feats = selector.support_
        return feats

    def _auto_linear_randomforest_selector(self, X, y, kwargs):
        '''
        Automated feature selection by both built in models LinearRegression & RandomForest.
        
        Both types of models are run independently with RFECV, the resulting sets of features
        are then scored with 2-fold CV by an independent model. Final scoring model can
        be configured at init by passing the model instance to final_scoring_model argument (Lightgbm by default).
        Resulting features are selected taking into account the validation score and 
        allowed_score_gap argument.

        Parameters
        ----------
        X : pd.DataFrame
            train features
        y : pd.Series
            train labels
        kwargs : keyword arguments
            arguments for sklearn.feature_selection.RFECV

        Returns
        -------
        None.

        '''
        from sklearn.model_selection import cross_val_score as cvs

        linear_model = self._get_linear_model()
        randomforest_model = self._get_randomforest_model()
        
        selector_lr = self._get_selector(linear_model, y, kwargs)
        selector_rf = self._get_selector(randomforest_model, y, kwargs)
        
        self.printer.print(f'Running feature selection with {linear_model}', order = 2)
        feats_lr_flags = self._prepare_data_apply_selector(X, y, selector_lr, scale_data = True)

        self.printer.print(f'Running feature selection with {randomforest_model}', order = 2)
        feats_rf_flags = self._prepare_data_apply_selector(X, y, selector_rf, scale_data = False)
        
        scoring = self._get_sklearn_scoring_func_name(y)
        model = self._get_final_scoring_model(y)        
        self.printer.print(f'Scoring selected feats from linear and RF models by: {model}', order = 2)
        score_lr = np.mean(cvs(model, X[X.columns[feats_lr_flags]], y, scoring = scoring, cv = 3))
        score_rf = np.mean(cvs(model, X[X.columns[feats_rf_flags]], y, scoring = scoring, cv = 3))
        self.printer.print(f'RFE by linear model cv-score       : {np.round(score_lr,5)}', order = 4, leading_blank_paragraph=True)
        self.printer.print(f'linear model n selected features   : {len(feats_lr_flags[feats_lr_flags])}', order = 4)
        self.printer.print(f'RFE by RandomForest model cv-score : {np.round(score_rf,5)}', order = 4, leading_blank_paragraph=True)
        self.printer.print(f'RF model n selected features       : {len(feats_rf_flags[feats_rf_flags])}', order = 4)
        
        results_comparison_dict = {
            'linear':{
                'score':score_lr,
                'n_feats':len(feats_lr_flags[feats_lr_flags]),
                'feats_flags':feats_lr_flags
                },
            'RF':{
                'score':score_rf,
                'n_feats':len(feats_rf_flags[feats_rf_flags]),
                'feats_flags':feats_rf_flags
                }
            }      

        selected_model = self._select_based_on_dimensions(results_comparison_dict)
        selected_feats_flags = list(selected_model.values())[0]['feats_flags']
        choice_explanation_message = '' if not self._score_diff_percent else f'\n(validation on these feats is {np.round(self._score_diff_percent,5)}% less accurate but within {self.allowed_score_gap}% allowance)'
        self.printer.print(f'Keeping feats from RFE with {list(selected_model.keys())[0]} model {choice_explanation_message}', order = 3, leading_blank_paragraph=True, breakline='-')
        self.selected_features = X.columns[selected_feats_flags]
    
    def _get_final_scoring_model(self, y):
        '''Set up the default LGBM (regressor or classifier) or use the user defined model'''
        if self.final_scoring_model is None:
            from lightgbm import LGBMClassifier, LGBMRegressor
            verbosity = -1
            if self.objective == 'regression':
                model = LGBMRegressor(verbosity=verbosity, n_jobs = -1, 
                                      random_state = self.random_state)
            else:
                if len(np.unique(y)) == 2:
                    model = LGBMClassifier(objective = 'binary', verbosity=verbosity, 
                                           n_jobs = -1, random_state = self.random_state)
                else:
                    model = LGBMClassifier(objective = 'multiclass', verbosity=verbosity, 
                                           n_jobs = -1, random_state = self.random_state)
        else:
            model = self.final_scoring_model
        return model
        
    # -------------------------------------------------------------------------
    def _select_based_on_dimensions(self, results_comparison_dict):
        '''Header function for features selection from linear & RF feature elimination results'''
        if self.allowed_score_gap == 0:
            selected_model = self._get_result_with_highest_score(results_comparison_dict)
        else:
            selected_model = self._get_results_based_on_allowance(results_comparison_dict)
        return selected_model

    def _get_result_with_highest_score(self, results_comparison_dict):
        '''Return results_comparison_dict subset with the best score'''
        temp_comparison_dict = {}
        for key in results_comparison_dict.keys():
            temp_comparison_dict[key] = results_comparison_dict[key]['score']

        best_score_model_name = max(temp_comparison_dict, key=temp_comparison_dict.get)
        selected_model = {}
        selected_model[best_score_model_name] = results_comparison_dict[best_score_model_name]
        return selected_model

    def _get_results_based_on_allowance(self, results_comparison_dict):
        '''Header function for comparing the scores of both models, number of 
        selected feats and percent difference allowance'''
        selected_model = self._return_model_with_less_feats_and_best_score(results_comparison_dict)
        if len(selected_model.keys()) > 1:
            selected_model = self._extract_model_with_less_feats_if_within_allowance(selected_model)            
        return selected_model            

    def _return_model_with_less_feats_and_best_score(self, results_comparison_dict):
        '''If a model with higher score and less n_feats is yielded return only this model, 
        else return both models'''
        models_names = list(results_comparison_dict.keys())
        temp_comparison_dict = {}
        temp_comparison_dict[models_names[0]] = {}
        temp_comparison_dict[models_names[0]]['smallest_n_feats'] = results_comparison_dict[models_names[0]]['n_feats'] < results_comparison_dict[models_names[1]]['n_feats']
        temp_comparison_dict[models_names[0]]['highest_score'] = results_comparison_dict[models_names[0]]['score'] > results_comparison_dict[models_names[1]]['score']
        temp_comparison_dict[models_names[1]] = {}
        temp_comparison_dict[models_names[1]]['smallest_n_feats'] = results_comparison_dict[models_names[1]]['n_feats'] < results_comparison_dict[models_names[0]]['n_feats']
        temp_comparison_dict[models_names[1]]['highest_score'] = results_comparison_dict[models_names[1]]['score'] > results_comparison_dict[models_names[0]]['score']
        for key, value in temp_comparison_dict.items():
            if np.all(list(value.values())):
                selected_model = {}
                selected_model[key] = results_comparison_dict[key]
                return selected_model
        return results_comparison_dict
        
    def _extract_model_with_less_feats_if_within_allowance(self, selected_model):
        '''Return model with less feats and lower score if score difference percent is within allowance'''
        models_names = list(selected_model.keys())
        temp_comparison_dict = {}
        temp_comparison_dict[models_names[0]] = {}
        temp_comparison_dict[models_names[0]]['lower_score'] = selected_model[models_names[0]]['score'] < selected_model[models_names[1]]['score']
        temp_comparison_dict[models_names[1]] = {}
        temp_comparison_dict[models_names[1]]['lower_score'] = selected_model[models_names[1]]['score'] < selected_model[models_names[0]]['score']
        for key, value in temp_comparison_dict.items():
            if True in value.values():
                lower_score_model = key
            else:
                lower_score_model = self._get_model_with_less_feats_if_scores_are_equal(selected_model)                
        higher_score_model = [key for key in selected_model.keys() if key != lower_score_model][0]
        score_diff_percent = abs(selected_model[lower_score_model]['score'] / selected_model[higher_score_model]['score'] - 1)

        final_model = {}
        if score_diff_percent < self.allowed_score_gap:
            final_model[lower_score_model] = selected_model[lower_score_model]
            self._score_diff_percent = score_diff_percent
        else:
            final_model[higher_score_model] = selected_model[higher_score_model]
        return final_model

    def _get_model_with_less_feats_if_scores_are_equal(self, selected_model):
        '''Return model name with less n_feats if scores are identical'''
        min_n_feats = min(k['n_feats'] for k in selected_model.values()) 
        model_with_min_n_feats = [k for k in selected_model if selected_model[k]['n_feats'] == min_n_feats]
        return model_with_min_n_feats[0]
    
    # -------------------------------------------------------------------------
    def _get_selector(self, model, y, kwargs):
        '''Configure sklearn.feature_selection.RFECV based on target variable and kwargs'''
        final_kwargs = self._set_RFECV_kwargs_from_user_input_and_defaults(kwargs, y)
        selector = RFECV(model, **final_kwargs)
        return selector

    def _set_RFECV_kwargs_from_user_input_and_defaults(self, kwargs, y):
        '''Use user input kwargs for RFECV along with available kwargs for current version of sklearn
        and FeatureSelector default values for kwargs based on sklearn 1.0.1 version to define
        final set of arguments for RFECV. (used to set up RFECV for sklearn 1.0.1 and earlier versions)
        
        '''
        # define defaults for all available kwargs for RFECV in 1.0.1 sklearn version
        sklearn_101_default_kwargs = { 
            'step':1,
            'min_features_to_select':5,
            'cv':2,
            'n_jobs':-1,
            'importance_getter':'auto',
            'scoring':self._get_sklearn_scoring_func_name(y),
            'verbose':0
            }
        available_kwargs = self._get_available_RFECV_params()
        final_kwargs = {}
        for arg in available_kwargs:
            final_kwargs[arg] = kwargs.get(arg, sklearn_101_default_kwargs[arg])
        return final_kwargs

    def _get_available_RFECV_params(self):
        '''Get names of keyword arguments for the user's sklearn.feature_selection.RFECV version'''
        import inspect
        available_args = str(inspect.signature(RFECV)).strip('(').strip(')').split(',')
        available_args = [x for x in available_args if '=' in x]
        available_args = [x.split('=')[0] for x in available_args]
        available_args = [x.replace(' ','') for x in available_args]
        return available_args    
    
    # -------------------------------------------------------------------------
    def _get_linear_model(self):
        '''Return Ridge() or LogisticRegression() from sklearn'''
        if self.objective == 'regression':
            from sklearn.linear_model import Ridge
            return Ridge(random_state = self.random_state)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state = self.random_state)
        
    def _get_randomforest_model(self):
        '''Return RFClassifier or RFRegressor from sklearn'''
        if self.objective == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=50, max_depth=2, n_jobs=-1, 
                                         random_state = self.random_state)
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=50, max_depth=2, n_jobs=-1, 
                                          random_state = self.random_state)
        
    def _subset_data(self, X, y):
        '''Subset train dataset by n megabytes.
        Used to speed up experiments during feature engineering & model tuning.'''
        temp = pd.concat([X, pd.Series(y, name='target')], axis=1)
        data_size = np.round(temp.memory_usage().sum()/(1024*1024), 2)
    
        if data_size > self.subset_size_mb:
            batch = self.subset_size_mb / data_size
            experimental_data = temp.sample(frac=batch)
            experimental_data.reset_index(drop=True, inplace=True)
            X = experimental_data.drop('target', axis=1)
            y = experimental_data.target
            self.printer.print(f'Data decreased for experiments. Working with {np.round(batch*100,2)}% of data', order = 3)
            del experimental_data
        else:
            self.printer.print('Experiments are carried out on complete dataset', order = 3)
        del temp, data_size
        return X, y  

    def _scale_data(self, X):
        '''Normalize data'''
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def _transform_data_to_float_32(self, X_subset, y_subset):
        '''Transform to float32, drop rows with np.nan, np.inf'''
        X_subset = pd.DataFrame(X_subset).astype('float32')
        # Replace infinite updated data with nan
        X_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaN
        X_subset.dropna(inplace=True)       
        y_subset = pd.Series(y_subset).loc[X_subset.index]
        return X_subset, y_subset

    def _get_sklearn_scoring_func_name(self, y):
        '''Return metric name for scoring the feature selection'''
        if self.objective == 'regression':
            return 'neg_root_mean_squared_error'
        else:
            if len(np.unique(y)) == 2:
                return 'roc_auc'
            else:
                return 'neg_log_loss'

    def transform(self, test):
        '''Subset data by features that were selected at fit_transform()'''
        self._validate_pandas(test)
        self.printer.print('Applying RFECV (FeatureSelector)', order=1)
        self.printer.print(f'Selected {len(self.selected_features)} features from {test.shape[1]}', order=3)
        return test[self.selected_features]