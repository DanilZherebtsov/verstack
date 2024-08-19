import operator
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from copy import copy
from functools import partial
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from verstack.tools import timer
from verstack.tools import Printer
from verstack.lgbm_optuna_tuning.lgb_metrics import get_pruning_metric, get_optimization_metric_func, define_objective, classification_metrics, regression_metrics, get_eval_score, print_lower_greater_better, supported_metrics
from verstack.lgbm_optuna_tuning.args_validators import *
from verstack.lgbm_optuna_tuning.optuna_tools import Distribution, OPTUNA_DISTRIBUTIONS_MAP, SearchSpace
# BACKLOG: add option to pass different init params on class __init__
# BACKLOG: add option to pass different param_grid for optimization
# OPTIONAL TODO - add factorization to targets and inverse_transform predictions for classification (E.g. Ghosts, prudential)

supported_gridsearch_params = [
    'boosting_type', 'num_iterations', 'learning_rate', 'num_leaves', 'max_depth', 
    'min_data_in_leaf', 'min_sum_hessian_in_leaf', 'bagging_fraction', 'feature_fraction',
    'max_delta_step', 'lambda_l1', 'lambda_l2', 'linear_lambda', 'min_gain_to_split',
    'drop_rate', 'top_rate', 'min_data_per_group', 'max_cat_threshold'
    ]

class LGBMTuner:

    __version__ = '1.4.2'

    def __init__(self, **kwargs):
        '''
        Class to automatically tune LGBM model with optuna.
        
        Model type (regressor/classifier) is inferred based on target variable & metric.
        Init parameters and search space are inferred by built in logic.

        Parameters
        ----------
        metric : str
            evaluation metric name.
        trials : int, optional
            number of hyperparameter search trials. The default is 200.
        refit : bool, optional
            Flag to refit the model with optimized parameters on whole dataset. 
            The default is True.
        verbosity : int, optional
            console verbosity level: 0 - no output except for optuna CRITICAL errors and builtin exceptions; 
            (1-5) based on optuna.logging options. The default is 1.
        visualization : bool, optional
            flag to print optimization & feature importance plots. The default is True.
        seed : int
            random_state.
        device_type : str
            cpu/gpu/cuda/cuda_exp.
        custom_lgbm_params : dict
            custom lgbm parameters to be passed to the model, please refer to LGBM documentation for available parameters.
        eval_results_callback : func
            callback function to be applied on the eval_results dictionary that is being populated
            with evaluation metric score upon completion of each training trial.            
            Example: 
                def log_results_callback(results): 
                    # save results to disk
                    with open('eval_results.json', 'w') as f:
                        json.dump(results, f)
                
                lgbm_tuner = LGBMTuner(metric = 'accuracy', eval_results_callback = log_results_callback)

        stop_training_callback : func
            callback function to stop training based on a condition. 
            Example: 
                def stop_callback(): 
                    # stop training if variable value in file is changed
                    with open('stop_training.txt', 'r') as f:
                        if eval(f.read()):
                            return True
                    return False

                lgbm_tuner = LGBMTuner(metric = 'accuracy', stop_training_callback = stop_callback)

        grid: dict
            Parameters search space for optimization. 
            
            A default search space is populated at fit, depending on number of trials:
            "feature_fraction" = (Distribution.UNIFORM, low=0.5, high=1.0) 
            "num_leaves" = (Distribution.INTUNIFORM, low=16, high=255)
            if self.trials > 30:
                "bagging_fraction" = (Distribution.UNIFORM, low=0.5, high=1.0) 
                "min_sum_hessian_in_leaf" = (Distribution.LOGUNIFORM, low=1e-3, high=10.0)
            if self.trials > 100:
                "lambda_l1" = (Distribution.LOGUNIFORM, low=1e-8, high=10.0)
                "lambda_l2" = (Distribution.LOGUNIFORM, low=1e-8, high=10.0)

            Grid can be modified with the following parameters to be included in the Search Space:
                boosting_type : 'gbdt', 'dart', 'rf' 
                num_iterations : >= 0
                learning_rate : > 0.0
                num_leaves : 1 < num_leaves <= 131072
                max_depth : 
                min_data_in_leaf : >= 0
                min_sum_hessian_in_leaf : >= 0.0
                bagging_fraction : 0.0 < bagging_fraction <= 1.0
                feature_fraction : 0.0 < feature_fraction <= 1.0
                max_delta_step : 
                lambda_l1 : >=0
                lambda_l2 : >=0
                linear_lambda : >=0
                min_gain_to_split : >=0.0
                drop_rate : 0.0 <= drop_rate <= 1.0
                top_rate : 0.0 <= top_rate <= 1.0
                min_data_per_group : > 0
                max_cat_threshold : > 0

        Returns
        -------
        None.

        '''
        self.verbosity = kwargs.get('verbosity', 1)
        self.verbose = True if self.verbosity > 0 else False
        self.printer = Printer(self.verbose)
        self.metric = kwargs.get('metric')
        self.trials = kwargs.get('trials', 100)
        self.refit = kwargs.get('refit', True)
        self.visualization = kwargs.get('visualization', True)
        self.seed = kwargs.get('seed', 42)
        self.device_type = kwargs.get('device_type', 'cpu')
        self.custom_lgbm_params = kwargs.get('custom_lgbm_params', {})
        self.target_minimum = None
        self._fitted_model = None
        self._feature_importances = None
        self._study = None # save optuna study for plotting
        self.target_classes = None
        self._init_params = None
        self._best_params = None
        self.eval_results = {} # evaluation metric results per each trial storage
        self.eval_results_callback = kwargs.get('eval_results_callback', None)
        self.stop_training_callback = kwargs.get('stop_training_callback', None)
        self.search_space = self._get_default_search_space()
        self.grid = self._get_all_available_and_defined_grids()
        self.early_stopping_results = {} # stores early_stopping results per each trial

        # get user defined grid

    def _get_all_available_and_defined_grids(self):
        all_grids = {}
        for param in supported_gridsearch_params:
            if param in self.search_space:
                all_grids[param] = self.search_space[param].params
            else:
                all_grids[param] = None
        return all_grids
        

    # print init parameters when calling the class instance
    def __repr__(self):
        return f'LGBMTuner(Evaluation metric: {self._metric}\
            \n          trials: {self.trials}\
            \n          refit: {self.refit}\
            \n          verbosity: {self.verbosity}\
            \n          visualization: {self.visualization})\
            \n          device_type: {self.device_type})\
            \n          grid: {self.grid})'
    

    # Validate init arguments
    # =========================================================================
    # define cmmon functions for setters
    def _is_bool(self, val):
        return type(val) == bool

    def _is_int(self, val):
        return type(val) == int
    # -------------------------------------------------------------------------
    # metric
    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        if type(value) != str : raise TypeError('metric must be a string')
        if value not in supported_metrics: raise KeyError(f'LGBMTuner supports the following evaluation metrics: {supported_metrics}')
        self._metric = value
    # -------------------------------------------------------------------------
    # trials
    @property
    def trials(self):
        return self._trials

    @trials.setter
    def trials(self, value):
        if not self._is_int(value) : raise TypeError('trials must be an integer')
        if value <= 0: raise ValueError('trials must be an integer > 0')
        self._trials = value
    # -------------------------------------------------------------------------
    # refit
    @property
    def refit(self):
        return self._refit

    @refit.setter
    def refit(self, value):
        if not self._is_bool(value) : raise TypeError('acceptable refit options are True/False')
        self._refit = value
    # -------------------------------------------------------------------------
    # verbosity
    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if not self._is_int(value) : raise TypeError('verbosity must be an integer')
        if value not in range(0,6) : raise ValueError('acceptable verbosity options are [0,1,2,3,4,5]')
        self._verbosity = value
    # -------------------------------------------------------------------------
    # visualization
    @property
    def visualization(self):
        return self._visualization

    @visualization.setter
    def visualization(self, value):
        if not self._is_bool(value) : raise TypeError('acceptable visualization options are True/False')
        self._visualization = value
    # -------------------------------------------------------------------------
    # device_type
    @property
    def device_type(self):
        return self._device_type

    @device_type.setter
    def device_type(self, value):
        if not value in ['cpu', 'gpu', 'cuda', 'cuda_exp'] : raise TypeError('acceptable device_type options are cpu/gpu/cuda/cuda_exp')
        self._device_type = value
    # -------------------------------------------------------------------------    
    # seed
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        if not self._is_int(value) : raise TypeError('seed must be an integer')
        self._seed = value
    # -------------------------------------------------------------------------    
    # only getters for the following arguments
    # -------------------------------------------------------------------------    
    # init_params
    @property
    def init_params(self):
        return self._init_params
    # -------------------------------------------------------------------------    
    # best_params
    @property
    def best_params(self):
        return self._best_params
    # -------------------------------------------------------------------------    
    # feature_importances
    @property
    def feature_importances(self):
        if not self._fitted_model:
            raise AttributeError('LGBMTuner.fit(refit = True) must be applied before feature_importances can be displayed')
        return self._feature_importances
    # -------------------------------------------------------------------------    
    # fitted_model
    @property
    def fitted_model(self):
        return self._fitted_model
    # -------------------------------------------------------------------------    
    # study
    @property
    def study(self):
        return self._study
    # -------------------------------------------------------------------------
    # eval_results_callback
    @property
    def eval_results_callback(self):
        return self._eval_results_callback

    @eval_results_callback.setter
    def eval_results_callback(self, value):
        if not value is None:
            if not hasattr(value, '__call__') : raise TypeError('eval_results_callback must be a function')
        self._eval_results_callback = value
    # -------------------------------------------------------------------------    
    def _init_params_on_input(self, rows_num, y):
        '''
        Get model parameters depending on dataset parameters.

        Parameters
        ----------
        rows_num : int
            number of rows in training data.
        y : pd.Series
            train labels

        Returns
        -------
        None.

        '''
        # TODO: use number of features
        
        # default lgbm parameters - basis for minor changes based on dataset length
        default_params_classification = {"task": "train",
                                        "learning_rate": 0.05,
                                        "num_leaves": 128,
                                        "feature_fraction": 0.7,
                                        "bagging_fraction": 0.7,
                                        "bagging_freq": 1,
                                        "max_depth": -1,
                                        "verbosity": -1,
                                        "lambda_l1": 1,
                                        "lambda_l2": 0.0,
                                        "min_split_gain": 0.0,
                                        "zero_as_missing": False,
                                        "max_bin": 255,
                                        "min_data_in_bin": 3,
                                        "num_iterations": 10000,
                                        "early_stopping_rounds": 100,
                                        "random_state": 42,
                                        "device_type": self.device_type}

        default_params_regression = {"learning_rate": 0.05,
                                    "num_leaves": 32,
                                    "feature_fraction": 0.9,
                                    "bagging_fraction": 0.9,
                                    "verbosity": -1,
                                    "num_iterations": 10000,
                                    "early_stopping_rounds": 100,
                                    "random_state": 42,
                                    "device_type": self.device_type}


        task = define_objective(self.metric, y)
        
        if task == 'regression':
            self._init_params = copy(default_params_regression)
        else:
            self._init_params = copy(default_params_classification)

        # define additional params based on task type
        if task == 'binary':
            self._init_params['num_classes'] = 1 
        elif task == 'multiclass':
            self._init_params['num_classes'] = y.nunique()
        # -------------------------------------------------------------------------
        # populate objective and metric based
        self._init_params['objective'] = define_objective(self.metric, self.target_classes)
        self._init_params['metric'] = get_pruning_metric(self.metric, self.target_classes)
        # -------------------------------------------------------------------------
        # do not use all available threads and make sure cpu_count is not negative
        cpu_count = int(np.where(os.cpu_count()-2 < 0, 1, os.cpu_count()-2))

        self._init_params['num_threads'] = cpu_count

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200
        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200
        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 1200
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            self._init_params["num_leaves"] = 128 if task == "regression" else 244
        elif rows_num > 100000:
            self._init_params["num_leaves"] = 64 if task == "regression" else 128
        elif rows_num > 50000:
            self._init_params["num_leaves"] = 32 if task == "regression" else 64
            # params['lambda_l1'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            self._init_params["num_leaves"] = 32 if task == "regression" else 32
            self._init_params["lambda_l1"] = 0.5 if task == "regression" else 0.0
        elif rows_num > 10000:
            self._init_params["num_leaves"] = 32 if task == "regression" else 64
            self._init_params["lambda_l1"] = 0.5 if task == "regression" else 0.2
        elif rows_num > 5000:
            self._init_params["num_leaves"] = 24 if task == "regression" else 32
            self._init_params["lambda_l1"] = 0.5 if task == "regression" else 0.5
        else:
            self._init_params["num_leaves"] = 16 if task == "regression" else 16
            self._init_params["lambda_l1"] = 1 if task == "regression" else 1

        self._init_params["learning_rate"] = init_lr
        self._init_params["num_iterations"] = ntrees

        # disable early stopping if 'scale_pos_weight' or 'is_unbalance' is used, because 
        # in such case with severe disbalance it always stops at first iteration and underfits
        if 'scale_pos_weight' in self._init_params or 'is_unbalance' in self._init_params:
            del self._init_params["early_stopping_rounds"]
        else:
            self._init_params["early_stopping_rounds"] = es
    # -----------------------------------------------------------------------------

    def _get_default_search_space(self):
        '''Sample hyperparameters from suggested

        Parameters
        ----------
        estimated_n_trials : int
            number of optuna trails.

        Returns
        -------
        search_space : dict
            optimization search space grid and type of distribution for each parameter.

        '''
        # TODO: create addditional options based on bigger estimated_n_trials
        search_space = {}

        search_space["feature_fraction"] = SearchSpace(Distribution.UNIFORM, low=0.5, high=1.0) 
        search_space["num_leaves"] = SearchSpace(Distribution.INTUNIFORM, low=16, high=255)

        if self.trials > 30:
            search_space["bagging_fraction"] = SearchSpace(Distribution.UNIFORM, low=0.5, high=1.0) 
            search_space["min_sum_hessian_in_leaf"] = SearchSpace(Distribution.LOGUNIFORM, low=1e-3, high=10.0)

        if self.trials > 100:
            search_space["lambda_l1"] = SearchSpace(Distribution.LOGUNIFORM, low=1e-8, high=10.0)
            search_space["lambda_l2"] = SearchSpace(Distribution.LOGUNIFORM, low=1e-8, high=10.0)

        return search_space
    # -----------------------------------------------------------------------------

    def _sample_from_search_space(self, trial):
        '''
        Get params for a trial.

        Parameters
        ----------
        trial : optuna.Trial
            trial object.
        suggested_params : dict
            init params on input.

        Raises
        ------
        ValueError
            Optuna distribution error.

        Returns
        -------
        trial_values : dict
            trial parameters consisting from suggested_params modified by grid.

        '''
        trial_values = copy(self._init_params)
        for parameter, SearchSpace in self.search_space.items():
            if SearchSpace.distribution_type in OPTUNA_DISTRIBUTIONS_MAP:
                trial_values[parameter] = getattr(trial, OPTUNA_DISTRIBUTIONS_MAP[SearchSpace.distribution_type])(
                    name=parameter, **SearchSpace.params)
            else:
                for key, value in self.search_space.items():
                    print(key)
                    print(value.distribution_type)
                raise ValueError(f"Optuna does not support distribution {SearchSpace.distribution_type}")
        return trial_values
    # -----------------------------------------------------------------------------

    def _get_dtrain_dvalid_objects(self, X, y, metric, seed = None, return_raw_valid = False):
        '''
        Create lgbm.Datasets for training and validation.
        
        By default splits without defined random_state in order to replicate CV (sort of).
        Seed is used for optimize_num_iterations.

        Parameters
        ----------
        X : pd.DataFrame
            train features.
        y : pd.Series
            train labels.
        metric : str
            evaluation metric name.
        seed : int/None
            random state for split
        return_raw_valid : bool
            Flag to return valid_x, valid_y in addition to dvalid (for get_validation_score())

        Returns
        -------
        dtrain : lgbm.Dataset
            train data.
        dvalid : ldgb.Dataset
            valid data.

        '''
        if metric in classification_metrics:
            train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25, stratify = y)
        else:
            train_x, valid_x, train_y, valid_y = train_test_split(X, y, random_state = seed, test_size=0.25)
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)
        if return_raw_valid:
            return dtrain, dvalid, valid_x, valid_y
        else:
            return dtrain, dvalid
    # -----------------------------------------------------------------------------
    def fit_optimized(self, X, y):
        '''
        Train model with tuned params on whole train data

        Parameters
        ----------
        X : np.array
            train features.
        y : np.array
            train target.

        Returns
        -------
        None.

        '''
        validate_numpy_ndarray_arguments(X)
        validate_numpy_ndarray_arguments(y)
        
        import lightgbm as lgb
        self.printer.print('Fitting optimized model with the follwing params:', order=2)
        if self.verbosity > 0:
            for key, value in self._best_params.items():
                print(f'{key:<33}: {value}')
        self._fitted_model = lgb.train(self._best_params, lgb.Dataset(X,y))

    # ------------------------------------------------------------------------------------------
    def _get_target_minimum(self, y):
        '''Record target minimum value for replacing negative predictions in regression.'''
        if self.metric in regression_metrics:    
            self.target_minimum = min(y)

    # ------------------------------------------------------------------------------------------
    def _get_validation_score(self, trial, dtrain, dvalid, valid_x, valid_y, optimization_metric_func, params, pruning_callback):
        '''
        Train model with trial params and validate on valid set against the defined metric.
        Print optimization result every iteration.
        If evaluation metric != optimization metric, print evaluation metric.

        Parameters
        ----------
        trial : optuna.Trial #1
            parameters tuning iteration.
        dtrain : lgb.Dataset 
            train data.
        dvalid : lgb.Dataset 
            valid data.
        valid_x : pd.DataFrame 
            valid features.
        valid_y : pd.Series 
            valid target.
        optimization_metric_func : func 
            scorer function.
        params : dict #1
            model parameters.
        pruning_callback : func
            callback function.

        Returns
        -------
        result : float
            validation result.

        '''
        gbm = lgb.train(params, dtrain, valid_sets=[dvalid], callbacks=[pruning_callback])
        pred = gbm.predict(valid_x)

        result = optimization_metric_func(valid_y, pred)

        optimization_direction = 'lower-better'

        self.printer.print(f'Trial number: {trial.number} finished', order=3)
        self.printer.print(f'Optimization score ({optimization_direction:<4}): {optimization_metric_func.__name__}: {result}', order=4)
        # save early stopping results per each trial for further use in best_params
        self.early_stopping_results[trial.number] = gbm.best_iteration
        # save evaluation metric results per each trial
        self.eval_results[f'train_trial_{trial.number}_out_of_{self.trials}'] = result
        # calculate & print eval_metric only if eval_metric != optimization_metric
        if self.metric != optimization_metric_func.__name__:
            eval_score = get_eval_score(valid_y, pred, self.metric, params['objective'])
            self.printer.print(f'Evaluation score ({print_lower_greater_better(self.metric):<4}): {self.metric}: {eval_score}', order=4)
            # save evaluation metric results per each trial
            self.eval_results[f'train_trial_{trial.number}_out_of_{self.trials}'] = eval_score
        self.printer.print(breakline='.')

        if self.eval_results_callback:
            self.eval_results_callback(self.eval_results)

        return result

    # ------------------------------------------------------------------------------------------
    def _objective(self, trial, X, y):
        '''
        Define objective for optuna tiral training.
        
        - Create optimization metric based on evaluatin metric passed by user
        - Create train/valid splits
        - Define suggested initial parameters for LGB model based on data and eveluation metric
        - Define the search space for some params within suggested parameters
        - Create final params dict based on suggested initial params and the a step in optimization search space grid
        - Define a pruning callback to stop training of unpromissed trails
        - Train a given trail & validate by optimization_metric_func

        Parameters
        ----------
        trial : optuna.Trial
            training step.
        X : pd.DataFrame
            train features.
        y : pd.Series
            train labels.
        metric : str
            evaluation metric name.

        Returns
        -------
        result : float
            optimization metric validation result.

        '''

        if self.stop_training_callback is not None:
            stop = self.stop_training_callback()
            if stop:
                print('STOPPING CALLBACK INITIALIZED')
                trial.study.stop()

        optimization_metric_func = get_optimization_metric_func(self.metric)
        dtrain, dvalid, valid_x, valid_y = self._get_dtrain_dvalid_objects(X, y, self.metric, return_raw_valid = True)
        params = self._sample_from_search_space(trial)
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, get_pruning_metric(self.metric, self.target_classes))
        result = self._get_validation_score(trial, dtrain, dvalid, valid_x, valid_y, optimization_metric_func, params, pruning_callback)

        return result 
    # ------------------------------------------------------------------------------------------
    def _check_refit_status(self, method):
        if not self.refit:
            raise ModuleNotFoundError(f'LGBMTuner.fit(refit = True) must be applied before {method} execution')
    # ------------------------------------------------------------------------------------------
    
    def predict(self, test, threshold = 0.5):
        '''
        Predict on test set with fitted model.
        
        For regression replace negative preds by self.target_minimum (if train labels
        were not negative).

        Parameters
        ----------
        test : pd.DataFrame
            test features.
        threshold : float, optional
            binary classification probability threshold. The default is 0.5.

        Returns
        -------
        pred : array
            predicted values.

        '''    
        self._check_refit_status('predict()')
        validate_features_argument(test)
        validate_threshold_argument(threshold)

        pred = self._fitted_model.predict(test.values, num_threads = self._init_params['num_threads'])
        
        # regression predict
        if self.metric in regression_metrics:
            if self.target_minimum >= 0:
                pred = np.where(pred < 0, self.target_minimum, pred)
            return pred
        else:
            # binary classification predict
            if self._fitted_model.params['objective'] == 'binary':
                pred = (pred > threshold).astype('int')
            # multiclass classification predict
            # TODO: SAVE CLASSES NAMES TO APPLY THEM BACK
            else:
                pred = np.argmax(pred, axis = 1)
            return pred
    # ------------------------------------------------------------------------------------------
    
    def predict_proba(self, test):
        '''
        Predict on test set (classification only) and return probabilities.

        Parameters
        ----------
        test : pd.DataFrame
            test features.

        Raises
        ------
        TypeError
            If self._fitted_model.params['objective'] == 'regression - notify
            that predict_proba() is for classification objectives only.

        Returns
        -------
        pred : array
            predicted values.

        '''
        self._check_refit_status('predict_proba()')
        validate_features_argument(test)

        if self.metric in regression_metrics:
            raise TypeError('predict_proba() is applicable for classification problems only')
        pred = self._fitted_model.predict(test.values, num_threads = self._init_params['num_threads'])
        return pred
    # ------------------------------------------------------------------------------------------
    
    def _populate_best_params_to_init_params(self, best_params):
        '''Populate the learned params into the suggested params'''
        # output params are temporary, because num_iterations tuning will follow
        temp_params = copy(self._init_params)
        for key, val in best_params.items():
            temp_params[key] = val
        # remove early_stopping & num_iterations from params (used during optuna optimization).
        # final early stopping will be trained during final_estimators_tuning
        if 'early_stopping_rounds' in temp_params:
            del temp_params['early_stopping_rounds']
        if 'num_iterations' in temp_params:
            del temp_params['num_iterations']        
        return temp_params
    # ------------------------------------------------------------------------------------------

    def _save_feature_importances(self, train_features):
        '''Save feature importances in class instance as a pd.Series'''
        feat_importances = pd.Series(self._fitted_model.feature_importance(), index = train_features)
        normalized_importances = np.round((lambda x: x/sum(x))(feat_importances),5)
        self._feature_importances = normalized_importances
    # ------------------------------------------------------------------------------------------

    def __configure_matplotlib_style(self,
                                     dark = True):
        '''Configure matplotlib style for plots'''
        try:
            styles = plt.style.available
            if dark:
                styles_to_set = ['dark_background']
                for style in styles:
                    if 'deep' in style:
                        styles_to_set.append(style)
            else:
                styles_to_set = []
                for style in styles:
                    if 'pastel' in style:
                        styles_to_set.append(style)
            for style in styles_to_set:
                plt.style.use(style)
        except Exception as e:
            print(f'Error while configuring matplotlib style: {e}')
            print('Default style will be used')
            pass
    # ------------------------------------------------------------------------------------------

    def _plot_static_fim(self,
                        feat_imp, 
                        figsize = (10,6), 
                        dark=True,
                        save=False,
                        display=True):
        """
        Plot feature importance as a horizontal bar chart.

        Parameters
        ----------
        feat_imp : pd.Series
            feature importance values.
        figsize : tuple, optional
            figure size. The default is (10,6).
        dark : bool, optional
            dark theme. The default is True.
        save : bool, optional
            save figure. The default is False.
        display : bool, optional
            display figure. The default is True.
        
        Returns
        -------
        None.
        
        """
        fig, ax = plt.subplots(figsize=figsize)
        plt.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        self.__configure_matplotlib_style(dark)

        if dark:
            name = 'FIM_DARK'
            ax.barh(feat_imp.index, feat_imp, alpha=0.8, color='#F99245')
            fig.set_facecolor('#20253c')
            ax.set_facecolor('#20253c')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title('Feature importances (sum up to 1)', color='white')
        else:
            name = 'FIM_LIGHT'
            ax.barh(feat_imp.index, feat_imp, alpha=0.8, color='#007bff')
            fig.set_facecolor('#dee0eb')
            ax.set_facecolor('#dee0eb')
            ax.tick_params(axis='x', colors='#212529')
            ax.tick_params(axis='y', colors='#212529')
            ax.set_title('Feature importances (sum up to 1)', color='#212529')
        if display:
            plt.show()
        if save:
            plt.savefig(f'{name}.png', dpi=300, facecolor=fig.get_facecolor(),
                        edgecolor='none', bbox_inches='tight')
            print(f'Feature Importance Plot is saved to {os.path.join(os.getcwd(), f"{name}.png")}')
    # ------------------------------------------------------------------------------------------
    def _plot_interactive_fim(self, plot_df, dark=True, save=False, display=True, plotly_fig_update_layout_kwargs={}):
        '''
        Create and save interactive plotly plot.
        
        Parameters
        ----------
        plot_df : pd.Series
            Series with features names and importances.
        dark : bool, optional
            Dark theme. The default is True.
        save : bool, optional
            Save plot to working directory. The default is False.
        display : bool, optional
            Display plot in console or in browser (for interactive plots). The default is True.
        plotly_fig_update_layout_kwargs : dict, optional
            Dictionary with plotly layout parameters. The default is {}.

        Returns
        -------
        None.
            
        '''
      
        BAR_COLOR = '#F99245'
        fig = px.bar(plot_df, 
                        x = 'importance', 
                        y = 'feature', 
                        orientation = 'h',
                        color_discrete_sequence = [BAR_COLOR]*len(plot_df),
                        text = 'importance_percent')
        default_plotly_dark_fig_update_layout_kwargs = {'plot_bgcolor':'#20253c', # plot color
                                                        'paper_bgcolor':'#2d3250', # html color
                                                        'font_color':'white',
                                                        'title_font_color':'lightgrey',
                                                        'xaxis':{'visible':False,
                                                                 'showticklabels':False,
                                                                 'showgrid':False},
                                                        'yaxis':{'showticklabels':True,
                                                                 'title':''}}
        default_plotly_light_fig_update_layout_kwargs = {'plot_bgcolor': '#dee0eb',
                                                         'paper_bgcolor': '#fff',
                                                         'font_color': '#34383d',
                                                         'title_font_color': 'lightgrey',
                                                         'xaxis': {'visible': False,
                                                                   'showticklabels': False,
                                                                   'showgrid': False},
                                                         'yaxis': {'showticklabels': True,
                                                                   'title': ''}}
        # set plotly update_layout kwargs            
        if plotly_fig_update_layout_kwargs:
            kwargs = plotly_fig_update_layout_kwargs
        else:
            if dark:
                name = 'FIM_DARK.html'
                kwargs = default_plotly_dark_fig_update_layout_kwargs
            else:
                name = 'FIM_LIGHT.html'
                kwargs = default_plotly_light_fig_update_layout_kwargs
        fig.update_layout(**kwargs)
        fig.update_traces(textposition='inside',
                            marker_line_color=BAR_COLOR)
        if display:
            try:
                fig.write_html(name, config={'displaylogo': False})
                self._display_html(name)
                if not save:
                    os.remove(name)
            except Exception as e:
                print(f'Display html error: {e}')
        if save:
            fig.write_html(name, config={'displaylogo': False})
            print(f'Feature Importance Plot is saved to {os.path.join(os.getcwd(), name)}')        
    # ------------------------------------------------------------------------------------------
    def _display_html(self, html_file):
        '''Run html plot in the default browser'''
        import webbrowser
        import time
        # 1st method how to open html files in chrome using
        filename = 'file:///'+os.getcwd()+'/' + html_file
        webbrowser.open_new_tab(filename)
        time.sleep(2)
        os.remove(html_file)
    # ------------------------------------------------------------------------------------------

    def plot_importances(self, 
                         n_features=15, 
                         figsize=(10,6), 
                         interactive=False, 
                         display=True, 
                         dark=True,
                         save=True,
                         plotly_fig_update_layout_kwargs={}):
        '''
        Plot feature importances.

        Can plot interactive html plot or static png plot.

        Parameters
        ----------
        n_features : int, optional
            Number of features to plot. The default is 15.
        figsize : tuple, optional
            Figure size. The default is (10,6).
        interactive : bool, optional
            Create & save to current wd interactive html plot. The default is False.
        display: bool, optional
            Display plot in browser. If False, plot will be saved in cwd. The default is True.
        dark: bool, optional
            Display dark/light version of plot. The default is True.
        save: bool, optional
            Save plot to cwd. The default is True.
        plotly_fig_update_layout_kwargs: dict, optional
            kwargs for plotly.fig.update_layout() function. The default is empty dict and default_plotly_fig_update_layout_kwargs will be used

        Returns
        -------
        None.

        '''
        self._check_refit_status('plot_importances()')
        validate_plot_importances_n_features_argument(n_features)
        validate_plot_importances_figsize_argument(figsize)
        validate_plotting_interactive_argument(interactive)
        if interactive:
            importances_for_html_plot = pd.DataFrame(self._feature_importances.nlargest(n_features).sort_values(), columns = ['importance'])
            importances_for_html_plot['feature'] = importances_for_html_plot.index
            importances_for_html_plot.reset_index(drop = True, inplace = True)
            importances_for_html_plot['importance_percent'] = [str(np.round(val*100,3))+'%' for val in importances_for_html_plot['importance']]
            self._plot_interactive_fim(importances_for_html_plot, 
                                       dark=dark, 
                                       save=save, 
                                       display=display, 
                                       plotly_fig_update_layout_kwargs=plotly_fig_update_layout_kwargs)
        else:
            importances_for_png_plot = self._feature_importances.nlargest(n_features).sort_values()           
            self._plot_static_fim(importances_for_png_plot, 
                                  figsize = figsize, 
                                  dark=dark, 
                                  save=save,
                                  display=display)
    # ------------------------------------------------------------------------------------------
    
    def plot_optimization_history(self, interactive=False, display=True):
        '''
        Plot parameters optimization history.

        Parameters
        ----------
        interactive : bool, optional
            Create & save to current wd interactive html plot. The default is False.
        display: bool, optional
            Display plot in browser. If False, plot will be saved in cwd. The default is True.

        Returns
        -------
        None.

        '''
        self._check_refit_status('plot_optimization_history()')
        validate_plotting_interactive_argument(interactive)
        
        if interactive:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self._study)
            fig.write_html("optimization_history_plot.html")
            if display:
                try:
                    self._display_html("optimization_history_plot.html")
                except Exception as e:
                    print(f'Display html error: {e}')
                    print(f'Optimization History Plot is saved to {os.path.join(os.getcwd(), "optimization_history_plot.html")}')
        else:
            from optuna.visualization.matplotlib import plot_optimization_history
            import matplotlib.pyplot as plt
            plot_optimization_history(self._study)
            plt.show()
    # ------------------------------------------------------------------------------------------
    
    def plot_param_importances(self, interactive=False, display=True):
        '''
        Plot parameters importance.

        Parameters
        ----------
        interactive : bool, optional
            Create & save to current wd interactive html plot. The default is False.
        display: bool, optional
            Display plot in browser. If False, plot will be saved in cwd. The default is True.

        Returns
        -------
        None.

        '''
        self._check_refit_status('plot_param_importance()')
        validate_plotting_interactive_argument(interactive)

        if interactive:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self._study)
            fig.write_html("param_importances_plot.html")
            if display:
                try:
                    self._display_html('param_importances_plot.html')
                except Exception as e:
                    print(f'Display html error: {e}')
                    print(f'Param Importances Plot is saved to {os.path.join(os.getcwd(), "param_importances_plot.html")}')
        else:
            from optuna.visualization.matplotlib import plot_param_importances
            import matplotlib.pyplot as plt
            plot_param_importances(self._study)
            plt.show()
    # ------------------------------------------------------------------------------------------
    
    def plot_intermediate_values(self, interactive=False, legend=False, display=True):
        '''
        Plot optimization trials history. Shows successful and terminated trials.

        Parameters
        ----------
        interactive : bool, optional
            Create & show in default browsersave to current wd interactive html plot. The default is False.
        legend : bool, optional
            Flag to include legend in the static (not interactive) plot. The default is False.
        display: bool, optional
            Display plot in browser. If False, plot will be saved in cwd. The default is True.

        Returns
        -------
        None.

        '''        
        self._check_refit_status('plot_intermediate_values()')
        validate_plotting_interactive_argument(interactive)
        validate_plotting_legend_argument(legend)
        
        if interactive:
            from optuna.visualization import plot_intermediate_values
            fig = plot_intermediate_values(self._study)
            fig.write_html("intermediate_values_plot.html")
            if display:
                try:
                    self._display_html('intermediate_values_plot.html')
                except Exception as e:
                    print(f'Display html error: {e}')
                    print(f'Intermediate Values Plot is saved to {os.path.join(os.getcwd(), "intermediate_values_plot.html")}')
        else:
            from optuna.visualization.matplotlib import plot_intermediate_values
            import matplotlib.pyplot as plt
            fig = plot_intermediate_values(self._study)
            if not legend:
                fig.get_legend().remove()
            plt.show()
    # ------------------------------------------------------------------------------------------

    def _set_optuna_verbosity(self, value):
        '''Set optimizaiton console output verbosity based on optuna.logging options'''
        value_to_optuna_verbosity_dict = {0 : 'CRITICAL',
                                          1 : 'CRITICAL',
                                          2 : 'ERROR',
                                          3 : 'WARNING',
                                          4 : 'INFO',
                                          5 : 'DEBUG'}
        optuna.logging.set_verbosity(getattr(optuna.logging, value_to_optuna_verbosity_dict[value]))
    # ------------------------------------------------------------------------------------------

    def _contains_float(self, iterable):
        '''Check if float any floats are in iterable'''
        return float in [type(x) for x in iterable]

    def _all_ints(self, iterable):
        '''Check if iterable contains only ints'''
        return np.all([type(x)==int for x in iterable])

    def _align_grid_and_search_space(self):
        '''Redefine self.search_space for optuna based on self.grid which could be amended by user'''
        unsupported_params = []
        for param_name, param_grid in self.grid.items():
            if param_name not in supported_gridsearch_params:
                unsupported_params.append(param_name)
                continue
            if isinstance(param_grid, list):
                self.search_space[param_name] = SearchSpace(Distribution.CHOICE, choices = param_grid)
            elif isinstance(param_grid, tuple):
                if self._contains_float(param_grid):
                    self.search_space[param_name] = SearchSpace(Distribution.UNIFORM, low = min(param_grid), high = max(param_grid))
                elif self._all_ints(param_grid):
                    self.search_space[param_name] = SearchSpace(Distribution.INTUNIFORM, low = min(param_grid), high = max(param_grid))
            elif isinstance(param_grid, dict):
                if 'choices' in param_grid:
                    self.search_space[param_name] = SearchSpace(Distribution.CHOICE, choices = param_grid['choices'])
                elif 'low' in param_grid:
                    if self._contains_float(param_grid.values()):
                        self.search_space[param_name] = SearchSpace(Distribution.UNIFORM, low = param_grid['low'], high = param_grid['high'])
                    elif self._all_ints(param_grid.values()):
                        self.search_space[param_name] = SearchSpace(Distribution.INTUNIFORM, low = param_grid['low'], high = param_grid['high'])
        if unsupported_params:
            self.printer.print(f'Following changed parameters are not supported for tuning: {unsupported_params}', order = 'error', trailing_blank_paragraph=True)
        self.grid = {key: value for key, value in self.grid.items() if key not in unsupported_params}
    
    @timer
    def fit(self, X, y, optuna_study_params = None):
        '''
        Find optimized parameters for LightGBM model based on training data and 
        metric (defined by user at __init__).
        
        Considering the lower-better strategy:
            Regression optimization metric
                is selected based on eval_metric (passed by user at __init__), except for r2.
                If eval_metric == 'r2', then optimization metric is 'mean_squared_error'.
            Classification optimization metric
                Allways log_loss        
    
        LGB Classifier/Regressor is inferred based on metric (defined by user at __init__) 
        and target variable statistics.
        
        Initial LGBM parameters are inferred based on data stats & built in logic and are accesable
        by self._init_params
        
        Param_grid for hyperparameters search is inferred based on data stats & built in logic.
        
        Parameters
        ----------
        X : pd.DataFrame
            train features.
        y : pd.Series
            train labels.
        optuna_study_params : dict, optional
            Parameters for optuna study. The default is None. Custom optuna.study.create_study parameters
            https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html

        Returns
        -------
        None.

        '''
        validate_features_argument(X)
        validate_target_argument(y)

        optimization_metric_func = get_optimization_metric_func(self.metric)
        
        self.printer.print('Initiating LGBMTuner.fit', order=1)
        self.printer.print('Settings:', order=3)
        self.printer.print(f'Trying {self.trials} trials', order=4)
        self.printer.print(f'Evaluation metric: {self.metric} ', order=4)
        self.printer.print(f'Study direction: minimize {optimization_metric_func.__name__}', order=4)
        print()
            
        self.target_classes = y.unique().tolist()
        self._get_target_minimum(y)
        self._init_params_on_input(len(X), y)
        # update the predefined params with custom params passed by user
        self._init_params.update(self.custom_lgbm_params)
        self._align_grid_and_search_space()
        self._set_optuna_verbosity(self.verbosity)
        
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        
        optuna_params = {'pruner':optuna.pruners.MedianPruner(n_warmup_steps=5),
                         'sampler':sampler,
                         'direction':'minimize'}
        # incorporate user defined params if passed
        if optuna_study_params is not None:
            optuna_params.update(optuna_study_params)
        study = optuna.create_study(**optuna_params)

        optimization_function = partial(self._objective, X = X.values, y = y.values)

        study.optimize(optimization_function, n_trials = self.trials)

        # populate the learned params into the suggested params
        temp_params = self._populate_best_params_to_init_params(study.best_params)
        # extract early stopping results from best trial
        
        if not 'is_unbalance' in self._init_params or not 'scale_pos_weight' in self._init_params:
            best_trial_number = study.best_trial.number
            num_iterations_in_best_trial = self.early_stopping_results[best_trial_number]
            temp_params['num_iterations'] = num_iterations_in_best_trial
        # tune num_iterations    
        # iteration, best_score = self.optimize_num_iterations(X.values, y.values, temp_params)
        # temp_params['num_iterations'] = iteration
        self._best_params = temp_params        
        if self.refit:
            self.fit_optimized(X.values, y.values)
            self._save_feature_importances(X.columns)
        self._study = study

        if self.visualization:
            self.plot_optimization_history()
            self.plot_param_importances()
            self.plot_intermediate_values()
            if self.refit:
                self.plot_importances(dark=False, save=False)
        # clean up
        self.eval_results_callback = None
        self.stop_training_callback = None
        # --------------------------------
        break_symbol = '|'
        print()
        self.printer.print(f"Optuna hyperparameters optimization finished", order=3)
        self.printer.print(f"Best trial number:{study.best_trial.number:>2}{break_symbol:>5}     {optimization_metric_func.__name__}:{study.best_trial.value:>29}", order=4, breakline='-')