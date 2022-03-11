import operator
import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from copy import copy
from functools import partial
import warnings
from verstack.tools import timer
warnings.filterwarnings("ignore")
from verstack.lgbm_optuna_tuning.lgb_metrics import get_pruning_metric, get_optimization_metric_func, define_objective, classification_metrics, regression_metrics, get_eval_score, print_lower_greater_better, supported_metrics, get_n_rounds_optimization_metric
from verstack.lgbm_optuna_tuning.args_validators import *
from verstack.lgbm_optuna_tuning.optuna_tools import Distribution, OPTUNA_DISTRIBUTIONS_MAP, SearchSpace
# BACKLOG: add option to pass different init params on class __init__
# BACKLOG: add option to pass different param_grid for optimization
# OPTIONAL TODO - add factorization to targets and inverse_transform predictions for classification (E.g. Ghosts, prudential)

class LGBMTuner:

    __version__ = '0.0.8'

    def __init__(self, metric, trials = 100, refit = True, verbosity = 1, visualization = True, seed = 42):
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

        Returns
        -------
        None.

        '''
        self.metric = metric
        self.trials = trials
        self.refit = refit        
        self.verbosity = verbosity
        self.visualization = visualization
        self.seed = seed
        self.target_minimum = None
        self._fitted_model = None
        self._feature_importances = None
        self._study = None # save optuna study for plotting
        self.target_classes = None
        self._init_params = None
        self._best_params = None

    # print init parameters when calling the class instance
    def __repr__(self):
        return f'LGBMTuner(Evaluation metric: {self._metric}\
            \n          trials: {self.trials}\
            \n          refit: {self.refit}\
            \n          verbosity: {self.verbosity}\
            \n          visualization: {self.visualization})'

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

    @refit.setter
    def visualization(self, value):
        if not self._is_bool(value) : raise TypeError('acceptable visualization options are True/False')
        self._visualization = value
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
                                        "n_estimators": 3000,
                                        "early_stopping_rounds": 100,
                                        "random_state": 42}

        default_params_regression = {"learning_rate": 0.05,
                                    "num_leaves": 32,
                                    "colsample_bytree": 0.9,
                                    "subsample": 0.9,
                                    "verbosity": -1,
                                    "n_estimators": 3000,
                                    "early_stopping_rounds": 100,
                                    "random_state": 42}


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
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            self._init_params["num_leaves"] = 32 if task == "regression" else 32
            self._init_params["reg_alpha"] = 0.5 if task == "regression" else 0.0
        elif rows_num > 10000:
            self._init_params["num_leaves"] = 32 if task == "regression" else 64
            self._init_params["reg_alpha"] = 0.5 if task == "regression" else 0.2
        elif rows_num > 5000:
            self._init_params["num_leaves"] = 24 if task == "regression" else 32
            self._init_params["reg_alpha"] = 0.5 if task == "regression" else 0.5
        else:
            self._init_params["num_leaves"] = 16 if task == "regression" else 16
            self._init_params["reg_alpha"] = 1 if task == "regression" else 1

        self._init_params["learning_rate"] = init_lr
        self._init_params["n_estimators"] = ntrees
        self._init_params["early_stopping_rounds"] = es
    # -----------------------------------------------------------------------------

    def _get_default_search_space(self, estimated_n_trials):
        '''Sample hyperparameters from suggested.

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

        search_space["colsample_bytree"] = SearchSpace(Distribution.UNIFORM, low=0.5, high=1.0) 
        search_space["num_leaves"] = SearchSpace(Distribution.INTUNIFORM, low=16, high=255)

        if estimated_n_trials > 30:
            search_space["subsample"] = SearchSpace(Distribution.UNIFORM, low=0.5, high=1.0) 
            search_space["min_sum_hessian_in_leaf"] = SearchSpace(Distribution.LOGUNIFORM, low=1e-3, high=10.0)

        if estimated_n_trials > 100:
            search_space["reg_alpha"] = SearchSpace(Distribution.LOGUNIFORM, low=1e-8, high=10.0)
            search_space["reg_lambda"] = SearchSpace(Distribution.LOGUNIFORM, low=1e-8, high=10.0)

        return search_space
    # -----------------------------------------------------------------------------

    def _sample_from_search_space(self, optimization_search_space, trial):
        '''
        Get params for a trial.

        Parameters
        ----------
        optimization_search_space : dict
            grid for selecting parameters.
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
            trial parameters consisting from suggested_params modified by optimization_search_space.

        '''
        trial_values = copy(self._init_params)
        for parameter, SearchSpace in optimization_search_space.items():
            if SearchSpace.distribution_type in OPTUNA_DISTRIBUTIONS_MAP:
                trial_values[parameter] = getattr(trial, OPTUNA_DISTRIBUTIONS_MAP[SearchSpace.distribution_type])(
                    name=parameter, **SearchSpace.params)
            else:
                for key, value in optimization_search_space.items():
                    print(key)
                    print(value.distribution_type)
                raise ValueError(f"Optuna does not support distribution {SearchSpace.distribution_type}")
        return trial_values
    # -----------------------------------------------------------------------------

    def _get_dtrain_dvalid_objects(self, X, y, metric, seed = None, return_raw_valid = False):
        '''
        Create lgbm.Datasets for training and validation.
        
        By default splits without defined random_state in order to replicate CV (sort of).
        Seed is used for optimize_n_estimators.

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

    def optimize_n_estimators(self, X, y, params, verbose_eval = 100):
        '''
        Optimize n_estimators for lgb model.
        
        Here (after all the tuning) the actual eval_metric is used for early stopping.
        get_n_rounds_optimization_metric(self.metric) returns the feval function 
        from lgb_metrics in an acceptable format.

        Parameters
        ----------
        X : np.array
            train features.
        y : np.array
            train target.
        params : dict
            model parameters.

        Returns
        -------
        int
            best_iteration for further n_estimators param.
        best_score : float
            validation score from best_iteration

        '''
        validate_numpy_ndarray_arguments(X)
        validate_numpy_ndarray_arguments(y)
        validate_params_argument(params)
        validate_verbose_eval_argument(verbose_eval)

        if self.verbosity > 0:
            print('\nTune n_estimators with early_stopping\n')
            verbose_eval_rounds = verbose_eval
        else:
            verbose_eval_rounds = None

        # temporarily change the params['metric'] for the best performance on the eval_metric
        n_rounds_optimization_params = params.copy()
        n_rounds_optimization_params['metric'] = 'None'

        dtrain, dvalid = self._get_dtrain_dvalid_objects(X, y, self.metric, seed = self.seed)

        if n_rounds_optimization_params['objective'] == 'multiclass':
            # disable custom feval function for multiclass 
            # lgbm predicts multiple classes in a single array 
            # without a way to assign predicted probability to a certain class
            # use built in multi_logloss instead
            del n_rounds_optimization_params['metric']
            lgb_model = lgb.train(n_rounds_optimization_params, 
                                  dtrain, 
                                  10000, 
                                  valid_sets=[dtrain, dvalid], 
                                  valid_names=['train', 'valid'],
                                  verbose_eval = verbose_eval_rounds, 
                                  early_stopping_rounds=self._init_params['early_stopping_rounds'])
        else:
            lgb_model = lgb.train(n_rounds_optimization_params, 
                                  dtrain, 
                                  10000, 
                                  valid_sets=[dtrain, dvalid], 
                                  valid_names=['train', 'valid'],
                                  verbose_eval = verbose_eval_rounds, 
                                  early_stopping_rounds=self._init_params['early_stopping_rounds'],
                                  feval = get_n_rounds_optimization_metric(self.metric))
        
        best_score = list(lgb_model.best_score['valid'].values())[0]
        
        return lgb_model.best_iteration, best_score

    # ------------------------------------------------------------------------------------------
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
        if self.verbosity > 0:
            print('-'*73)
            print('\nFitting optimized model with the follwing params:\n')
            for key, value in self._best_params.items():
                print(f'{key:<33}: {value}')
            print('-'*73)
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

        if trial.number/5 % 1 == 0:
            if self.verbosity > 0:
                print(f'\n\n   Trial number: {trial.number} finished')    
                print(f'Optimization score ({optimization_direction:<4}): {optimization_metric_func.__name__}: {result}')
            #!!!!!  TEST THIS LINE BELOW ON ALL EVAL & OPTIMIZATION METRICS
                # calculate & print eval_metric only if eval_metric != optimization_metric
                if self.metric != optimization_metric_func.__name__:
                    eval_score = get_eval_score(valid_y, pred, self.metric, params['objective'])
                    print(f'Evaluation score ({print_lower_greater_better(self.metric):<4}): {self.metric}: {eval_score}\n')
                print('.'*73)
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
        optimization_metric_func = get_optimization_metric_func(self.metric)
        dtrain, dvalid, valid_x, valid_y = self._get_dtrain_dvalid_objects(X, y, self.metric, return_raw_valid = True)
        optimization_search_space = self._get_default_search_space(estimated_n_trials = 200)    
        params = self._sample_from_search_space(optimization_search_space, trial)
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
        # output params are temporary, because n_estimators tuning will follow
        temp_params = copy(self._init_params)
        for key, val in best_params.items():
            temp_params[key] = val
        # remove early_stopping & n_estimators from params (used during optuna optimization).
        # final early stopping will be trained during final_estimators_tuning
        del temp_params['early_stopping_rounds']
        del temp_params['n_estimators']
        
        return temp_params
    # ------------------------------------------------------------------------------------------

    def _save_feature_importances(self, train_features):
        '''Save feature importances in class instance as a pd.Series'''
        feat_importances = pd.Series(self._fitted_model.feature_importance(), index = train_features)
        normalized_importances = np.round((lambda x: x/sum(x))(feat_importances),5)
        self._feature_importances = normalized_importances
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

    def plot_importances(self, n_features = 15, figsize = (10,6), interactive = False):   
        '''
        Plot feature importances.

        Parameters
        ----------
        n_features : int, optional
            Number of features to plot. The default is 15.
        figsize : tuple, optional
            Figure size. The default is (10,6).
        interactive : bool, optional
            Create & save to current wd interactive html plot. The default is False.

        Returns
        -------
        None.

        '''
        self._check_refit_status('plot_importances()')
        validate_plot_importances_n_features_argument(n_features)
        validate_plot_importances_figsize_argument(figsize)
        validate_plotting_interactive_argument(interactive)

        if interactive:
            import plotly.express as px
            plot_df = pd.DataFrame(self._feature_importances.nlargest(n_features).sort_values(), columns = ['importance'])
            plot_df['feature'] = plot_df.index
            plot_df.reset_index(drop = True, inplace = True)
            plot_df['importance_percent'] = [str(np.round(val*100,3))+'%' for val in plot_df['importance']]
            
            fig = px.bar(plot_df, x = 'importance', y = 'feature', 
                         orientation='h', 
                         color = 'importance', 
                         color_continuous_scale = 'ice_r',  #_r for reversing color map
                         title = 'Interactive Feature Importance Plot',
                         text = 'importance_percent')
            fig.write_html('feature_importance_plot.html')
            try:
                self._display_html('feature_importance_plot.html')
            except Exception as e:
                print(f'Display html error: {e}')
                print(f'Optimization History Plot is saved to {os.path.join(os.getcwd(), "feature_importance_plot.html")}')
        else:
            import matplotlib.pyplot as plt
            importances_for_plot = self._feature_importances.nlargest(n_features).sort_values()
            importances_for_plot.plot(kind = 'barh', figsize = figsize, color = 'grey')
            for i, v in enumerate(importances_for_plot):
                plt.text(v, i, str(v), color='grey', fontsize = 10, va  = 'center')
            plt.title('Feature importances (sum up to 1)', color = 'grey')
            plt.show()
    # ------------------------------------------------------------------------------------------
    
    def plot_optimization_history(self, interactive = False):
        '''
        Plot parameters optimization history.

        Parameters
        ----------
        interactive : bool, optional
            Create & save to current wd interactive html plot. The default is False.

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
    
    def plot_param_importances(self, interactive = False):
        '''
        Plot parameters importance.

        Parameters
        ----------
        interactive : bool, optional
            Create & save to current wd interactive html plot. The default is False.

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
    
    def plot_intermediate_values(self, interactive = False, legend = False):
        '''
        Plot optimization trials history. Shows successful and terminated trials.

        Parameters
        ----------
        interactive : bool, optional
            Create & show in default browsersave to current wd interactive html plot. The default is False.
        legend : bool, optional
            Flag to include legend in the static (not interactive) plot. The default is False.

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
        
    @timer
    def fit(self, X, y):
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

        Returns
        -------
        None.

        '''
        validate_features_argument(X)
        validate_target_argument(y)

        optimization_metric_func = get_optimization_metric_func(self.metric)
        
        if self.verbosity > 0:
            print('-'*73)
            print('Initiating lgbm tuning with optuna')
            print(f' Trying {self.trials} trials')
            print(f' Evaluation metric: {self.metric} ')
            print(f' Study direction: minimize {optimization_metric_func.__name__}')#{get_study_direction(metric)} {optimization_metric_func.__name__})
            print('-'*73)

        self.target_classes = y.unique().tolist()
        self._get_target_minimum(y)
        self._init_params_on_input(len(X), y)
        self._set_optuna_verbosity(self.verbosity)
        
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), 
                                    sampler=sampler, 
                                    direction = 'minimize')#get_study_direction(metric))

        optimization_function = partial(self._objective, X = X.values, y = y.values)

        study.optimize(optimization_function, n_trials = self.trials)


        # populate the learned params into the suggested params
        temp_params = self._populate_best_params_to_init_params(study.best_params)

        # tune n_estimators    
        iteration, best_score = self.optimize_n_estimators(X.values, y.values, temp_params)
        temp_params['n_estimators'] = iteration
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
                self.plot_importances()
        
        if self.verbosity > 0:
            
            if self.best_params['objective'] == 'multiclass':
                n_rounds_eval_metric = 'multi_logloss'
            else:
                n_rounds_eval_metric = self.metric
            
            break_symbol = '|'
            print(f"\nOptuna hyperparameters optimization finished")
            print(f"Best trial number:{study.best_trial.number:>2}{break_symbol:>5}     {optimization_metric_func.__name__}:{study.best_trial.value:>29}")
            print('-'*73)
            print(f'n_estimators optimization finished')
            print(f'best iteration:{iteration:>5}{break_symbol:>5}     {n_rounds_eval_metric}:{best_score:>19}')
            print('='*73)
