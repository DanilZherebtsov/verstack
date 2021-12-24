import pandas as pd
import numpy as np

def validate_features_argument(value):
    if not isinstance(value, pd.DataFrame): 
        raise TypeError('X (features) must be a pandas DataFrame')
    if len(value) == 0:
        raise Exception('X (features) must contain data')

def validate_target_argument(value):
    if not isinstance(value, pd.Series):
        raise TypeError('y (target variable) must be a pandas Series')
    if len(value) == 0:
        raise Exception('X (features) must contain data')

def validate_threshold_argument(value):
    if not isinstance(value, float):
        raise TypeError('threshold must be a float')
    if not 0 < value < 1:
        raise ValueError('threshold must can take values 0 < threshold < 1')
    
def validate_plotting_interactive_argument(value):
    if not isinstance(value, bool):
        raise TypeError('interactive argument can be True/False')

def validate_plotting_legend_argument(value):
    if not isinstance(value, bool):
        raise TypeError('legent argument can be True/False')

def validate_plot_importances_n_features_argument(value):
    if not isinstance(value, int):
        raise TypeError('n_features argument must be of type int')        
    if value <= 0:
        raise ValueError('n_features argument must be positive')
        
def validate_plot_importances_figsize_argument(value):
    if not isinstance(value, tuple):
        raise TypeError('figsize argument must be a tuple. E.g. (15,10)')
    if len(value) != 2:
        raise ValueError('figsize must contain two values. E.g. (15,10)')
    if np.any([val <= 0 for val in value]):
        raise ValueError('figsize must contain two positive values. E.g. (15,10)')

def validate_numpy_ndarray_arguments(value):
    if not isinstance(value, np.ndarray):
        raise TypeError('Arguments to fit_optimized() must be of type numpy.array')

def validate_params_argument(value):
    if not isinstance(value, dict):
        raise TypeError('params argument must be of type dict')

def validate_verbose_eval_argument(value):
    if not isinstance(value, int):
        raise TypeError('verbose_eval argument must be of type int')
    if value <= 0:
        raise ValueError('verbose_eval must be positive')
