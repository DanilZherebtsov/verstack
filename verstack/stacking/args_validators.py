import pandas as pd

def validate_objective(value):
    if not isinstance(value, str): 
        raise TypeError('objective must be a string')
    if value not in ['regression', 'binary', 'multiclass']:
        raise NameError('objective can take values: "regression", "binary", "multiclass"')

def validate_bool_arg(value):
    '''Validate arguments "auto", "meta_feats", "verbose"'''
    if not isinstance(value, bool):
        raise TypeError('auto must be bool')

def validate_num_auto_layers(value):
    if not isinstance(value, int): 
        raise TypeError('num_auto_layers must be an integer')
    if value not in [1,2]:
        raise ValueError('num_auto_layers can take values: 1, 2')

def validate_epochs(value):
    if not isinstance(value, int): 
        raise TypeError('epochs must be an integer')
    if value < 1:
        raise ValueError('epochs must be a positive integer')

def validate_gridsearch_iterations(value):
    if not isinstance(value, int): 
        raise TypeError('gridsearch_iterations must be an integer')
    if value < 0:
        raise ValueError('gridsearch_iterations must be >=0')

def validate_stacking_feats_depth(value):
    if not isinstance(value, int): 
        raise TypeError('stacking_feats_depth must be an integer')
    if value not in 4 in range(5):
        raise ValueError('stacking_feats_depth can take values: 1, 2, 3, 4')

def validate_models_list(value):
    '''Validate add_layer(models_list)'''

    def validate_class_instance(value):
        '''Validate model in models_list is a class instance'''
        if not hasattr(value, '__dict__'):
            print(f'{value} is not a valid model, it must be a class instance')

    if not isinstance(value, list):
        raise TypeError('models_list must be a list')
    if len(value) < 1:
        raise ValueError('models_list must include at least 1 model')
    for val in value:
        validate_class_instance(val)
        
def validate_fit_transform_args(X, y):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas.DataFrame')
    if not isinstance(y, pd.Series):
        raise TypeError('y must be a pandas.Series')
    if not len(X) == len(y):
        raise ValueError('X and y must same length')

def validate_transform_args(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('X must be a pandas.DataFrame')
