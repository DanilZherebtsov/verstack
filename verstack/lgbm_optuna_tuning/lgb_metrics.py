#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:34:45 2022

@author: danil
"""

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from mlxtend.evaluate import lift_score as lift
from sklearn.metrics import log_loss

EPSILON = 1e-10

supported_metrics = ['mae', 'mse', 'rmse', 'rmsle', 'mape', 'smape', 'rmspe', 'r2', 
                     'auc', 'gini', 'log_loss', 
                     'accuracy', 'balanced_accuracy',
                     'precision', 'precision_weighted',  'precision_macro',
                     'recall', 'recall_weighted', 'recall_macro',
                     'f1', 'f1_weighted', 'f1_macro',
                     'lift']

regression_metrics = ['mae', 'mse', 'rmse', 'rmsle', 'mape', 'smape', 'rmspe', 'r2']
classification_metrics = ['auc', 'gini', 'log_loss', 
                          'accuracy', 'balanced_accuracy',
                          'precision', 'precision_weighted', 'precision_macro',
                          'recall', 'recall_weighted', 'recall_macro',
                          'f1', 'f1_weighted', 'f1_macro',
                          'lift']

# =============================================================================
# REGRESSION METRICS

def remove_negatives(pred, real):
    '''Remove identical indexes from pred / real if either of these arrays contain negative values'''
    neg_in_pred_idx = np.array(np.where(pred<0)).flatten()
    neg_in_true_idx = np.array(np.where(real<0)).flatten()
    
    neg_in_pred_and_true = list(set(np.concatenate((neg_in_pred_idx, neg_in_true_idx))))
    pred_non_negative = np.delete(pred, neg_in_pred_and_true)
    real_non_negative = np.delete(real, neg_in_pred_and_true)
    return pred_non_negative, real_non_negative

def _error(real, pred):
    """ Simple error """

    real = np.array(real)
    return real - pred
# ------------------------------------------------------------------------------

def _percentage_error(real, pred):
    """
    Percentage error
    Note:
        1. Result is NOT multiplied by 100
        2. Rows with 0 values in target are excluded as division by 0 is impossible and
            a small EPSILON will yield a huge percentage error when relative error will be small
    """

    ind = real != 0
    real = np.array(real[ind])
    pred = pred[ind]
    return _error(real, pred) / (real + EPSILON)
# ------------------------------------------------------------------------------

# mse imported from sklearn
# ------------------------------------------------------------------------------

# mae imported from sklearn
# ------------------------------------------------------------------------------

def rmse(real, pred):
    return mse(real, pred, squared = False)
# ------------------------------------------------------------------------------

def rmsle(real, pred):
    '''Changes negative predictions to 0 for correct calculation'''
    try:
        return msle(real, pred, squared = False)
    except ValueError:
        pred_non_negative, real_non_negative = remove_negatives(pred, real)
        return msle(real_non_negative, pred_non_negative, squared = False)

# ------------------------------------------------------------------------------

# mape imported from sklearn
# ------------------------------------------------------------------------------

def smape(real, pred):
    """
    Symmetric Mean Absolute Percentage Error
    Note:
        1. Result is NOT multiplied by 100
        2. Rows with 0 values in target are excluded as division by 0 is impossible and
            a small EPSILON will yield a huge percentage error when relative error will be small
    """

    ind = real != 0
    real = np.array(real[ind])
    pred = pred[ind]
    return np.mean(2.0 * np.abs(real - pred) / ((np.abs(real) + np.abs(pred)) + EPSILON))
# ------------------------------------------------------------------------------

def rmspe(real, pred):
    """
    Root Mean Squared Percentage Error
    Note:
        1. Result is NOT multiplied by 100
        2. Rows with 0 values in target are excluded as division by 0 is impossible and
            a small EPSILON will yield a huge percentage error when relative error will be small
    """

    ind = real != 0
    real = np.array(real[ind])
    pred = pred[ind]
    return np.sqrt(np.mean(np.square(_percentage_error(real, pred))))
# ------------------------------------------------------------------------------

# r2 imported from sklearn
# =============================================================================

# CLASSIFICATION METRICS

# auc imported from sklearn
# ------------------------------------------------------------------------------

def gini(real, pred):
    roc_auc = auc(real, pred)
    return 2*roc_auc - 1
# ------------------------------------------------------------------------------

# log_loss imported from sklearn
# ------------------------------------------------------------------------------

# acc imported from sklearn
# ------------------------------------------------------------------------------

# bacc imported from sklearn
# ------------------------------------------------------------------------------

# precision imported from sklearn

def precision_weighted(real, pred):
    score = precision(real, pred, average = 'weighted')
    return score
# ------------------------------------------------------------------------------

def precision_macro(real, pred):
    score = precision(real, pred, average = 'macro')
    return score
# ------------------------------------------------------------------------------

# recall imported from sklearn
# ------------------------------------------------------------------------------

def recall_weighted(real, pred):
    score = recall(real, pred, average = 'weighted')
    return score
# ------------------------------------------------------------------------------

def recall_macro(real, pred):
    score = recall(real, pred, average = 'macro')
    return score
# ------------------------------------------------------------------------------

# f1 imported from sklearn
# ------------------------------------------------------------------------------

def f1_weighted(real, pred):
    score = f1(real, pred, average = 'weighted')
    return score
# ------------------------------------------------------------------------------

def f1_macro(real, pred):
    score = f1(real, pred, average = 'macro')
    return score
# ------------------------------------------------------------------------------

# lift imported from mlxtend.evaluate.lift_score
# =============================================================================

def get_optimization_metric_func(eval_metric):
    '''
    Define optimization metric based on evaluation metric (passed by user at __init__).

    Considering the lower-better strategy:
        Regression optimization metric
            is selected based on eval_metric (passed by user at __init__), except for r2.
            If eval_metric == 'r2', then optimization metric is 'mean_squared_error'.
        Classification optimization metric
            Allways log_loss    

    Parameters
    ----------
    eval_metric : str
        evaluation metric passed by user.

    Returns
    -------
    func : function
        optimization metric function.

    '''
    if eval_metric in regression_metrics:
        if eval_metric == 'r2':
            func = globals()['mse']
        else:
            func = globals()[eval_metric]
    # for classification it is always log_loss
    else:
        func = globals()['log_loss']
    return func

# -----------------------------------------------------------------------------

def get_evaluation_metric(eval_metric):
    '''
    Devine evaluation (not to be optimized) metric based on user input.

    Parameters
    ----------
    eval_metric : str
        evaluation metric name.

    Returns
    -------
    func : function
        evaluation metric function.

    '''
    func = globals()[eval_metric]
    return func
# -----------------------------------------------------------------------------

def get_eval_score(labels, pred, eval_metric, objective):
    '''
    Evaluate model agains the eval_metric passed by user.

    Parameters
    ----------
    labels : pd.Series/array
        true labels.
    pred : array
        predicted labels.
    metric : str
        evaluation metric name.
    objective : str
        training objective: regression/binary/multiclass.

    Returns
    -------
    float
        evaluation score.

    '''

    # TODO = tune threshold here
    eval_func = get_evaluation_metric(eval_metric)
    if objective == 'binary':
        if eval_metric not in ['auc', 'gini', 'log_loss']:
            pred = (pred > 0.5).astype('int')
    elif objective == 'multiclass':
        if eval_metric not in ['log_loss']:
            pred = np.argmax(pred, axis = 1)
    return eval_func(labels, pred)
# ------------------------------------------------------------------------------
supported_lgb_metrics_dict = {'mae':'l1',
                              'mse':'l2',
                              'rmse':'rmse',
                              'mape':'mape'}
#                              'auc':'auc'} # auc is supported, but for since always minimize, use log_loss for pruning callback

def get_pruning_metric(eval_metric, target_classes):
    '''
    Define optimization metric for pruning based on evaluation metric (passed by user)
    and LGBM supported metrics.

    Parameters
    ----------
    eval_metric : str
        evaluation metric passed by user.
    target_classes : list
        target unique classes.

    Returns
    -------
    func : function
        optimization metric function.

    '''
    if eval_metric in supported_lgb_metrics_dict:
        func = supported_lgb_metrics_dict[eval_metric]
    else:
        if eval_metric in regression_metrics:
            func = 'l2'
        else:
            if len(target_classes) == 2:
                func = 'binary_logloss'
            else:
                func = 'multi_logloss'
    return func
# ------------------------------------------------------------------------------

def define_objective(metric, y):
    """
    Select an appropriate objective model based on metric and target variable.

    Agrs:
        metric (str): metric name.
        y (pd.Series): target variable series.
        
    Returns:
        objective (str): objective value for lgbm params.

    """

    if metric in regression_metrics:
        objective = 'regression'
    else:
        if len(set(y)) == 2:
            objective='binary'
        else:
            objective='multiclass'
    return objective
# =============================================================================

def print_lower_greater_better(metric):
    if metric in ['mae', 'mse', 'rmse', 'rmsle', 'mape', 'smape', 'rmspe', 'log_loss']:
        return 'lower-better'
    else:
        return 'greater-better'



# LGBM custom metrics for nrounds optimization only. Used in optimize_n_estimators()
# function. 'pred' object is lgbm.Dataset - pred.label attribute is used to extract the 
# actual predicted labels

def get_n_rounds_optimization_metric(eval_metric):
    '''Create evaluation function in lgbm.train supported format for n_rounds optimization'''
    func = globals()['lgb_' + eval_metric]
    return func

def lgb_mae(pred, real):
    ''' sklearn.metrics.mean_squared_error wrapper for LGB '''
    is_higher_better = False
    score = mse(real.label, pred)
    return 'lgb_mae', score, is_higher_better

def lgb_mse(pred, real):
    ''' sklearn.metrics.mean_squared_error wrapper for LGB '''
    is_higher_better = False
    score = mse(real.label, pred)
    return 'lgb_mse', score, is_higher_better

def lgb_rmse(pred, real):
    '''sklearn.metrics.mse(squared = False) wrapper for LGB '''
    is_higher_better = False
    score = rmse(real.label, pred)
    return 'lgb_rmse', score, is_higher_better

def lgb_rmsle(pred, real):
    '''sklearn.metrics.mean_squared_log_error(squared = False) wrapper for LGB'''
    is_higher_better = False
    try:
        score = msle(real.label, pred, squared = False)
    except ValueError:
        pred_non_negative, real_non_negative = remove_negatives(pred, real.label)        
        score = msle(real_non_negative, pred_non_negative, squared = False)
    return 'lgb_rmsle', score, is_higher_better

def lgb_mape(pred, real):
    '''sklearn.metrics.mean_absolute_percentage_error wrapper for LGB'''
    is_higher_better = False
    score = mape(real.label, pred)
    return 'lgb_mape', score, is_higher_better

def lgb_smape(pred, real):
    '''custom smape function - defined above.'''
    is_higher_better = False
    score = smape(real.label, pred)
    return 'lgb_smape', score, is_higher_better

def lgb_rmspe(pred, real):
    is_higher_better = False
    score = rmspe(real.label, pred)
    return 'lgb_rmspe', score, is_higher_better

def lgb_r2(pred, real): 
    ''' sklearn.metrics.r2_score wrapper for LGB.'''
    is_higher_better = True
    score = r2(real.label, pred)
    return 'lgb_r2', score, is_higher_better

def lgb_auc(pred, real):
    ''' sklearn.metrics.roc_auc wrapper for LGB '''
    is_higher_better = True
    score = auc(real.label, pred)
    return 'lgb_auc', score, is_higher_better

def lgb_gini(pred, real):
    is_higher_better = True
    score = gini(real.label, pred)
    return 'lgb_gini', score, is_higher_better

def lgb_log_loss(pred, real):
    ''' sklearn.metrics.log_loss wrapper for LGB '''
    is_higher_better = False
    score = log_loss(real.label, pred)
    return 'lgb_log_loss', score, is_higher_better

def lgb_accuracy(pred, real):
    ''' sklearn.metrics.accuracy_score wrapper for LGB '''
    is_higher_better = True
    score = accuracy(real.label, pred>0.5)
    return 'lgb_accuracy', score, is_higher_better

def lgb_balanced_accuracy(pred, real):
    ''' sklearn.metrics.balanced_accuracy_score wrapper for LGB '''
    is_higher_better = True
    score = balanced_accuracy(real.label, pred>0.5)
    return 'lgb_balanced_accuracy', score, is_higher_better

def lgb_precision(pred, real):
    ''' sklearn.metrics.precision_score wrapper for LGB '''
    is_higher_better = True
    score = precision(real.label, pred>0.5)
    return 'lgb_precision', score, is_higher_better

def lgb_precision_weighted(pred, real):
    ''' sklearn.metrics.precision_score wrapper for LGB '''
    is_higher_better = True
    score = precision(real.label, pred>0.5, average = 'weighted')
    return 'lgb_precision_weighted', score, is_higher_better

def lgb_precision_macro(pred, real):
    ''' sklearn.metrics.precision_score wrapper for LGB '''
    is_higher_better = True
    score = precision(real.label, pred>0.5, average = 'macro')
    return 'lgb_precision_macro', score, is_higher_better

def lgb_recall(pred, real):
    ''' sklearn.metrics.recall_score wrapper for LGB '''
    is_higher_better = True
    score = recall(real.label, pred>0.5)
    return 'lgb_recall', score, is_higher_better

def lgb_recall_weighted(pred, real):
    ''' sklearn.metrics.recall_score wrapper for LGB '''
    is_higher_better = True
    score = recall(real.label, pred>0.5, average = 'weighted')
    return 'lgb_recall_weighted', score, is_higher_better

def lgb_recall_macro(pred, real):
    ''' sklearn.metrics.recall_score wrapper for LGB '''
    is_higher_better = True
    score = recall(real.label, pred>0.5, average = 'macro')
    return 'lgb_recall_macro', score, is_higher_better

def lgb_f1(pred, real):
    ''' sklearn.metrics.f1_score wrapper for LGB '''
    is_higher_better = True
    score = f1(real.label, pred>0.5)
    return 'lgb_f1', score, is_higher_better

def lgb_f1_weighted(pred, real):
    ''' sklearn.metrics.f1_score wrapper for LGB '''
    is_higher_better = True
    score = f1(real.label, pred>0.5, average = 'weighted')
    return 'lgb_f1_weighted', score, is_higher_better

def lgb_f1_macro(pred, real):
    ''' sklearn.metrics.f1_score wrapper for LGB '''
    is_higher_better = True
    score = f1(real.label, pred>0.5, average = 'macro')
    return 'lgb_f1_macro', score, is_higher_better

def lgb_lift(pred, real):
    ''' mlxtend.evaluate.lift_score wrapper for LGB '''
    is_higher_better = True
    score = lift(real.label, pred>0.5)
    return 'lgb_lift', score, is_higher_better

# ------------------------------------------------------------------------------
'''
def get_study_direction(metric):
    from lgb_metrics import regression_metrics, classification_metrics
    if metric in regression_metrics:
            direction = 'minimize'
    else:
        if metric != 'log_loss':
            direction = 'maximize'
        else:
            direction = 'minimize'
    return direction
'''

