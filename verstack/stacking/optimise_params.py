from verstack.tools import Printer
from sklearn.model_selection import RandomizedSearchCV

def get_grid_search_instance(model, params, objective, gridsearch_iterations):
    '''Create RandomisedGridSearch instance based on model, params, objective, gridsearch_iterations'''
    if objective == 'regression':
        scoring = 'neg_mean_squared_error'
    else:
        scoring = 'neg_log_loss'
    rand_gridsearch = RandomizedSearchCV(model,
                                         param_distributions=params,
                                         scoring=scoring,
                                         cv=3,
                                         n_iter = gridsearch_iterations)
    return rand_gridsearch
# -----------------------------------------------------------------------------
def optimise_params(model, X, y, objective, gridsearch_iterations, verbose):
    '''
    Optimise hyperparameters of model if model is supported by the predefined models types:
        - lightgbm.sklearn.LGBMRegressor / lightgbm.sklearn.LGBMClassifier
        - xgboost.sklearn.XGBRegressor / xgboost.sklearn.XGBClassifier
        - sklearn.ensemble.GradientBoostingRegressor / sklearn.ensemble.GradientBoostingClassifier
        - sklearn.tree._classes.ExtraTreeRegressor / sklearn.tree._classes.ExtraTreeClassifier
        - sklearn.ensemble._forest.RandomForestRegressor / sklearn.ensemble._forest.RandomForestClassifier
        - sklearn.linear_model._logistic.LogisticRegression
        - sklearn.linear_model._ridge.Ridge
        - sklearn.neighbors._regression.KNeighborsRegressor / sklearn.neighbors._classification.KNeighborsClassifier
        - sklearn.svm._classes.SVR / sklearn.svm._classes.SVC
        - sklearn.tree._classes.DecisionTreeRegressor / sklearn.tree._classes.DecisionTreeClassifier

    If model not supported, it is returned not optimised.
    
    Parameters
    ----------
    model : model class instance
        model to optimise.
    X : pd.DataFrame
        train features.
    y : pd.Series
        train labels.
    objective : str
        regression/binary/multiclass.
    gridsearch_iterations : int
        number of gridsearch iterations.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    '''
    import sklearn
    import lightgbm
    import xgboost
    printer = Printer(verbose=verbose)
    printer.print(f'Optimising model hyperparameters', 3)
    try:
        # LGBM OPTIMIZATION
        if type(model) in [lightgbm.sklearn.LGBMRegressor, lightgbm.sklearn.LGBMClassifier]:
            params = {'colsample_bytree'  : [0.5, 0.7, 1],
                      'subsample'         : [0.5, 0.7, 1],
                      'num_leaves'        : [16, 32, 64, 128, 255],
                      'reg_alpha'         : [1e-8, 1e-4, 1, 10],
                      'reg_lambda'        : [1e-8, 1e-4, 1, 10]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # XGB OPTIMIZATION
        if type(model) in [xgboost.sklearn.XGBRegressor, xgboost.sklearn.XGBClassifier]:
            params = {'subsample'         : [0.5, 0.7, 1.0],
                      'colsample_bytree'  : [0.5, 0.7, 1.0],
                      'min_child_weight'  : [0.1, 1, 5, 10, 20],
                      'gamma'             : [0.5, 1, 1.5, 2, 5],
                      'eta'               : [0.005, 0.01, 0.1]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # GradientBoosting OPTIMIZATION
        if type(model) in [sklearn.ensemble.GradientBoostingRegressor, sklearn.ensemble.GradientBoostingClassifier]:
            params = {'subsample'         : [0.5, 0.7, 1.0],
                      'max_features'      : [0.5, 0.7, 1.0],
                      'min_samples_split' : [2, 5, 10, 20],
                      'min_samples_leaf'  : [1, 3, 5, 10],
                      'learning_rate'     : [0.005, 0.01, 0.1]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # ExtraTrees OPTIMIZATION
        if type(model) in [sklearn.tree._classes.ExtraTreeRegressor, sklearn.tree._classes.ExtraTreeClassifier]:
            params = {'max_features'      : [0.5, 0.7, 1.0],
                      'min_samples_split' : [2, 5, 10, 20],
                      'min_samples_leaf'  : [1, 3, 5, 10]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # RandomForest OPTIMIZATION
        if type(model) in [sklearn.ensemble._forest.RandomForestRegressor, sklearn.ensemble._forest.RandomForestClassifier]:
            params = {'max_samples'       : [0.5, 0.7, 1.0],
                      'min_samples_split' : [2, 5, 10, 20],
                      'min_samples_leaf'  : [1, 3, 5, 10]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # LogisticRegression OPTIMIZATION
        if type(model) == sklearn.linear_model._logistic.LogisticRegression:
            params = {'penalty'           : ['l1', 'l2'],
                      'C'                 : [0.5, 0.7, 1]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # Ridge OPTIMIZATION
        if type(model) == sklearn.linear_model._ridge.Ridge:
            params = {'alpha'             : [0.5, 0.7, 1]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # KNN OPTIMIZATION
        if type(model) in [sklearn.neighbors._regression.KNeighborsRegressor, sklearn.neighbors._classification.KNeighborsClassifier]:
            params = {'weights'           : ['uniform', 'distance'],
                      'leaf_size'         : [10, 15, 20, 30, 40, 50]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # SVM OPTIMIZATION
        if type(model) in [sklearn.svm._classes.SVR, sklearn.svm._classes.SVC]:
            params = {'C'                 : [0.5, 0.7, 1]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        # DecisionTree OPTIMIZATION
        if type(model) in [sklearn.tree._classes.DecisionTreeRegressor, sklearn.tree._classes.DecisionTreeClassifier]:
            params = {'min_samples_split' : [2, 5, 10, 20],
                      'min_samples_leaf'  : [1, 3, 5, 10]}
            rand_gridsearch = get_grid_search_instance(model, params, objective, gridsearch_iterations)
        # ---------------------------------------------------------------------
        rand_gridsearch.fit(X, y)
        model.set_params(**rand_gridsearch.best_params_)

    except:
        printer.print(f'Model not in optimisation list {model}', order = 4)
    
    return model
# -----------------------------------------------------------------------------
