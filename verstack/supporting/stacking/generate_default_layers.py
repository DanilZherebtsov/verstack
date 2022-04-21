# classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
# regression imports
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
# internal import
from verstack.stacking.kerasModel import kerasModel

def generate_default_layers(objective, epochs = 200, verbose = True):
    '''Initialize two layers with 14 and 2 models

    Parameters
    ----------
    objective : str
        flag to initialize regressors or classifiers.
    epochs : int, optional
        number of epochs for the neural networks compilation. The default is 200.

    Returns
    -------
    layer_1 : list
        14 predefined initialised models.
    layer_2 : list
        2 predefined initialised models.

    '''    
    if objective == 'regression':
        layer_1 = [LGBMRegressor(max_depth = 12, n_jobs = -1),
                   XGBRegressor(max_depth = 10, n_jobs = -1),
                   GradientBoostingRegressor(max_depth = 7),
                   kerasModel(objective = objective, num_layers = 3, epochs = epochs, verbose = verbose),
                   kerasModel(objective = objective, num_layers = 2, epochs = epochs, verbose = verbose),
                   kerasModel(objective = objective, num_layers = 1, epochs = epochs, verbose = verbose),
                   ExtraTreeRegressor(max_depth = 12),
                   RandomForestRegressor(max_depth = 7, n_jobs = -1),
                   LinearRegression(),
                   KNeighborsRegressor(n_neighbors=15, n_jobs = -1),
                   KNeighborsRegressor(n_neighbors=10, n_jobs = -1),
                   SVR(kernel = 'rbf'),
                   DecisionTreeRegressor(max_depth = 15), 
                   DecisionTreeRegressor(max_depth = 8)]

        layer_2 = [LGBMRegressor(max_depth = 3), 
                   Ridge()]
    else:
        layer_1 = [LGBMClassifier(max_depth = 12, n_jobs = -1),
                   XGBClassifier(max_depth = 10, n_jobs = -1),
                   GradientBoostingClassifier(max_depth = 7),
                   kerasModel(objective = objective, num_layers = 3, epochs = epochs, verbose = verbose),
                   kerasModel(objective = objective, num_layers = 2, epochs = epochs, verbose = verbose),
                   kerasModel(objective = objective, num_layers = 1, epochs = epochs, verbose = verbose),
                   ExtraTreeClassifier(max_depth = 12),
                   RandomForestClassifier(max_depth = 7, n_jobs = -1),
                   LogisticRegression(),
                   KNeighborsClassifier(n_neighbors=15, n_jobs = -1),
                   KNeighborsClassifier(n_neighbors=10, n_jobs = -1),
                   SVC(kernel = 'rbf'),
                   DecisionTreeClassifier(max_depth = 15),
                   DecisionTreeClassifier(max_depth = 8)]

        layer_2 = [LGBMClassifier(max_depth = 3), 
                   LogisticRegression()]   

    return layer_1, layer_2

