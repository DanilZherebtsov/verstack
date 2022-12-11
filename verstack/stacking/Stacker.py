import os
import pickle
import keras
import tensorflow as tf
import copy
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from verstack.stacking.optimise_params import optimise_params
from verstack.stacking.generate_default_layers import generate_default_layers
from verstack.stacking.args_validators import *
from verstack.tools import timer, Printer


'''
TODO: 
    1. Add scaling pipeline for linear models
'''

class Stacker:
    
    __version__ = '0.1.1'
    
    def __init__(self, 
                 objective, 
                 auto = False, 
                 num_auto_layers = 2, 
                 metafeats = True, 
                 epochs = 200, 
                 gridsearch_iterations = 10,
                 stacking_feats_depth = 1,
                 include_X = False,
                 verbose = True):
        '''
        Automatic stacking ensemble configuration, training, and features population to train/test sets.
        
        If stacking more than 1 layer, subsequent layers will can be trained either 
            on the predictions of the previous layer/layers including/not including 
                metafeats and/or original X features; based on the init settings.
        
        Additional statistical meta features can be created from the stacked predictions:
            - pairwise differences between the stacked predictions are created for all pairs (recursively)
            - mean and std for all the stacked features in a layer are created as two extra meta feats
            
        Includes manual and automatic stacking layers creation. 
        - Manual mode reuires user to define models instances, add them to Stacker layers via .add_layer() method.
            Layers must be added after Stacker initialisation separately.
            After creating stacking features using the fit_transform() method, user can add additional layers and 
                perform fit_transform()/transform() on the train/test sets (which have to include stacked features 
                from the previous iteration again to update the Stacker instance and add new features to the train/test sets.
        
        - Automatic mode constructs 2 layers:
            layer_1: 14 models
                - LGBM(max_depth = 12)
                - XGB(max_depth = 10, n_jobs = -1)
                - GradientBoosting(max_depth = 7)
                - kerasModel(num_layers = 3)
                - kerasModel(num_layers = 2)
                - kerasModel(num_layers = 1)
                - ExtraTree(max_depth = 12)
                - RandomForest(max_depth = 7)
                - Linear/LogisticRegression()
                - KNeighbors(n_neighbors=15)
                - KNeighbors(n_neighbors=10)
                - SVR(kernel = 'rbf')
                - DecisionTree(max_depth = 15)
                - DecisionTree(max_depth = 8)

            layer_2: two models
                - LGBM(max_depth = 3)
                - Ridge()
            
            Automatic mode will performs gridsearch of hyperparameters (others than the defined by default)
            on all the models in the default list.

                 num_auto_layers = 2, 
                 metafeats = True, 
                 epochs = 200, 
                 gridsearch_iterations = 10):
        
        Parameters
        ----------
        objective : str
            task type: 'regression', 'binary', 'multiclass'.
        auto : bool, optional
            automated construction of two layers with 14 and 2 models accordingly. The default is False.
        auto_num_layers : int, optional
            Number of layers to create in auto mode. Can except either 1 or 2. The default is 2.
        metafeats : bool, optional
            Flag to create statistical meta features for each layer. The default is True.
        epochs : int, optional
            Number of epochs for the 3 neural networks defined in the auto mode. The default is 200.
        gridsearch_iteration : int
            Number of gridsearch iterations for optimising hyperparameters of models in auto mode. The default is 10.
        stacking_feats_depth : int, optinal
            Defines the features used by subsequent layers to train the stacking models. 
            Can take values between 1 and 4 where:
                1 = use predictions from one previous layer
                2 = use predictions from one previous layer and meta features
                3 = use predictions from all previous layers
                4 = use predictions from all previous layers and meta features
                The default is 1.
        include_X : bool
            Flag to use original X features for subsequent layer training. The default is False.
        verbose : bool
            Flag to print stacking progress to the console. The default is True.

        Returns
        -------
        None.

        '''
        self.verbose = verbose
        self.printer = Printer(verbose=self.verbose)
        self.objective = objective
        self.auto = auto
        self.num_auto_layers = num_auto_layers
        self.metafeats = metafeats
        self.layers = {}
        self.trained_models = {}
        self._trained_models_list_buffer = None
        self._extra_layers_for_test_set_application = []
        self.epochs = epochs
        self.gridsearch_iterations = gridsearch_iterations
        self.stacking_feats_depth = stacking_feats_depth
        self.include_X = include_X
        self.stacked_features = {} #lists of stacked features by layers

        self._set_default_layers()

    # print init parameters when calling the class instance
    def __repr__(self):
        return f'Stacker(objective: {self.objective}\
            \n        auto: {self.auto}\
            \n        num_auto_layers: {self.num_auto_layers}\
            \n        metafeats: {self.metafeats}\
            \n        epochs : {self.epochs}\
            \n        gridsearch_iterations: {self.gridsearch_iterations}\
            \n        stacking_feats_depth: {self.stacking_feats_depth}\
            \n        include_X: {self.include_X}\
            \n        verbose : {self.verbose})'

    
    # VALIDATE INIT ARGUMENTS
    # -------------------------------------------------------------------------
    # objective
    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        validate_objective(value)
        self._objective = value
    # -------------------------------------------------------------------------
    # auto
    @property
    def auto(self):
        return self._auto

    @auto.setter
    def auto(self, value):
        validate_bool_arg(value)
        self._auto = value
    # -------------------------------------------------------------------------
    # num_auto_layers
    @property
    def num_auto_layers(self):
        return self._num_auto_layers

    @num_auto_layers.setter
    def num_auto_layers(self, value):
        validate_num_auto_layers(value)
        self._num_auto_layers = value
    # -------------------------------------------------------------------------
    # metafeats
    @property
    def metafeats(self):
        return self._metafeats

    @metafeats.setter
    def metafeats(self, value):
        validate_bool_arg(value)
        self._metafeats = value
    # -------------------------------------------------------------------------
    # epochs
    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        validate_epochs(value)
        self._epochs = value
    # -------------------------------------------------------------------------
    # gridsearch_iterations
    @property
    def gridsearch_iterations(self):
        return self._gridsearch_iterations

    @gridsearch_iterations.setter
    def gridsearch_iterations(self, value):
        validate_gridsearch_iterations(value)
        self._gridsearch_iterations = value
    # -------------------------------------------------------------------------
    # include_X
    @property
    def include_X(self):
        return self._include_X

    @include_X.setter
    def include_X(self, value):
        validate_bool_arg(value)
        self._include_X = value
    # -------------------------------------------------------------------------
    # verbose
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        validate_bool_arg(value)
        self._verbose = value
    # =========================================================================

    def _set_default_layers(self):
        '''Enable or not enable default layers based on self.auto init attribute'''
        if not self.auto:
            return
        else:
            layers = self._get_default_layers()
            for layer in layers:
                self.add_layer(layer)

    def _get_default_layers(self):
        '''Configure 1 or 2 default layers'''
        layers = []        
        layer_1, layer_2 = generate_default_layers(self.objective, self.epochs, self.verbose)
        layers.append(layer_1)
        if self.num_auto_layers == 2:
            layers.append(layer_2)
        return layers

    def _create_feat_name(self, layer):
        '''Assign feature name based on layer and consecutive number'''
        feat_count_in_layer = len(self.trained_models[layer].keys())        
        feat_name = '_'.join([layer, str(feat_count_in_layer)])
        return feat_name

    # -------------------------------------------------------------------------   
    def _predict_class_proba_or_value(self, model, X):
        '''Create single vector with predictions: 
            - classes for multiclass
            - positive class probabilities for binary
            - predicted numeric values for regression
            
        '''
        if self.objective == 'binary':
            try:
                pred = model.predict_proba(X)[:,1].flatten()
            except:
                pred = model.predict(X).flatten()
        else:
            pred = model.predict(X).flatten()
        return pred

    def _predict_by_model(self, model, X):
        '''Create predictions by passed model.'''
        pred = self._predict_class_proba_or_value(model, X)
        pred_series = pd.Series(pred)
        return pred_series
    # -------------------------------------------------------------------------            
    def _train_predict(self, model, X_train, y_train, X_test):
        '''Fit configured model, predict outputs by _predict_class_proba_or_value() method'''
        model.fit(X_train, y_train)
        pred = self._predict_class_proba_or_value(model, X_test)
        return pred, model

    def _configure_kfold_splitter(self):
        '''Initialize stratified or random kfold iterator based on objective'''
        if self.objective == 'regression':
            kfold = KFold(n_splits = 4, shuffle = True)
        else:
            kfold = StratifiedKFold(n_splits = 4, shuffle = True)
        return kfold

    def _train_predict_by_model(self, model, X, y):
        '''Train/predict by model on folds, create prediction series from each fold prediction
            Optimize model hyperparameters if self.auto
            Save list of trained model instances into self._trained_models_list_buffer
            for latter moving them into self.trained_models storage.
                
        '''
        # create placeholder for stacked feat
        pred_series = pd.Series(index = X.index)
        trained_models_list = []
        kfold = self._configure_kfold_splitter()

        if self.gridsearch_iterations > 0:
            model = optimise_params(model, X, y, self.objective, self.gridsearch_iterations, self.verbose)

        fold = 0
        for train_ix, test_ix in kfold.split(X,y):
            X_train = X.loc[train_ix, :]
            y_train = y.loc[train_ix]
            X_test = X.loc[test_ix, :]
            # create independent model instance for each fold
            try:
                fold_model = copy.deepcopy(model)
            except ValueError:
                fold_model = copy.copy(model)
            pred, model = self._train_predict(fold_model, X_train, y_train, X_test)
            pred_series.loc[test_ix] = pred.flatten()
            trained_models_list.append(fold_model)            
            fold+=1
            if fold%2 == 0:
                self.printer.print(f'fold {fold} trained/predicted', 4)
        self._trained_models_list_buffer = trained_models_list
        return pred_series

    def _get_stack_feat(self, model, X, y = None):
        '''Apply stacking features creatin to either train or test set'''
        if isinstance(y, pd.Series):
            new_feat = self._train_predict_by_model(model, X, y)
        else:
            new_feat = self._predict_by_model(model, X)
        return new_feat

    # -------------------------------------------------------------------------
    def _create_layer_name(self):
        '''Create layer name from next consecutive number'''
        layer_keys = list(self.layers.keys())
        new_layer_ix = len(layer_keys) + 1
        new_layer_name = 'layer_' + str(new_layer_ix)
        return new_layer_name
    
    def _increment_layer(self, models_list):
        '''Increment layer with models to self.layers'''
        if not self.layers:
            self.layers['layer_1'] = models_list
        else:
            new_layer = self._create_layer_name()
            self.layers[new_layer] = models_list
            
    def add_layer(self, models_list):
        '''
        Add layer to Stacker, requires list of models instances as argument.

        Models list is added to Stacker.layers
        
        Parameters
        ----------
        models_list : list
            list of models instances each of which must contain methods:
                - fit()
                - predict()
                - predict_proba() for classification tasks

        Returns
        -------
        None.

        '''        
        validate_models_list(models_list)
        self._increment_layer(models_list)
    # -------------------------------------------------------------------------
    def _get_stacked_layer_feats(self, X, layer, metafeats):
        '''Get list of features names created within layer'''
        if metafeats:
            layer_feats = self.stacked_features[layer]
        else:
            layer_feats = list(self.trained_models[layer].keys())
        return layer_feats
            
    def _get_original_feats(self, X):
        '''Get names of original X feats (without any created stacking features)'''
        stacked_feats = list(self.stacked_features.values())
        if type(stacked_feats[0]) == list: # flatten nested list
            stacked_feats = sum(stacked_feats, [])
        original_feats = [col for col in X if col not in stacked_feats]
        return original_feats

    def _get_applicable_feats(self, X, layer):
        '''Get list of applicable features for trainin/prediction:
            - for layer_1  : all featues in X
            - for layer_2+ : preceeding layer outputs
                or preceeding layers outputs 
                    can include metafeats self.metafeats
                    can include original X features if self.include_X
            
        '''

        layer_number = int(layer.split('_')[1])
        
        if layer_number > 1:          
            # get last layer for creating applicable_feats based on a single preceeding layer
            preceeding_layers_list = list(self.stacked_features.keys())
            if layer in preceeding_layers_list:
                preceeding_layers_list = preceeding_layers_list[: preceeding_layers_list.index(layer)]
            last_layer = [l for l in preceeding_layers_list if l != layer][-1]
            
            generate_metafeats_flag = self.metafeats
            
            if self.stacking_feats_depth == 1:
                applicable_feats = self._get_stacked_layer_feats(X, last_layer, metafeats = generate_metafeats_flag)
            elif self.stacking_feats_depth == 2:
                applicable_feats = self._get_stacked_layer_feats(X, last_layer, metafeats = generate_metafeats_flag)
            elif self.stacking_feats_depth == 3:
                applicable_feats = []
                for l in preceeding_layers_list:
                    applicable_feats += self._get_stacked_layer_feats(X, l, metafeats = generate_metafeats_flag)
            else: # self.stacking_feats_depth == 4:
                applicable_feats = []
                for l in preceeding_layers_list:
                    applicable_feats += self._get_stacked_layer_feats(X, l, metafeats = generate_metafeats_flag)

            if self.include_X:
                # include original X features
                applicable_feats = applicable_feats + self._get_original_feats(X)
        else:
            applicable_feats = X.columns.tolist()
        return applicable_feats
    # -------------------------------------------------------------------------

    def _extract_most_common_multiclass_pred(self, preds_lists):
        '''Extract single (most) common prediction from list of lists with predictions from multiple models
            applicable for multiclass predictions
            
        '''
        def most_common_of_all_preds(lst):
            return max(set(lst), key=lst.count)
        
        # create list of tuples with n predictions from n models for each value
        preds_tuples = list(list(zip(*preds_lists)))
        result = [most_common_of_all_preds(pred) for pred in preds_tuples]
        return result                    

    def _average_predictions(self, preds_from_models, models_cnt):
        '''Create singe predictions vector from predictions from multiple models'''
        if self.objective == 'multiclass':
            preds_from_models = pd.Series(self._extract_most_common_multiclass_pred(preds_from_models))
        else:
            preds_from_models = preds_from_models/models_cnt
        return preds_from_models

    def _store_preds_together(self, preds_from_models, pred):
        '''Store predictions from multiple models into a single object.
            Append multiclass predictions to list of lists
            or sum binary probabilities/regression predictions in a series
        
        '''
        if self.objective == 'multiclass':
            preds_from_models.append(list(pred))                       
        else:                        
            if type(preds_from_models) != pd.Series:
                preds_from_models = pred
            else:
                preds_from_models += pred
        return preds_from_models
    # -------------------------------------------------------------------------

    def _create_new_feats_in_test(self, X, y, layer, applicable_feats):
        '''Header function to create stacking feats in train set by models in layer'''
        new_feats = []

        for feat_name in self.trained_models[layer].keys():                      
            models_list = self.trained_models[layer][feat_name]
            # create placeholder for predicting with all models for feat
            preds_from_models = []
            models_cnt = 0
            for model in models_list:
                pred = self._get_stack_feat(model, X[applicable_feats], y)
                preds_from_models = self._store_preds_together(preds_from_models, pred)
                models_cnt+=1
            preds_from_models = self._average_predictions(preds_from_models, models_cnt)
            preds_from_models.name = feat_name
            new_feats.append(preds_from_models)                
            self.printer.print(f'predicted with model {len(new_feats)}', 3)
        return new_feats
    
    def _create_new_feats_in_train(self, X, y, layer, applicable_feats):
        '''Header function to create stacking feats in test set by models in layer'''        
        new_feats = []
        for model in self.layers[layer]:
            feat_name = self._create_feat_name(layer)
            new_feat = self._get_stack_feat(model, X[applicable_feats], y)
            # append trained models from buffer to self.trained_models_list for layer/feature
            self.trained_models[layer][feat_name] = self._trained_models_list_buffer
            # clean up
            self._trained_models_list_buffer = None
            new_feats.append(pd.Series(new_feat, name = feat_name))
        return new_feats
    # -------------------------------------------------------------------------
        
    def _apply_single_layer(self, layer, X, y = None):
        '''Create stacked feats from all models in layer for train or test set
            train and test set is distinguished by y being None or not

        '''
        if y is None:
            self.printer.print(f'Predicting with {layer} models', 2)
        else:
            self.printer.print(f'Training/predicting with {layer} models', 2)
        cols_before_layer_stacking = X.columns.tolist()
        # create layer placeholder in self.trained_models list
        if layer not in self.trained_models.keys():
            self.trained_models[layer] = {}
        # get list of feature names to train/predict by models in layer
        applicable_feats = self._get_applicable_feats(X, layer)


        # create stacked feats in test set
        if y is None:
            new_feats = self._create_new_feats_in_test(X, y, layer, applicable_feats)
        # ---------------------------------------------------------------------
        # create stacked feats in train set
        else:
            new_feats = self._create_new_feats_in_train(X, y, layer, applicable_feats)
        for feat in new_feats:
            X = pd.concat([X, feat], axis = 1) 
        # ---------------------------------------------------------------------            
        # add metafeats to layer
        if self.metafeats:
            X = self._create_stacking_meta_features(X, layer)            
        cols_after_layer_stacking = [col for col in X if col not in cols_before_layer_stacking]
        if layer not in self.stacked_features.keys():
            self.stacked_features[layer] = cols_after_layer_stacking
        return X
    # -------------------------------------------------------------------------
    def _get_features_pairs_recursive(self, layer):
        '''Create list of all possible pairs combination in list'''
        feats = layer.copy()
        pairs = []
        for ix in range(len(feats)):
            feat = feats.pop(0)
            for feat2 in feats:
                pairs.append([feat, feat2])
        return pairs
    
    def _get_diff(self, feat1, feat2):
        '''Subtract pd.Series(feat2_ from pd.Series(feat1)'''
        return feat1 - feat2
    # -------------------------------------------------------------------------
    def _create_stacking_meta_features(self, X, layer):
        '''Generate statistics from stacked predictions; not applicable for multiclass'''

        if self.objective == 'multiclass':
            return X
        else:
            features_names_in_layer = list(self.trained_models[layer].keys())
            features_pairs = self._get_features_pairs_recursive(features_names_in_layer)
            metafeats_df = pd.DataFrame()        
            for pair in features_pairs:
                feat1 = pair[0]
                feat2 = pair[1]
                meta_feat_name = f'diff_{feat1}_{feat2}'
                diff = self._get_diff(X[feat1], X[feat2])            
                metafeats_df[meta_feat_name] = diff
            metafeats_df[f'{layer}_std'] = X[features_names_in_layer].std(axis = 1)
            metafeats_df[f'{layer}_mean'] = X[features_names_in_layer].mean(axis = 1)
            return pd.concat([X, metafeats_df], axis = 1)
        
    def _apply_all_layers(self, X, y = None):
        '''Wrapper for calling the self._apply_single_layer() method on all predefined layers'''
        for layer in list(self.layers.keys()):
            X = self._apply_single_layer(layer, X, y)
        return X

    def _apply_all_or_extra_layers_to_train(self, X, y = None):
        '''Perform initial stacking (on train set) on predefined layers or additional stacking with layers added 
            after calling fit_transform() method on initially predefined layers
            
        '''
        # search for layers added after fit_transform()
        layers_added_after_fit_transform = [x for x in self.layers.keys() if x not in self.trained_models.keys()]
        # append layers added after fit transform to buffer for test set transform()
        self._extra_layers_for_test_set_application += layers_added_after_fit_transform

        # apply extra layers on train set
        if layers_added_after_fit_transform:
            for layer in layers_added_after_fit_transform:
                X = self._apply_single_layer(layer, X, y)
        else:
            # if no extra layers apply all layers on train set
            X = self._apply_all_layers(X, y)      
        return X
    
    def _apply_all_or_extra_layers_to_test(self, X):
        '''Perform initial stacking (on train set) on predefined layers or additional stacking with layers added 
            after calling fit_transform()/transform() methods on initially predefined layers
        
        '''
        # iterate over self._extra_layers_for_test_set_application list
        if self._extra_layers_for_test_set_application:
            for ix in range(len(self._extra_layers_for_test_set_application)):
                layer = self._extra_layers_for_test_set_application.pop(0)
                X = self._apply_single_layer(layer, X)
        else:
            # if no extra layers apply all layers on test set
            X = self._apply_all_layers(X)          
        return X

    @timer
    def fit_transform(self, X, y):
        '''
        Train/predict/append to X stacking features from models defined in self.layers.

        Parameters
        ----------
        X : pd.DataFrame - train features
        y : pd.Series - train target.

        Returns
        -------
        X_with_stacked_feats : pd.DataFrame
            train featues with appended stacking features.

        '''
        X.reset_index(drop=True, inplace=True)
        self.printer.print('Initiating Stacker.fit_transform', order=1)
        validate_fit_transform_args(X, y)
        X_with_stacked_feats = X.reset_index(drop=True).copy()
        X_with_stacked_feats = self._apply_all_or_extra_layers_to_train(X_with_stacked_feats, y)
        return X_with_stacked_feats
    
    def transform(self, X):
        '''Create stacking features on the test set from models saved in self.trained_models
        Parameters
        ----------
        X : pd.DataFrame - test features

        Returns
        -------
        X_with_stacked_feats : pd.DataFrame
            test featues with appended stacking features.

        '''
        self.printer.print('Initiating Stacker.transform', order=1)
        validate_transform_args(X)
        X_with_stacked_feats = X.reset_index(drop=True).copy()
        X_with_stacked_feats = self._apply_all_or_extra_layers_to_test(X_with_stacked_feats)
        return X_with_stacked_feats
    
    def save_stacker(self, path):
        '''Save trained models and Stacker instance to the specified path
        
        Parameters
        ----------
        path : str
            path to the directory where the models will be saved
            save_stacker method will create a directory stacker_saved_model in the specified path
            
        Returns
        -------
        None.'''
    
        filepath = os.path.join(path, 'stacker_saved_model')
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        for layer in self.trained_models.keys():
            for model_ix_in_layer in self.trained_models[layer]:
                save_dir = self._make_dir_for_layer_models(filepath, model_ix_in_layer)
                models_lst = self.trained_models[layer][model_ix_in_layer]
                for ix, model in enumerate(models_lst):
                    model_save_path = os.path.join(save_dir, str(ix))
                    try:
                        self._save_keras_model(model, model_save_path)                
                    except AttributeError:
                        self._pickle_model(model, model_save_path)
        self._save_stacker_instance(filepath)
        print('Stacker instance saved to', filepath)
    
    def _save_stacker_instance(self, path):
        '''Save stacker instance without trained models'''
        self_copy = copy.copy(self)
        self_copy.trained_models = {}
        with open(f'{path}/stacker.p', 'wb') as f:
            pickle.dump(self_copy, f)
    
    def _save_native_keras_model(self, model, path):
        '''Save native keras model'''
        tf.keras.models.save_model(model, path)
    
    def _save_keras_model(self, model, path):
        '''Save keras model as a stacker wrapped keras model  in two steps: 
        stacker pickle instance and keras.save_model object'''
        try:
            model_copy = copy.deepcopy(model)
        except:
            model_copy = copy.copy(model)
        # extract the native keras model from stacker instance
        keras_model = model_copy.model
        # same the native keras model
        self._save_native_keras_model(keras_model, path)
        # save the verstack.stacking.kerasModel instance without the actual keras model
        model_copy.model = None
        with open(f'{path}/verstack.stacking.kerasModel', 'wb') as f:
            pickle.dump(model_copy, f)
        
    def _pickle_model(self, model, path):
        '''Save model with pickle'''
        with open(path, 'wb') as f:
            pickle.dump(model, f)
                    
    def _make_dir_for_layer_models(self, filepath, model_ix_in_layer):
        '''Make directory for fold models within one model in layer'''
        save_dir = os.path.join(filepath, model_ix_in_layer)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        return save_dir

