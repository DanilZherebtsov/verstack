import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
import pytest
from common import generate_data
from verstack import Stacker
import shutil
# test overall Stacker not being broken
def test_Stacker_fit_transform():
    '''Test if Stacker will stack features'''
    df = generate_data(processed=True)
    module = Stacker('regression', auto=True, epochs=5, gridsearch_iterations=4)
    X = df.drop('y', axis = 1)
    y = df['y']
    X_train = module.fit_transform(X, y)
    X_test = module.transform(X)
    
    result_correct_treansform_train_and_test = np.all(X_train.columns == X_test.columns)
    result_layers_with_feats_present_in_stacker = len(module.stacked_features)>0
    result_layers_with_feats_present_in_X = X.shape[1] < X_train.shape[1]
    assert result_layers_with_feats_present_in_stacker
    assert result_layers_with_feats_present_in_X
    assert result_correct_treansform_train_and_test

def test_Stacker_save_load():
    '''Test if Stacker will save and load'''
    df = generate_data(processed=True)
    module = Stacker('regression', auto=True, epochs=5, gridsearch_iterations=2)
    X = df.drop('y', axis = 1)
    y = df['y']
    X_train = module.fit_transform(X, y)
    X_test = module.transform(X)
    
    module.save_stacker('./')
    
    from verstack.stacking.load_stacker import load_stacker
    module_loaded = load_stacker('./stacker_saved_model')
    X_test_after_upload = module_loaded.transform(X)
    
    result_shapes_of_transform_before_save_and_transform_after_load_are_identical = X_test.shape == X_test_after_upload.shape
    shutil.rmtree('./stacker_saved_model')
    assert result_shapes_of_transform_before_save_and_transform_after_load_are_identical
