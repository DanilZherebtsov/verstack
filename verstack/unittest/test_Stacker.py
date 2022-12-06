import sys
sys.path.append('../../')
import numpy as np
import pytest
from common import generate_data
from verstack import Stacker

# test overall Stacker not being broken
def test_Stacker():
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