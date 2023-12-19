import os
import tensorflow as tf
import keras
import pickle
from verstack import Stacker

# LOAD STACKER INSTANCE
def load_stacker(path):
    '''Load a stacker instance from a path.
    
    Parameters
    ----------
    path : str
        Path to the saved_stacker_model folder
        
    Returns
    -------
    stacker_instance : Stacker
        The trained stacker instance'''

    stacker_instance = unpickle_stacker_instance(path)
    stacker_instance = create_layers_to_load_models(stacker_instance)
    layers = get_arrange_layers(path)

    for layer in layers:
        layer_path = os.path.join(path, layer)
        models_ixs = sorted(os.listdir(layer_path))
        for model_ix in models_ixs:
            model_path = os.path.join(layer_path, model_ix)
            try:
                model = load_stacker_wrapper_model(model_path)
            except Exception:
                model = unpickle_object(model_path)
            stacker_instance = add_model_to_layer(stacker_instance, model, layer)

    return stacker_instance

def add_model_to_layer(stacker_instance, model, layer):
    '''Add model to the stacker instance
    
    Parameters
    ----------
    stacker_instance : Stacker
        stacker instance
    model : object
        model to add to the stacker instance
    layer : str
        layer to add the model to
        
    Returns
    -------
    stacker_instance : Stacker
        stacker instance with the model added to the layer'''
    stacker_top_layer = '_'.join(layer.split('_')[:2])
    if layer not in stacker_instance.trained_models[stacker_top_layer]:
        stacker_instance.trained_models[stacker_top_layer][layer] = []
    stacker_instance.trained_models[stacker_top_layer][layer].append(model)
    return stacker_instance

def unpickle_object(path):
    '''Load a pickled object'''
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_stacker_wrapper_model(path):
    '''Load the verstack wrapper for keras model'''
    with open(f'{path}/verstack.stacking.kerasModel', 'rb') as f:
        model = pickle.load(f)
    # load actual keras model
    keras_model = tf.keras.models.load_model(path)
    model.model = keras_model
    return model

def unpickle_stacker_instance(path):
    '''Unpickle the stacker instance (without trained models)'''
    with open(f'{path}/stacker.p', 'rb') as f:
        stacker_instance = pickle.load(f)
    return stacker_instance
    
def create_layers_to_load_models(stacker_instance):
    '''Create a dict to store the trained models'''
    for layer in stacker_instance.layers.keys():
        stacker_instance.trained_models[layer] = {}
    return stacker_instance

def get_arrange_layers(path):
    '''Get the layers in the correct order'''
    layers = os.listdir(path)
    layers = sorted([l for l in layers if 'layer' in l])
    layers = [val.split('_') for val in layers]
    layers = sorted(layers, key = lambda x: (int(x[1]), int(x[2])))
    layers = ['_'.join(lst) for lst in layers]
    return layers
