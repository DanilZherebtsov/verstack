import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# accomodate for older keras versions
try:
    from keras.utils.np_utils import to_categorical as to_cat
except:
    from keras.utils import to_categorical as to_cat
from verstack import Factorizer
tf.get_logger().setLevel('ERROR')

class kerasModel:
    
    def __init__(self, objective, num_layers = 3, epochs = 200, verbose = True):
        '''
        Automatically configurable neural network Regressor/Classifier with mostly fixed architecture.
            - Regressor/Classifier and loss function (mse/binary_crossentropy/categorical_crossentropy are defined by objective
            - number of layers is defined by num_layers argument
            - number of epochs is defined by epochs argument

        Parameters
        ----------
        objective : str
            regression/multiclass/binary for defining the model configuration.
        num_layers : int, optional
            number of layers. can take values from 1 to 3. The default is 3.
        epochs : int, optional
            number of epochs. The default is 200.

        Returns
        -------
        None.

        '''
        self.objective = objective
        self.num_classes = None
        self.num_layers = num_layers
        self.model = None
        self.target_encoder = None
        self.epochs = epochs
        self.verbose = verbose

    def _get_params_from_objective(self, intermediate_activation = False, output_layer = False, loss = False): # self
        '''Define one of the three network parameters
            - intermediate_activation: activation function for all but final layer
            - output_layer: output layer configured based on objective and number of output classes
            - loss: loss function name based on objective
            
        '''
        if intermediate_activation:
            return 'relu' if self.objective == 'regression' else 'tanh'
        if output_layer: 
            if self.objective == 'regression':
                layer = layers.Dense(1)
            elif self.objective == 'binary':
                layer = layers.Dense(2, activation = 'sigmoid')
            else:
                layer = layers.Dense(self.num_classes, activation = 'softmax')
            return layer
        if loss:
            if self.objective == 'regression':
                loss_func_name = 'mean_squared_error'
            elif self.objective == 'binary':
                loss_func_name = 'binary_crossentropy'
            else:
                loss_func_name = 'categorical_crossentropy'
            return loss_func_name

    def _get_architecture(self, normalizer):
        '''Create one of the three predefined architectures with 1/2/3 layers'''
        activation = self._get_params_from_objective(intermediate_activation=True)
        output_layer = self._get_params_from_objective(output_layer=True)
        
        if self.num_layers == 3:
            architecture = [normalizer,
                            layers.Dense(64, activation = activation),
                            layers.Dense(16, activation = activation),
                            layers.Dense(64, activation = activation),
                            output_layer]
        elif self.num_layers == 2:
            architecture = [normalizer,
                            layers.Dense(128, activation = activation),
                            layers.Dense(16, activation = activation),
                            output_layer]
        else:
            architecture = [normalizer,
                            layers.Dense(64, activation = activation),
                            output_layer]
        return architecture

    def build_and_compile_model(self, norm):
        '''Build and compile model'''
        model = keras.Sequential(self._get_architecture(norm))

        model.compile(loss = self._get_params_from_objective(loss=True),
                      optimizer = tf.keras.optimizers.Adam(0.001))
        return model

    def transform_categoric_target(self, y):
        '''Factorize categoric target for classification tasks, save encoder for inverse_transform after predict'''
        enc = Factorizer()
        y = enc.fit_transform(pd.DataFrame(np.array(y), columns = ['target']), 'target')
        y = to_cat(y)
        self.target_encoder = enc
        return y

    def fit(self, X, y):
        if self.verbose:
            verbose = 1
        else:
            verbose = 0
        '''Fit model on X, y, save model to self.model'''
        self.num_classes = y.nunique()
        normalizer = preprocessing.Normalization()
        normalizer.adapt(np.array(X))
        model = self.build_and_compile_model(normalizer)
        if self.objective in ['binary', 'multiclass']:
            y = self.transform_categoric_target(y)
        model.fit(X, y, validation_split=0.2, verbose=verbose, epochs = self.epochs)
        self.model = model
        
    def predict(self, X):
        '''Predict with saved model on X, inverse_transform predictions if classification'''
        pred = self.model.predict(X)          
        if self.objective == 'binary':
            pred = pred[:,1]    
        elif self.objective == 'multiclass':
            pred = np.argmax(pred, 1)
            if self.target_encoder:
                pred = np.array(self.target_encoder.inverse_transform(pd.DataFrame(pred, columns = ['target'])))
        return pred