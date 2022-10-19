import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import load_data

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras import utils
from tensorflow.keras import losses
from tensorflow.keras import backend as K

import gc

class MTLDNN:
    def __init__(self, ni, params):
        self.model = MTLDNN.build_model(ni, params)
        
    def clonar(self, ni, params):
        self.model = MTLDNN.build_model(ni, params)
        
    def set_seed(s):
        # 0. clear backend session
        K.clear_session()
        
        # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
        import os
        os.environ['PYTHONHASHSEED']=str(s)

        # 2. Set `python` built-in pseudo-random generator at a fixed value
        import random
        random.seed(s)

        # 3. Set `numpy` pseudo-random generator at a fixed value
        import numpy as np
        np.random.seed(s)

        # 4. Set the `tensorflow` pseudo-random generator at a fixed value
        import tensorflow as tf
        tf.random.set_seed(s)
        # for later versions: 
        # tf.compat.v1.set_random_seed(seed_value)

    def build_model(ni, params):
        act = params.get('act')
        n, n1, n2, n3 = params.get('arq')
        prob_h1, prob_h2, prob_h3 = params.get('dropout')
        momentum_batch_norm = params.get('momentum_batch_norm')
        
        # ENTRADA
        model_input = Input(shape=(ni,))
    
        # CAPAS COMUNES A TODAS LOS TARGETS
        x = Dense(n, activation=act)(model_input)
        # x = BatchNormalization(momentum=momentum_batch_norm)(x)
        x = Dropout(prob_h1)(x)
        x = Dense(n1, activation=act)(x)
        # x = BatchNormalization(momentum=momentum_batch_norm)(x)
        x = Dropout(prob_h2)(x)
        x = Dense(n2, activation=act)(x)
        # x = BatchNormalization(momentum=momentum_batch_norm)(x)
        x = Dropout(prob_h3)(x)
    
        # CAPAS ESPECIFICAS A CADA TARGET (1 a 10)
        y1 = Dense(units = 1, activation= 'sigmoid', name = 'output_1')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y2 = Dense(units = 1, activation= 'sigmoid', name = 'output_2')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y3 = Dense(units = 1, activation= 'sigmoid', name = 'output_3')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y4 = Dense(units = 1, activation= 'sigmoid', name = 'output_4')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y5 = Dense(units = 1, activation= 'sigmoid', name = 'output_5')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y6 = Dense(units = 1, activation= 'sigmoid', name = 'output_6')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y7 = Dense(units = 1, activation= 'sigmoid', name = 'output_7')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y8 = Dense(units = 1, activation= 'sigmoid', name = 'output_8')(Dropout(prob_h3)(Dense(n, activation=act)(x)))    
        y9 = Dense(units = 1, activation= 'sigmoid', name = 'output_9')(Dropout(prob_h3)(Dense(n, activation=act)(x)))
        y10 =Dense(units = 1, activation= 'sigmoid', name = 'output_10')(Dropout(prob_h3)(Dense(n, activation=act)(x)))

        # MODELO FINAL
        model = Model(inputs=model_input, outputs=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10])
        return model
    
    def compilar(self, params):
        # set seed before compiling
        MTLDNN.set_seed(s=42)
        
        loss = params.get('loss')
        optimizer = Adam(params.get('l_rate'))
        metrics = params.get('metrics')
        self.model.compile(loss=loss, optimizer=optimizer, metrics = metrics)
    
    def entrenar(self, data, params):
        MTLDNN.set_seed(s=42)
        
        # data
        train, test = data
        x_train, y_train = train
        x_test, y_test = test
        
        #early stopping
        min_delta_val = params.get('min_delta')
        patience_val = params.get('patience')
        w = params.get('w')
        
        # early stopping
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta_val,
                                   patience=patience_val,
                                   verbose=2,
                                   mode='min',
                                   restore_best_weights=True)
        # checkpoint
        ckpt_name = 'model.h5'
        ckpt = ModelCheckpoint(ckpt_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')

        # fit
        n_epochs = params.get('n_epochs')
        
        self.model.fit(x = x_train,
                       y = y_train,
                       epochs = n_epochs,
                       batch_size = 128,
                       validation_data = (x_test,y_test),
                       callbacks = [early_stop, ckpt],
                       class_weight = w,
                       verbose = 0)
        
    def predecir(self, X):
        
        # computar predicciones
        pred = self.model.predict(X)
            
        # obtener etiquetas predichas para cada output
        y0 = np.where(pred[0] > 0.5, 1,0)
        y1 = np.where(pred[1] > 0.5, 1,0)
        y2 = np.where(pred[2] > 0.5, 1,0)
        y3 = np.where(pred[3] > 0.5, 1,0)
        y4 = np.where(pred[4] > 0.5, 1,0)
        y5 = np.where(pred[5] > 0.5, 1,0)
        y6 = np.where(pred[6] > 0.5, 1,0)
        y7 = np.where(pred[7] > 0.5, 1,0)
        y8 = np.where(pred[8] > 0.5, 1,0)
        y9 = np.where(pred[9] > 0.5, 1,0)
        
        # obtener etiquetas para task_1
        preds = [y0 , y1 , y2 , y3 , y4 , y5 , y6 , y7 , y8 , y9 ]
        
        return preds