import json
import re

from tensorflow.python.keras.layers.core import Dense
f = open('parameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
vocabSize=param['vocabSize']
sequence_length=param['sequence_length']
#end

# architecture on keras website 
def architecture1(vocabSize=5000,sequence_length=120,dropout_rate=0.3):
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    SCHEMA = [
        layers.Embedding( vocabSize , 10, input_length=sequence_length ),
        layers.Dropout(0.5),
        layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
        layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
        layers.GlobalMaxPooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="linear", name="predictions")
    ]
    model = Sequential(SCHEMA)
    model.compile(
        loss='mean_squared_logarithmic_error', 
        optimizer=Adam(),
        metrics=['mse']
    )
    return model

def architecture2(vocabSize=5000,sequence_length=120,dropout_rate=0.3):
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    SCHEMA = [
        layers.Embedding( vocabSize , 10, input_length=sequence_length ),
        layers.Dropout(0.5),
        layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
        layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
        layers.GlobalMaxPooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="linear", name="predictions")
    ]
    model = Sequential(SCHEMA)
    # model.compile(
    #     loss='mean_squared_logarithmic_error', 
    #     optimizer=Adam(),
    #     metrics=['mse']
    # )
    model.compile(
        loss='mean_squared_error',# 'mean_squared_logarithmic_error', 
        optimizer= Adam(),
        metrics=['mse','acc']
    )
    return model

# model for sentiment analysis on imdb dataset 
def architecture3(vocabSize=5000,sequence_length=120,dropout_rate=0.3):
    from tensorflow.keras.activations import relu
    from tensorflow.keras.layers import Embedding,LSTM, Dropout, Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import plot_model
    dropout_rate = 0.3
    from tensorflow.keras.activations import relu
    activation_func = relu
    SCHEMA = [
        Embedding( vocabSize , 10, input_length=sequence_length ),
        LSTM( 32 ) ,
        LSTM( 32 ) ,
        Dropout(dropout_rate),
        Dense( 256 , activation=activation_func ) ,
        Dropout(dropout_rate),
        Dense( 128 , activation=activation_func ) ,
        Dropout(dropout_rate),
        Dense( 64 , activation=activation_func ) ,
        Dropout(dropout_rate),
        Dense( 32 , activation=activation_func ) ,
        Dropout(dropout_rate),
        Dense(1, activation="linear", name="predictions")
    ]
    model = Sequential(SCHEMA)
    model.compile(
        loss='mean_squared_logarithmic_error', 
        optimizer= 'sgd',  #Adam()
        metrics=['mse','acc']
    )
    # model.compile(loss='mean_squared_error',
    #           optimizer='sgd',
    #           metrics=['mae', 'acc'])
    return model

import tensorflow as tf
import math
def summation(n, x):
    ans = tf.constant(0)
    
    for i in range(1,n+1):
        a = tf.cast(math.pi, tf.float32)
        b = tf.cast(tf.math.multiply(tf.cast(i, type(x)),x), tf.float32)
        c = tf.math.multiply(a,b)
        d = tf.math.multiply(tf.cast((2*math.pi), type(c)), c)
        e = tf.math.divide(tf.math.sin(d),tf.cast(i))
        f = tf.multiply(tf.case(tf.math.pow(1, i), type(e)),e)
        ans = tf.math.add(tf.cast(ans, type(f)),f)

def roundValue(x):
    z = tf.math.multiply(tf.cast((1/math.pi), type(summation(5,x))),summation(5,x))
    return tf.math.add(tf.cast(x,type(z)),z)

def custom_activation(x):
    from keras import backend as K
    return roundValue(x)

def architecture4(vocabSize=5000,sequence_length=120,dropout_rate=0.3):
    # Custom activation function
    from keras.layers import Activation
    from keras import backend as K
    from keras.utils.generic_utils import get_custom_objects

    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    
    import tensorflow as tf 
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    SCHEMA = [
        layers.Embedding( vocabSize , 10, input_length=sequence_length ),
        layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3),
        layers.GlobalMaxPooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1,activation="relu", name="predictions"),
        layers.Activation(custom_activation),
    ]
    model = Sequential(SCHEMA)
    # model.compile(
    #     loss='mean_squared_logarithmic_error', 
    #     optimizer=Adam(),
    #     metrics=['mse']
    # )
    model.compile(
        loss='mean_squared_error',# 'mean_squared_logarithmic_error', 
        optimizer= Adam(),
        metrics=['mse','acc']
    )
    return model


def model_framework():
    # this model architecture needs to be changed 
    return architecture4(vocabSize=vocabSize, sequence_length=sequence_length,dropout_rate=0.3)
