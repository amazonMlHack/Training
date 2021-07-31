import json

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

def model_framework():
    # this model architecture needs to be changed 
    return architecture2(vocabSize=vocabSize, sequence_length=sequence_length,dropout_rate=0.3)
