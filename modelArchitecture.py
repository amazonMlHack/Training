import json
f = open('userDefinedParameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
vocabSize=param['vocabSize']
sequence_length=param['sequence_length']
#end

# # model for sentiment analysis on imdb dataset 
# def classification_model_new_LSTM(vocabSize=5000,sequence_length=120,dropout_rate=0.3):
#     from tensorflow.keras.activations import relu
#     from tensorflow.keras.layers import Embedding,LSTM, Dropout, Dense
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.utils import plot_model
#     dropout_rate = 0.3
#     from tensorflow.keras.activations import relu
#     activation_func = relu
#     SCHEMA = [
#         Embedding( vocabSize , 10, input_length=sequence_length ),
#         LSTM( 32 ) ,
#         Dropout(dropout_rate),
#         Dense( 32 , activation=activation_func ) ,
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ]
#     model = Sequential(SCHEMA)
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam() ,
#         metrics=[ 'accuracy' ]
#     )
#     return model

def model_framework():
    # this model architecture needs to be changed 
    return classification_model_new_LSTM(vocabSize=vocabSize, sequence_length=sequence_length,dropout_rate=0.3)
