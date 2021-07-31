import json
import re
f = open('parameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
vocabSize=param['vocabSize']
sequence_length=param['sequence_length']
#end
train_path = "./Dataset/" # source data
test_path = "./Dataset/" # test data for evaluation.

import pandas as pd

'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output.
Name - Name of the csv file
Train - If train is True, both the data and labels are returned. Else only the data is returned
'''
def retrieve_data(input_dir='./Dataset/',name="train.csv"):
    data_dir = input_dir + name
    data = pd.read_csv(data_dir)
    cols = len(data.columns)
    if cols == 1:
        # x = v.fit_transform(df['Review'].values.astype('U'))  ## Even astype(str) would work
        X = data['TEXT'].astype(str)
        return X
    if cols == 2:    
        X = data['TEXT'].astype(str)
        Y = data['BROWSE_NODE_ID'].astype(int)
        return X, Y

'''
TFIDF_PROCESS takes the data to be fit as the input and returns a vectorizer of the tfidf as output
Data - The data for which the bigram model has to be fit
'''
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_process(data,max_features=vocabSize):
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf = True)#, min_df = 0.02, max_df = 0.97)
    vectorizer.fit(data)
    return vectorizer

# Padding the sequences to a fixed length
from tensorflow.keras.preprocessing.sequence import pad_sequences
def add_padding_to_Xdata(xTrain,xTest, sequence_length):
    xTrain = pad_sequences( xTrain , maxlen=sequence_length , padding='pre', value=0 )
    xTest = pad_sequences( xTest , maxlen=sequence_length , padding='pre', value=0 )
    return xTrain,xTest

def sanityEmbeddings(processedText, vectorizer, tokenizer):
    documentTermMatrix = vectorizer.transform(processedText).toarray()
    vocabDictionary = vectorizer.vocabulary_
    intEmbeddings = []
    temp = []
    i = 0

    for document in processedText:
        topTfidfEmbeddings = {vocabDictionary.get(token) : documentTermMatrix[i][vocabDictionary.get(token)] for token in tokenizer(document) if vocabDictionary.get(token) is not None}

        #take middle 90% of the sorted embeddings based on tfidf score
        topTfidfEmbeddings = dict(sorted(topTfidfEmbeddings.items(), key = lambda item: item[1], reverse = False)
        [round(len(topTfidfEmbeddings) * 0.05):
        round(len(topTfidfEmbeddings) * 0.95)])

        for token in tokenizer(document):
            embedding = vocabDictionary.get(token)

            if embedding in topTfidfEmbeddings:
                temp.append(embedding)

        i += 1

        intEmbeddings.append(temp)
        temp = []

    return intEmbeddings


import time
import pickle
from cleanData import cleaner
def load_data_self_preprocess(processData=True):
    start = time.time()
    if processData is True:
        feedback=cleaner()
        if feedback:
            print("successfully created csv files")
    (xTrain_text, yTrain) = retrieve_data(input_dir=train_path,name="finalCleanedTrain.csv")
    print ("Retrieved the training data. Now will retrieve the test data in the required format")
    (xTest_text,yTest) = retrieve_data(input_dir=test_path,name="finalCleanedTest.csv")
    print ("Retrieved the test data. Now will initialize the model \n\n")
    print("As per choice we will use vocabulary size as {}".format(vocabSize))
    print('We will try to fit our train data usinf tfidf_vectorizer')

    tfidf_vectorizer = tfidf_process(xTrain_text,max_features=vocabSize)
    tokenizer = tfidf_vectorizer.build_tokenizer()

    with open('vocab.pkl', 'wb') as pklFile:
        pickle.dump(tfidf_vectorizer.vocabulary_, pklFile)

    with open('vectorizer.pkl', 'wb') as pklFile:
        pickle.dump(tfidf_vectorizer, pklFile)

    xTrain = sanityEmbeddings(xTrain_text, tfidf_vectorizer, tokenizer)
    xTest = sanityEmbeddings(xTest_text, tfidf_vectorizer, tokenizer)
    xTrain,xTest=add_padding_to_Xdata(xTrain,xTest, sequence_length)
    end=time.time()
    print('The data preparation took {} ms'.format(end-start))
    return (xTrain,yTrain),(xTest,yTest)


if __name__=='__main__':
    print("This file is for preparing the Amazon Dataset")
    print("Not meant to be run directly")
