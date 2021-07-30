import json
f = open('parameters.json','r')
param = json.load(f)
f.close()

train=param['train']==1
processData=param['processData']==1
user_test = param['userTest'] == 1

# To prepare prediction file when model is already trained 
from evaluate import userTest
if user_test:
    userTest()
    exit()

# Loading Datset
from preprocess import load_data_self_preprocess
(xTrain,yTrain),(xTest,yTest)= load_data_self_preprocess(processData=processData)

# Loading Code to Train Model
from modelTrainer import trainModel
if train==True:
    trainModel(xTrain,yTrain)

# Loading Code to Evaluate the results on test data
from evaluate import generateReport
generateReport(xTest,yTest)

# Save Model Diagram (not working as of now on windows)
from evaluate import saveModelArchitecture
saveModelArchitecture()







