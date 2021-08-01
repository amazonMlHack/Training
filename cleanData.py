# Put the notebook cleaning code here 

# also create a dummy function which can process all the cleaning we have done on a single record 
# with train file name train.csv 
# with test file name test.csv 

# IMPORTS
import pandas as pd
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string

# init objects
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
wn=WordNetLemmatizer()

nltk.download('stopwords')
en_stopwords = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


# cleaning functions
    
# extras
def needed_isupper(word):
    if word.isupper():
        return True 
    else:
        for ch in list(word):
            if ord(ch)>=ord('a') and ord(ch)<=ord('z'):
                return False
            else:
                continue
        return True

def removeSingleDouble(tokens):
    bigWords=[]
    for t in tokens:
        if len(t)==1 or len(t)==2:
            # discard 
            continue
        else:
            bigWords.append(t)
    return bigWords

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# main functions
def cleanTitle(title):
    if pd.isnull(title):
        return ""
    title = re.sub(r"\([^()]*\)", "", title) 
    tokens = tokenizer.tokenize(title)
    temp = []
    for word in tokens:
        if needed_isupper(word)==False:
            temp.append(word)
    tokens=temp
    new_tokens = [token.lower() for token in tokens if token.lower() not in en_stopwords]
    stemmed_tokens = [wn.lemmatize(token) for token in new_tokens]
    bigTokens=removeSingleDouble(stemmed_tokens)
    cleaned_title = ' '.join(bigTokens)
    return cleaned_title

def cleanDescription(desc):
    temp = []
    if pd.isnull(desc):
        return ""
    if desc=="":
        return desc
    desc = re.sub('[^A-Za-z]+', ' ', desc) # to remove special chars and  numbers  
    desc = cleanhtml(desc)
    tokens = tokenizer.tokenize(desc)
    temp = []
    for word in tokens:
        if needed_isupper(word)==False:
            temp.append(word)
    tokens=temp
    new_tokens = [token.lower() for token in tokens if token.lower() not in en_stopwords]
    stemmed_tokens = [wn.lemmatize(token) for token in new_tokens]
    bigTokens=removeSingleDouble(stemmed_tokens)
    cleaned_desc = ' '.join(bigTokens)
    return cleaned_desc

'''
how to remove punctuation 
# string.punctuation
# '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# '!Hello.'.strip(string.punctuation)
# 'Hello'

# ' '.join(word.strip(string.punctuation) for word in "Hello, world. I'm a boy, you're a girl.".split())
# "Hello world I'm a boy you're a girl"  
'''

def cleanBulletPoints(bullets):
    if pd.isnull(bullets):
        return ""
  
    if bullets[0]=='[' and bullets[-1]==']':
        bullets=bullets[1:-1] 
  
    bullets = ' '.join(word.strip(string.punctuation) for word in bullets.split())
    tokens = tokenizer.tokenize(bullets)
    temp = []
    for word in tokens:
        if needed_isupper(word)==False:
            temp.append(word)
    tokens=temp
    new_tokens = [token.lower() for token in tokens if token.lower() not in en_stopwords]
    stemmed_tokens = [wn.lemmatize(token) for token in new_tokens]
    bigTokens=removeSingleDouble(stemmed_tokens)
    cleaned_bullet = ' '.join(bigTokens)
    return cleaned_bullet

def cleanBrand(brand):
    if pd.isnull(brand):
        return ""
    else:
        return brand.lower()

def cleanBrowse_Node_ID(id):
    return [id]


# clean functions

# 1- For Train
def cleanDataTrain(dfTrain):
    # cleaned Dataframe
    df = pd.DataFrame(columns=['Title', 'Description', 'Bullet Points', 'Brand', 'Browse-Node-ID'])
    count=0
    for ind in dfTrain.index:
        count = count + 1
        cleanTitle_col = cleanTitle(dfTrain['TITLE'][ind])
        cleanDesc_col = cleanDescription(dfTrain['DESCRIPTION'][ind])
        cleanPts_col = cleanBulletPoints(dfTrain['BULLET_POINTS'][ind])
        cleanBrand_col = cleanBrand(dfTrain['BRAND'][ind])
        cleanID_col = cleanBrowse_Node_ID(dfTrain['BROWSE_NODE_ID'][ind])
        df.loc[len(df.index)] = [cleanTitle_col, cleanDesc_col, cleanPts_col, cleanBrand_col, cleanID_col] 
        if count % 1000 == 0:
            print("count :",count)
    return df
# 2- For Test
def cleanDataTest(dfTest):
    # cleaned Dataframe
    df = pd.DataFrame(columns=['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND'])
    count=0
    for ind in dfTest.index:
        count = count + 1
        cleanTitle_col = cleanTitle(dfTest['TITLE'][ind])
        cleanDesc_col = cleanDescription(dfTest['DESCRIPTION'][ind])
        cleanPts_col = cleanBulletPoints(dfTest['BULLET_POINTS'][ind])
        cleanBrand_col = cleanBrand(dfTest['BRAND'][ind])
        df.loc[len(df.index)] = [cleanTitle_col, cleanDesc_col, cleanPts_col, cleanBrand_col] 
        if count % 1000 == 0:
            print("count :",count)
    return df

# concatanation function 
def concatanateDataSet(tempDF):
    cols = len(tempDF.columns) # if cols are 5 -> train, if cols are 4 -> test
    if cols == 5:
        # train data frame
        df = pd.DataFrame(columns=['TEXT', 'BROWSE_NODE_ID'])
        for ind in tempDF.index:
            text = tempDF['TITLE'][ind] +  tempDF['DESCRIPTION'][ind] +  tempDF['BULLET_POINTS'][ind] + tempDF['BRAND'][ind]
            id = tempDF['BROWSE_NODE_ID'][ind]
            df.loc[len(df.index)] = [text, id]            
    if cols == 4:
        # test data frame
        df = pd.DataFrame(columns=['TEXT'])
        for ind in tempDF.index:
            text = tempDF['TITLE'][ind] +  tempDF['DESCRIPTION'][ind] +  tempDF['BULLET_POINTS'][ind] + tempDF['BRAND'][ind]
            df.loc[len(df.index)] = [text]  
    return df         
    

# MAIN function
def cleaner(input_dir='./Dataset/'):
    # read the train and test files *FIll in the corrent Addresses
    train = pd.read_csv('/amazonDataset/train.csv', escapechar="\\", quoting=csv.QUOTE_NONE)
    test = pd.read_csv('/amazonDataset/test.csv', escapechar="\\", quoting=csv.QUOTE_NONE)
    
    # get cleaned DFs
    tempTrainDF = cleanDataTrain(train)
    tempTestDF = cleanDataTest(test)
    
    # optional - store into csv's before concatanation
    
    tempTrainDF.to_csv(input_dir + "cleanedTrain.csv")
    tempTestDF.to_csv(input_dir + "cleanedTest.csv")
    
    # Concatanation in both DFs
    trainDF = concatanateDataSet(tempTrainDF)
    testDF = concatanateDataSet(tempTestDF)
    
    # conversion of final DFs into CSV
    trainDF.to_csv(input_dir + "finalCleanedTrain.csv")
    testDF.to_csv(input_dir + "finalCleanedTest.csv")
    