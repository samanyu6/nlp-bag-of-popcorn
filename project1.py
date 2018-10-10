import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup as bs
import nltk
from nltk.corpus import stopwords as sw
import os
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.ensemble import RandomForestClassifier as rf

#instead of '/home/Desktop/sentAnalysis/labeledTrainData.csv' put the directory where you've stored the datasets downloaded from kaggle
train = pd.read_csv('/home/Desktop/sentAnalysis/labeledTrainData.tsv', header = 0, delimiter = '\t')
test = pd.read_csv('/home/Desktop/sentAnalysis/testData.tsv',header = 0, delimiter = '\t')

def para2words(para):
    para_text = bs(para).get_text()
    para_lettersonly = re.sub("[^a-zA-Z]","",para_text)
    para_words = para_lettersonly.lower().split()
    stops = set(sw.words("english"))
    para_meaning = [n for n in para_words if not n in stops]
    return(" ".join(para_meaning))


num_reviews = train["review"].size
clean_train = []
for i in range(0,num_reviews):
    clean_train.append(para2words(train["review"][i]))

vector = cv(analyzer = "word", tokenizer = None, preprocessor= None, max_features = 5000)

train_df = vector.fit_transform(clean_train)
train_df = train_df.toarray()

forest = rf(n_estimators=150)
forest = forest.fit(train_df,train["sentiment"])

numofrev = len(test["review"])
cleanreview = []
for i in range(0,numofrev):
    cleanreview.append( para2words(test["review"][i]))

test_df = vector.transfor(cleanreview)
test_df = test_df.toarray()

result = forest.predict(test_df)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result})
output.to_csv(op.csv)
