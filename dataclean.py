import inline as inline
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
from math import e
import itertools

import matplotlib.pyplot as plt

from sklearn import datasets
data = pd.read_csv("/Users/yaoqiangwu/Desktop/Comp551/comp551project2/training.1600000.processed.noemoticon.csv",names=["sentiment","2","3","4","5",'tweets'], encoding="ISO-8859-1")
#slice data by specify the num of row we rea for example if we read 160000 is 20 percents
data = data.tail(20000)
data = data.drop('2',1)
data = data.drop('3',1)
data = data.drop("4",1)
data = data.drop("5",1)
data['sentiment']=data['sentiment'].replace(4,1)
data['tweets']=data['tweets'].astype(str)
positives = data.loc[data.sentiment==1]
negative = data.loc[data.sentiment ==0]
urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = r'[^A-Za-z0-9]'

data['tweets']= data['tweets'].str.lower()
data['tweets'] = data['tweets'].str.replace("can't",'can not')
data['tweets'] = data['tweets'].str.replace("he's",'he is')
data['tweets'] = data['tweets'].str.replace("there's",'there is')
data['tweets'] = data['tweets'].str.replace("we're",'we are')
data['tweets'] = data['tweets'].str.replace("that's",'that is')
data['tweets'] = data['tweets'].str.replace("won't",'will not')
data['tweets'] = data['tweets'].str.replace("wasn't",'was not')
data['tweets'] = data['tweets'].str.replace("isn't",'is not')
data['tweets'] = data['tweets'].str.replace("what's",'what is')
data['tweets'] = data['tweets'].str.replace("haven't",'have not')
data['tweets'] = data['tweets'].str.replace("hasn't",'has not')
data['tweets'] = data['tweets'].str.replace("there's",'there is')
data['tweets'] = data['tweets'].str.replace("he's",'he is')
data['tweets'] = data['tweets'].str.replace("you're",'you are')
data['tweets'] = data['tweets'].str.replace("i'm",'i am')
data['tweets'] = data['tweets'].str.replace("shouldn't",'should not')
data['tweets'] = data['tweets'].str.replace("isn't", "is not")
data['tweets'] = data['tweets'].str.replace("aren't", "are not")
data['tweets'] = data['tweets'].str.replace("here's", "here is")
data['tweets'] = data['tweets'].str.replace("you've", "you have")
data['tweets'] = data['tweets'].str.replace("what's", "what is")
data['tweets'] = data['tweets'].str.replace("couldn't", "could not")
data['tweets'] = data['tweets'].str.replace("we've", "we have")
data['tweets'] = data['tweets'].str.replace("who's", "who is")
data['tweets'] = data['tweets'].str.replace("y'all", "you all")
data['tweets'] = data['tweets'].str.replace("would've", "would have")
data['tweets'] = data['tweets'].str.replace("it'll", "it will")
data['tweets'] = data['tweets'].str.replace("we'll", "we will")
data['tweets'] = data['tweets'].str.replace("he'll", "he will")
data['tweets'] = data['tweets'].str.replace("i'll","i will")
data['tweets'] = data['tweets'].str.replace("they'll", "they will")
data['tweets'] = data['tweets'].str.replace("they'd", "they would")
data['tweets'] = data['tweets'].str.replace("don't", "do not")
data['tweets'] = data['tweets'].str.replace("i'd", "i would")
data['tweets'] = data['tweets'].str.replace("we'd", "we would")
data['tweets'] = data['tweets'].str.replace("let's", "let us")
data['tweets'] = data['tweets'].str.replace("i've", "i have")
data['tweets'] = data['tweets'].str.replace("that's", "that is")
data['tweets'] = data['tweets'].str.replace("doesn't", "does not")
data['tweets'] = data['tweets'].str.replace("ain't", "am not")
data['tweets'] = data['tweets'].str.replace("i'd", "i would")
data['tweets'] = data['tweets'].str.replace("could've", "could have")
data['tweets'] = data['tweets'].str.replace("youve", "you have")
data['tweets'] = data['tweets'].str.replace("hrs", "hours")
data['tweets'] = data['tweets'].str.replace("yrs", "years")
data['tweets'] = data['tweets'].str.replace(urlPattern, ' ')
data['tweets'] = data['tweets'].str.replace(userPattern,' ')
data['tweets'] =data['tweets'].str.replace('          ',' ')
data['tweets'] =data['tweets'].str.replace('         ',' ')
data['tweets'] =data['tweets'].str.replace('        ',' ')
data['tweets'] =data['tweets'].str.replace('       ',' ')
data['tweets'] =data['tweets'].str.replace('      ',' ')
data['tweets'] =data['tweets'].str.replace('     ',' ')
data['tweets'] =data['tweets'].str.replace('    ',' ')
data['tweets'] =data['tweets'].str.replace('   ',' ')
data['tweets'] =data['tweets'].str.replace('  ',' ')
data['tweets'] = data['tweets'].str.strip()

data['tweets'] = data.tweets.apply(lambda x: x[0:-1].split(' '))
newdata = data[['tweets','sentiment']]


data2 = pd.read_csv("/Users/yaoqiangwu/Desktop/Comp551/comp551project2/testdata.manual.2009.06.14.csv",names=["sentiment","2","3","4","5",'tweets'])
data2 = data2.drop('2',1)
data2 = data2.drop('3',1)
data2 = data2.drop("4",1)
data2 = data2.drop("5",1)
data2["sentiment"]=data2["sentiment"].replace(4,1)
data2['sentiment']=data2['sentiment'].astype(str)

data2 = data2[data2["sentiment"].str.contains("2") == False]
data2['sentiment']=data2['sentiment'].astype(int)

data2['tweets']=data2['tweets'].astype(str)
positives = data2.loc[data2.sentiment==1]
negative = data2.loc[data2.sentiment ==0]
urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = r'[^A-Za-z0-9]'

data2['tweets']= data2['tweets'].str.lower()
data2['tweets'] = data2['tweets'].str.replace("can't",'can not')
data2['tweets'] = data2['tweets'].str.replace("he's",'he is')
data2['tweets'] = data2['tweets'].str.replace("there's",'there is')
data2['tweets'] = data2['tweets'].str.replace("we're",'we are')
data2['tweets'] = data2['tweets'].str.replace("that's",'that is')
data2['tweets'] = data2['tweets'].str.replace("won't",'will not')
data2['tweets'] = data2['tweets'].str.replace("wasn't",'was not')
data2['tweets'] = data2['tweets'].str.replace("isn't",'is not')
data2['tweets'] = data2['tweets'].str.replace("what's",'what is')
data2['tweets'] = data2['tweets'].str.replace("haven't",'have not')
data2['tweets'] = data2['tweets'].str.replace("hasn't",'has not')
data2['tweets'] = data2['tweets'].str.replace("there's",'there is')
data2['tweets'] = data2['tweets'].str.replace("he's",'he is')
data2['tweets'] = data2['tweets'].str.replace("you're",'you are')
data2['tweets'] = data2['tweets'].str.replace("i'm",'i am')
data2['tweets'] = data2['tweets'].str.replace("shouldn't",'should not')
data2['tweets'] = data2['tweets'].str.replace("isn't", "is not")
data2['tweets'] = data2['tweets'].str.replace("aren't", "are not")
data2['tweets'] = data2['tweets'].str.replace("here's", "here is")
data2['tweets'] = data2['tweets'].str.replace("you've", "you have")
data2['tweets'] = data2['tweets'].str.replace("what's", "what is")
data2['tweets'] = data2['tweets'].str.replace("couldn't", "could not")
data2['tweets'] = data2['tweets'].str.replace("we've", "we have")
data2['tweets'] = data2['tweets'].str.replace("who's", "who is")
data2['tweets'] = data2['tweets'].str.replace("y'all", "you all")
data2['tweets'] = data2['tweets'].str.replace("would've", "would have")
data2['tweets'] = data2['tweets'].str.replace("it'll", "it will")
data2['tweets'] = data2['tweets'].str.replace("we'll", "we will")
data2['tweets'] = data2['tweets'].str.replace("he'll", "he will")
data2['tweets'] = data2['tweets'].str.replace("i'll","i will")
data2['tweets'] = data2['tweets'].str.replace("they'll", "they will")
data2['tweets'] = data2['tweets'].str.replace("they'd", "they would")
data2['tweets'] = data2['tweets'].str.replace("don't", "do not")
data2['tweets'] = data2['tweets'].str.replace("i'd", "i would")
data2['tweets'] = data2['tweets'].str.replace("we'd", "we would")
data2['tweets'] = data2['tweets'].str.replace("let's", "let us")
data2['tweets'] = data2['tweets'].str.replace("i've", "i have")
data2['tweets'] = data2['tweets'].str.replace("that's", "that is")
data2['tweets'] = data2['tweets'].str.replace("doesn't", "does not")
data2['tweets'] = data2['tweets'].str.replace("ain't", "am not")
data2['tweets'] = data2['tweets'].str.replace("i'd", "i would")
data2['tweets'] = data2['tweets'].str.replace("could've", "could have")
data2['tweets'] = data2['tweets'].str.replace("youve", "you have")
data2['tweets'] = data2['tweets'].str.replace("hrs", "hours")
data2['tweets'] = data2['tweets'].str.replace("yrs", "years")
data2['tweets'] = data2['tweets'].str.replace(urlPattern, ' ')
data2['tweets'] = data2['tweets'].str.replace(userPattern,' ')
data2['tweets'] =data2['tweets'].str.replace('          ',' ')
data2['tweets'] =data2['tweets'].str.replace('         ',' ')
data2['tweets'] =data2['tweets'].str.replace('        ',' ')
data2['tweets'] =data2['tweets'].str.replace('       ',' ')
data2['tweets'] =data2['tweets'].str.replace('      ',' ')
data2['tweets'] =data2['tweets'].str.replace('     ',' ')
data2['tweets'] =data2['tweets'].str.replace('    ',' ')
data2['tweets'] =data2['tweets'].str.replace('   ',' ')
data2['tweets'] =data2['tweets'].str.replace('  ',' ')
data2['tweets'] = data2['tweets'].str.strip()

data2['tweets'] = data2.tweets.apply(lambda x: x[0:-1].split(' '))






vectorizer = CountVectorizer(analyzer=lambda x: x)
train=vectorizer.fit_transform(newdata['tweets'])
traindata = vectorizer.fit_transform(newdata['tweets']).toarray()
sentiment = newdata['sentiment'].to_numpy()
test = vectorizer.transform(data2['tweets'])
testdata = vectorizer.transform(data2['tweets']).toarray()
testsentiment = data2['sentiment'].to_numpy()

tfidf_transformer = TfidfTransformer()
traintfidf = tfidf_transformer.fit_transform(train).toarray()
testtfidf = tfidf_transformer.transform(test).toarray()










