# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:07:32 2019

@author: SURAJ BHADHORIYA
"""

#load libraries
import pandas as pd
from sklearn.cross_validation import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB
#read datset
data=pd.read_csv("amazon_dataset.csv")
feature=["review"]
label=["rating"]
#make different dataframe
df=pd.DataFrame(data[feature+label][:40000])

#fillna values
df=df.fillna({'review':''})
#remove punchuation
df['review']=df.apply(lambda row:re.sub(r'\W+|\d+|_', ' ',row["review"]), axis=1)
print(df['review'][23])

#word tokenize                        
df["review_sim"] = df.apply(lambda row: word_tokenize(row["review"]), axis=1)
print(df['review_sim'][23])
#stop word 
stop_word=set(stopwords.words('english'))
print(len(stop_word))
print(stop_word)
#removing stop_word
df['new_review']=df['review_sim'].apply(lambda row:[wrd for wrd in row if  wrd not in stop_word ])
print(df['review_sim'][23])
print(df['new_review'][23])

#stemming
stemmer=PorterStemmer()
df['stem_review']=df['new_review'].apply(lambda row:[stemmer.stem(w) for w in row])
print(df['stem_review'][23])

#lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
df['stem_review']=df['stem_review'].apply(lambda row:[lemmatizer.lemmatize(w) for w in row])
print(df['stem_review'][21])
#make x and y with normalization
df['stem_review']=df['stem_review'].apply(lambda row:[w.lower() for w in row])
print(df['stem_review'][21])

df=df[df['rating'] !=3]
print(df['rating'])
df['sentimate']=df['rating'].apply(lambda rating:+1 if rating>3 else -1)
print(df['sentimate'])
x=df['stem_review']
y=df['sentimate']

#split data
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
#`document term matrix or bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer='word')
print(vectorizer)
train_matrix=vectorizer.fit_transform(X_train)
test_matrix=vectorizer.transform(X_test)
print(train_matrix)
print(test_matrix)
#naive bayes classification 
clf=MultinomialNB()
clf.fit(train_matrix,y_train)
#accuracy
accuracy=clf.score(test_matrix,y_test)
print(accuracy)
#prediction
pre=clf.predict(X_test)
print(pre)

