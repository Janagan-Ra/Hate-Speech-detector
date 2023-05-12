#!/usr/bin/env python
# coding: utf-8

# In[67]:


#Data set from Kaggle: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
import numpy as np
import pandas as pd
import nltk
import string
import re
from sklearn.feature_extraction. text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

stopWord = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english')


# # Intial preview of dataset.

# In[68]:


hateSpeechSet = pd.read_csv("labeled_data.csv")
print(hateSpeechSet.head(5))


# In[69]:


hateSpeechSet.info()


# In[70]:


hateSpeechSet.describe().T


# # Preprocessing and cleaning of Data

# In[61]:


df = hateSpeechSet[["tweet","class"]]
# class 0 -> Hate speech
# class 1 -> Offensive speech
# class 2 -> Neither
print(df.head(5))


# In[62]:


import warnings
warnings.filterwarnings('ignore')

for i in range(len(df["tweet"])):
    df["tweet"][i] = df["tweet"][i].lower()
    df["tweet"][i] = re.sub('[.?]', '', df["tweet"][i]) 
    df["tweet"][i] = re.sub('https?://\S+|www.\S+', '', df["tweet"][i])
    df["tweet"][i] = re.sub('<.?>+', '', df["tweet"][i])
    df["tweet"][i] = re.sub('[%s]'%re.escape(string.punctuation), '', df["tweet"][i])
    df["tweet"][i] = re.sub('\n', '', df["tweet"][i])
    df["tweet"][i] = re.sub('\w\d\w', '', df["tweet"][i])
    df["tweet"][i] = [word for word in df["tweet"][i].split(' ') if word not in stopWord]
    df["tweet"][i] = " ".join(df["tweet"][i])
    df["tweet"][i] = [stemmer.stem(word) for word in df["tweet"][i].split(' ')]
    df["tweet"][i] = " ".join(df["tweet"][i])  


# # Splitting Data

# In[63]:


tweets = np.array(df["tweet"])
verdict = np.array(df["class"])
countVectors = CountVectorizer() 
vTweets = countVectors.fit_transform(tweets) # Need to vectorize words to be able to preform machine learning (words -> number)

tweetTrain,tweetTest,verdictTrain,verdictTest = train_test_split(vTweets,verdict, test_size = 0.2, random_state = 42)


# # Modeling

# In[64]:


model= DecisionTreeClassifier()
model.fit(tweetTrain,verdictTrain)
print(accuracy_score(verdictTest,model.predict(tweetTest)))


# # Conclusion
# 
# From our accuracy score we can see our model is able to reliabliy detect offensive and hate speech with a ~87% accuracy.
# To imporove accuracy we it would be possible to follow different steps such as, using a new learning model, larger training data, and including more predictors.
# 

# # Demo

# In[65]:


# class 0 -> Hate speech
# class 1 -> Offensive speech
# class 2 -> Neither
testNeither = "This isn't an offensive comment"
vTestNeither = countVectors.transform([testNeither]).toarray()
print(model.predict(vTestNeither))


# In[66]:


testOffensive = "fuck woman shouldnt complain clean hous amp man alway take trash"
vTestOffensive = countVectors.transform([testOffensive]).toarray()
print(model.predict(vTestOffensive))

