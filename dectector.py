#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import nltk
import re
from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split
from sklearn. tree import DecisionTreeClassifier

stopWord = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.SnowballStemmer('english')


# # Intial preview of dataset.

# In[8]:


hateSpeechSet = pd.read_csv("labeled_data.csv")
print(hateSpeechSet.head(5))


# In[9]:


hateSpeechSet.info()


# In[10]:


hateSpeechSet.describe().T


# # Preprocessing and cleaning of Data

# In[ ]:


df = hateSpeechSet[["tweet","class"]]
# class 0 -> Hate speech
# class 1 -> Offensive speech
# class 2 -> Neither
print(df.head(5))


# In[ ]:


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


# In[ ]:




