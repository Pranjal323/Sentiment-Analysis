#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("youtube_dataset.csv")df


# In[3]:


#checking for null values
df.isnull().sum()


# In[4]:


df.dropna(inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


df["Video Name"]


# In[7]:


df.columns


# In[8]:


import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
#preprocess the comments
def preprocess_comment(comment):
    #comment to lowercase
    comment=comment.lower();
    #remove websites
    comment=re.sub(r'http\S+','',comment)
    #remove  whitespace
    comment=comment.strip()
    #remove punctuations
    comment=comment.translate(str.maketrans('','',string.punctuation))
    #remove numbers
    comment=re.sub(r'\d','',comment)
    comment=re.sub(r'\n','',comment)
    stop_words = set(stopwords.words('english'))
    comment = ' '.join([word for word in comment.split() if word not in stop_words])
    return comment

df['Sorted_comments']=df['Comment'].apply(preprocess_comment)
print(df['Comment'])
    


# In[16]:


print(df['Sorted_comments'])


# In[17]:


#Perform Sentiment Analysis
from textblob import TextBlob
def analyze_sentiment(comment):
    analysis=TextBlob(comment)
    sentiment=analysis.sentiment.polarity
    return sentiment

#apply sentiment analysis to the cleaned comment
df['sentiment']=df['Sorted_comments'].apply(analyze_sentiment)


# In[18]:


print(df['sentiment'])


# In[19]:


def categorize_sentiment(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment_Category'] = df['sentiment'].apply(categorize_sentiment)

# Display the first few rows of the dataset with sentiment categories
print(df[['sentiment', 'Sentiment_Category']].head())


# In[20]:


#Analyze and visualize the Results
import matplotlib.pyplot as plt
def plot_sentiments(sentiments):
    plt.hist(sentiments,bins=20,edgecolor='black')
    plt.title('Sentiment Analysis of You tube comments')
    plt.xlabel('Sentiment score')
    plt.ylabel('Number of Comments')
    plt.show()

plot_sentiments(df['sentiment'])


# In[21]:


import seaborn as sns
sns.countplot(x='Sentiment_Category', data=df)
plt.title('Sentiment Analysis of YouTube Comments')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Comments')
plt.show()

