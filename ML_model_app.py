#!/usr/bin/env python
# coding: utf-8

# Here we will be using Content Based Filtering
# 
# Content-Based Filtering: This method uses only information about the description and attributes of the items users has previously consumed to model user's preferences. In other words, these algorithms try to recommend items that are similar to those that a user liked in the past (or is examining in the present). In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended.
# 
# This data set consists of restaurants of Bangalore,India collected from Zomato.
# 
# Our aim is to create a content based recommender system in which when we will write a restaurant name, Recommender system will look at the reviews of other restaurants, and System will recommend us other restaurants with similar reviews and sort them from the highest rated.

# # Reccomendation System:
# 1. **Loading the dataset:** Load the data and import the libraries. <br>
# 2. **Data Cleaning:** <br>
#  - Deleting redundant columns.
#  - Renaming the columns.
#  - Dropping duplicates.
#  - Cleaning individual columns.
#  - Remove the NaN values from the dataset
#  - #Some Transformations
# 3. **Text Preprocessing**
#  - Cleaning unnecessary words in the reviews
#  - Removing links and other unncessary items
#  - Removing Symbols
# 4. **Recommendation System**

# ### Importing Libraries

# In[45]:


#Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import warnings
import pickle
from flask import Flask, render_template,request
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ### Loading the dataset

# In[2]:


#reading the dataset
zomato_real=pd.read_csv("./zomato.csv")
#zomato_real.head()


# In[3]:


#zomato_real.info()


# ### Data Cleaning and Feature Engineering

# In[4]:


#Deleting Unnnecessary Columns
zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column "dish_liked", "phone", "url" and saving the new dataset as "zomato"


# In[5]:


#Removing the Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)


# In[6]:


#Remove the NaN values from the dataset
zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)
#zomato.info()


# In[7]:


#Reading Column Names
#zomato.columns


# In[8]:


#Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
#zomato.columns


# In[9]:


#Some Transformations
zomato['cost'] = zomato['cost'].astype(str)
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float)
#zomato.info()


# In[10]:


#Reading Rate of dataset
#zomato['rate'].unique()


# In[11]:


#Removing '/5' from Rates
zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
#zomato['rate'].head()


# In[12]:


# Adjust the column names
zomato.name = zomato.name.apply(lambda x:x.title())
zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)
zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)
#zomato.cost.unique()


# In[13]:


#zomato.head()


# In[14]:


#zomato['city'].unique()


# In[15]:


#zomato.head()


# In[1]:


## Checking Null values
zomato.isnull().sum()


# In[17]:


## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()


# In[18]:


#zomato.head()


# In[19]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (1,5))

zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

#zomato.sample(3)


# In[20]:


#zomato.head()


# In[21]:


## Text Preprocessing


# Some of the common text preprocessing / cleaning steps are:
# 
#  - Lower casing
#  - Removal of Punctuations
#  - Removal of Stopwords
#  - Removal of URLs
#  - Spelling correction

# In[22]:


# 5 examples of these columns before text processing:
#zomato[['reviews_list', 'cuisines']].sample(5)


# In[23]:


## Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()
#zomato[['reviews_list', 'cuisines']].sample(5)


# In[24]:


## Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
#zomato[['reviews_list', 'cuisines']].sample(5)


# In[25]:


## Removal of Stopwords
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))


# In[26]:


## Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))


# In[27]:


#zomato[['reviews_list', 'cuisines']].sample(5)


# In[28]:


# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
pickle.dump(restaurant_names, open('names.pkl','wb'))
#restaurant_names


# In[29]:


def get_top_words(column, top_nu_of_words, nu_of_word):
    
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    
    bag_of_words = vec.fit_transform(column)
    
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:top_nu_of_words]


# In[30]:


#zomato.head()


# In[31]:


#zomato.sample(5)


# In[32]:


#zomato.shape


# In[33]:


#zomato.columns


# In[34]:


zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)


# In[35]:


import pandas

# Randomly sample 60% of the dataframe
df_percent = zomato.sample(frac=0.5)


# In[36]:


#df_percent.shape


# ### Term Frequency-Inverse Document Frequency
# Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document. This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each column represents a restaurant, as before.
# 
# TF-IDF is the statistical method of evaluating the significance of a word in a given document.
# 
# TF — Term frequency(tf) refers to how many times a given term appears in a document.
# 
# IDF — Inverse document frequency(idf) measures the weight of the word in the document, i.e if the word is common or rare in the entire document.
# The TF-IDF intuition follows that the terms that appear frequently in a document are less important than terms that rarely appear.
# Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix quite easily.

# In[37]:


df_percent.set_index('name', inplace=True)


# In[38]:


indices = pd.Series(df_percent.index)


# In[39]:


# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])


# In[40]:


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[41]:


def recommend(name, cosine_similarities = cosine_similarities):

    # Create a list to put top 10 restaurants
    recommend_restaurant = []

    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]

    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)

    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)

    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])

    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])

    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))

    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))

    return df_new


# In[46]:




# In[47]:


pickle.dump(recommend, open('model.pkl','wb'))


# app = Flask(__name__)
#
# @app.route('/')
# def home():
# #    model = pickle.load(open('model.pkl','rb'))
#     return 'Cinnamon'
# #    names = pickle.load(open('names.pkl','rb'))
# #    return render_template('index.html', rests=names)
#
# if __name__== "__main__":
#     app.run(debug=True)
# In[42]:


# HERE IS A RANDOM RESTAURANT. LET'S SEE THE DETAILS ABOUT THIS RESTAURANT:
#df_percent[df_percent.index == 'Pai Vihar'].head()


# In[52]:




# #### References
#  - [Recommender Systems in Python 101](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101)
#  - [How to build a Restaurant Recommendation Engine](https://medium.com/analytics-vidhya/how-to-build-a-restaurant-recommendation-engine-part-1-21aadb5dac6e)
#  - [Getting started with Text Preprocessing](kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing)

# ## End of the Notebook
