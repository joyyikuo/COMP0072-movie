import numpy as np
import pandas as pd
import json
import ast
import nltk
#nltk.download()
import string
from nltk.corpus import stopwords 
from nltk.corpus import wordnet 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#initial data processing
df=pd.read_csv('movies_metadata.csv')
df = df[df['adult'] == 'False']
df.set_index(['id', 'title'], inplace=True)
df.info()
df=df.fillna(0)
#Simple encoding 
df['original_language']=df['original_language'].astype('category').cat.codes.value_counts()
df['budget']=df['budget'].astype('int64')
df['revenue']=df['revenue'].astype('int64')
df['popularity']=df['popularity'].astype('float')
df['vote_average']=df['vote_average'].astype('float')
df['vote_count']=df['vote_count'].astype('int64')
df['overview']=df['overview'].astype('str')
df['tagline']=df['tagline'].astype('str')



#text normalisation, tokenisation, and filtering
def Text_Processing(text):
    #lower case
    text=text.lower()
    #remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.translate(str.maketrans('','','1234567890'))
    text = re.sub("[^a-zA-Z]+", " ", text)
    #tokenisation
    tokens = nltk.word_tokenize(text)
    #stop words
    stop_list = set(stopwords.words('english')) 
    filtered_tokens=[word for word in tokens if word not in stop_list]
    #lemmatisation
    wnl = WordNetLemmatizer()
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN    
    def lemmatize_with_pos(token):
        pos = nltk.pos_tag(token)
        lemm_words = [wnl.lemmatize(sw[0], get_wordnet_pos(sw[1])) for sw in pos]
        return lemm_words
    final_token=lemmatize_with_pos(filtered_tokens)
    return final_token

#TFIDF Vectorization

#Define the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df = 0.99,min_df = 0.01,tokenizer = Text_Processing)
#Fitting
tfidf_matrix_overview = tfidf_vectorizer.fit_transform(df["overview"])
print("TFIDF vector's dimension is",tfidf_matrix_overview.shape[1])
# Feature name for vector
words = tfidf_vectorizer.get_feature_names()
print(words)
