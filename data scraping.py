#extra feature extraction
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import ast
from sklearn.decomposition import PCA
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
from sklearn.decomposition import PCA

# Filter out adult films from all_data
all_data = pd.read_csv('movies_metadata.csv')
all_data = all_data[all_data['adult'] == 'False']
all_data.reset_index(drop=True, inplace=True)
all_data['id']=all_data['id'].astype('int64')

#extract credits and cast info from another dataset
#csv is too big to be uploaded to github, here's the address: https://www.kaggle.com/rounakbanik/the-movies-dataset#credits.csv
df_credits=pd.read_csv('credits.csv')
#transform string to dictionary
df_credits['cast'] = df_credits['cast'].apply(ast.literal_eval)
#df_credits['crew'] = df_credits['crew'].apply(ast.literal_eval)

#Find the right crew to extract
def job(x):
    for i in x:
        return i['job']
#print(df_credits['crew'].apply(job).value_counts())
# define extraction function for main cast and director
def main_cast_id(x):
    for i in x:
        return i['id']
#def director_id(x):
#    for i in x:
#        if i['job']=='Director':
#            return i['id']
#        else: 
#            return np.nan
#Extract main cast and the director
df_credits['main_cast_id']=df_credits['cast'].apply(main_cast_id)
#df_credits['director_id']=df_credits['crew'].apply(director_id)
#after extraction, I find out the directors available in this dataset is only around 5000, therefore, I decided to scrap the information from imdb
#merge the two dataframes
all_data=pd.merge(all_data,df_credits,on='id',how="inner")


#setup scraper
all_data['imdb_id']=all_data['imdb_id'].astype('str')
url=all_data['imdb_id'].apply(lambda x: 'https://www.imdb.com/title/'+x)
rows=len(url)+1
#create empty dataframe
all_data['director']=np.nan
all_data['content_rating']=np.nan
#scrap director
for i in range(rows):
    page = requests.get(url[i])
    bs = BeautifulSoup(page.text)
    
    try:
        js = json.loads(bs.select_one("script[type=application/ld+json]").text)
        all_data['director'][i]=js['director']['name']
    except TypeError:
        all_data['director'][i]=js['director'][0]['name']
    except (KeyError,AttributeError):
        pass
    #print(i)
#only 500 null value, so keep the feature and transform data type
all_data['director_code']=all_data['director'].astype('category').cat.codes

    #scrap content rating (PG-R)
for i in range(rows):
    page = requests.get(url[i])
    bs = BeautifulSoup(page.text)
    try:
        js = json.loads(bs.select_one("script[type=application/ld+json]").text)
        all_data['content_rating'][i]=js['contentRating']
    except (KeyError,AttributeError):
        pass
    #print(i)
#12000 null value in content rating, so not included 


#get tfidf matrix from overview
#Simple encoding 
all_data['overview']=all_data['overview'].astype('str')

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
        lemm_words = [wnl.lemmatize(word[0], get_wordnet_pos(word[1])) for word in pos]
        return lemm_words
    final_token=lemmatize_with_pos(filtered_tokens)
    #def stemming(token):
        #sno = nltk.stem.SnowballStemmer('english')
        #stem_words = [sno.stem(word) for word in lemma_token]
        #return stem_words
    return final_token
#TFIDF Vectorization
#Define the vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df = 0.99,min_df = 0.01,tokenizer = Text_Processing)
#Fitting
tfidf_matrix_overview = tfidf_vectorizer.fit_transform(all_data["overview"])
#print("TFIDF vector's dimension is",tfidf_matrix_overview.shape[1])
# Feature name for vector
words = tfidf_vectorizer.get_feature_names()
#print(words)
#dimensionality reduction
pca = PCA(n_components = 20, random_state = 2)
data20D = pca.fit_transform(tfidf_matrix_overview.toarray())
#print(data20D.shape)
#create overview dataframe and merge with the main dataset
tfidf = pd.DataFrame(data=data20D)
tfidf['id']=all_data['id']
tfidf.columns = ['overview_0', 'overview_1','overview_2','overview_3','overview_4','overview_5','overview_6','overview_7','overview_8','overview_9','overview_10','overview_11','overview_12','overview_13','overview_14','overview_15','overview_16','overview_17','overview_18','overview_19','id']
all_data=pd.merge(all_data,tfidf,on='id')

#export extra features to csv
extra_features=all_data.filter(['id','main_cast_id','director_code','overview_0', 'overview_1','overview_2','overview_3','overview_4','overview_5','overview_6','overview_7','overview_8','overview_9','overview_10','overview_11','overview_12','overview_13','overview_14','overview_15','overview_16','overview_17','overview_18','overview_19'], axis=1)
extra_features.to_csv("extra_features_updated",sep=',')
