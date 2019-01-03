#extra feature extraction
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import ast
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
#export extra features to csv
extra_features=df_all.filter(['id','main_cast_id','director_code'], axis=1)
extra_features.to_csv("extra_features",sep=',')