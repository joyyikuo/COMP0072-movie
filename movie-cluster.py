import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re

# Filter out adult films from all_data
all_data = pd.read_csv('movies_metadata.csv')
all_data = all_data[all_data['adult'] == 'False']
all_data.reset_index(drop=True, inplace=True)

# Select features of interest
feature = ['genres','id','original_language','popularity','release_date','revenue','vote_average']
all_data.loc[:,feature]



############ GENRE LABELING ############
data_genre = all_data['genres']

# split multiple genre label to individuals
split_genre = []
for i in data_genre.index:
    if data_genre[i] != '[]':
        split_genre.append(re.split('{', data_genre[i]))

# remove unnecessary items from list
for i in range(len(split_genre)):
    split_genre[i].remove('[')

# extract unique values for encoding
num_items = [len(items) for items in split_genre]
items_tot = sum(num_items)

id_key = []
genre_label = []
curr_index = 0

for i in range(len(split_genre)):
    curr_genre = []
    for j in range(len(split_genre[i])):
        _id = re.search('\'id\': (.+?),', split_genre[i][j])
        name = re.search('\'name\': \'(.+?)\'}', split_genre[i][j])

        id_key.append([_id.group(1), name.group(1)])
        curr_genre.append(name.group(1))
        curr_index += 1
    genre_label.append(curr_genre)

# function for creating list of unique items
def unique_items(in_list):
    unique_list = []
    for x in in_list:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

# create unique list and corresponding dataframe
unique_id = unique_items(id_key)
id_dict = pd.DataFrame(unique_id, columns=['id', 'name'])

genre_label_1h = np.zeros([len(genre_label), len(unique_id)])

# create array with movies as rows and genre types as col, encoding cols with genre listed as 1
for i in range(len(genre_label)):
    for j in range(len(unique_id)):
        if any(unique_id[j][1] in x for x in genre_label[i]):
            genre_label_1h[i][j] = 1

# save joined df as all_data_mod
genre_label_h = pd.DataFrame(genre_label_1h, columns=id_dict['name'])
all_data_mod = all_data.join(genre_label_h)

### USE all_data_mod FOR GENRE ENCODED DF ###
############ CAST AND CREW EXTRACTION ############
import json
import ast

df_credits=pd.read_csv('credits.csv')

#transform string to dictionary
df_credits['cast'] = df_credits['cast'].apply(ast.literal_eval)
df_credits['crew'] = df_credits['crew'].apply(ast.literal_eval)

#Find the right crew to extract
def job(x):
    for i in x:
        return i['job']
print(df_credits['crew'].apply(job).value_counts())
# define extraction function for main cast and director
def main_cast_id(x):
    for i in x:
        return i['id']
def director_id(x):
    for i in x:
        if i['job']=='Director':
            return i['id']
        else: 
            return np.nan
#Extract main cast and the director
df_credits['main_cast_id']=df_credits['cast'].apply(main_cast_id)
df_credits['director_id']=df_credits['crew'].apply(director_id)

#Simple encoding for features
all_data_mod=all_data_mod.fillna(0)
all_data_mod['original_language']=all_data_mod['original_language'].astype('category').cat.codes.value_counts()
all_data_mod['budget']=all_data_mod['budget'].astype('int64')
all_data_mod['revenue']=all_data_mod['revenue'].astype('int64')
all_data_mod['popularity']=all_data_mod['popularity'].astype('float')
all_data_mod['vote_average']=all_data_mod['vote_average'].astype('float')
all_data_mod['vote_count']=all_data_mod['vote_count'].astype('int64')
all_data_mod['overview']=all_data_mod['overview'].astype('str')
all_data_mod['tagline']=all_data_mod['tagline'].astype('str')
all_data_mod['id']=all_data_mod['id'].astype('int64')
#Convert release date to int
all_data_mod['release_date']=all_data_mod['release_date'].str.replace('-',"")
all_data_mod['release_date']=all_data_mod['release_date'].fillna(0)
all_data_mod['release_date']=all_data_mod['release_date'].astype('int64')



#merge two dataframes
df_all=pd.merge(all_data_mod,df_credits,on='id',how="outer")
