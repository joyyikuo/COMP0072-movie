import pandas as pd
import numpy as np
import re
from sklearn.cluster import MiniBatchKMeans
import random

# Filter out adult films from all_data
all_data = pd.read_csv('movies_metadata.csv')
all_data = all_data[all_data['adult'] == 'False']
all_data.reset_index(drop=True, inplace=True)


# function defining
# function for creating list of unique items
def unique_items(in_list):
    unique_list = []
    for x in in_list:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


def get_n_clusters(c_labels):
    k_range = range(1, 10)
    cluster_errors = []

    for k in k_range:
        clust = MiniBatchKMeans(n_clusters=k, random_state=3234)
        clust.fit(c_labels)
        cluster_errors.append(clust.inertia_)

    # plt.figure(figsize=(12, 6))

    # plt.plot(k_range, cluster_errors, marker="o", markersize=6)
    # plt.xlabel("Number of clusters")
    # plt.ylabel("Inertia")
    # plt.title("Elbow Method", fontsize=14)
    # plt.grid()

    # plt.show()

    min_error = np.where((cluster_errors == min(cluster_errors)))[0][0] + 1

    return min_error


def get_cluster_labels(num_clusters, labels):
    km = MiniBatchKMeans(n_clusters=num_clusters, random_state=234523)

    return km.fit(labels)


def get_movies(in_data, num_clusters):
    movie_titles = []
    for i in range(num_clusters):
        curr_list = in_data[in_data['clusters'] == i]
        movie_titles.append([curr_list.loc[random.sample(list(curr_list.index), 1), 'original_title'].item(), i])
    if num_clusters % 2 != 0:
        curr_list = in_data[in_data['clusters'] == (num_clusters - 1)]
        movie_titles.append([curr_list.loc[random.sample(list(curr_list.index), 1), 'original_title'].item(), i])

    return movie_titles


# General Setup
# Genre Labeling
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

<<<<<<< HEAD
# Join dummie variables for original language of film
lang_1h = pd.get_dummies(all_data_mod['original_language'])
lang_1h.rename(columns={'id': 'id_lang'}, inplace=True)
all_data_mod = all_data_mod.join(lang_1h)

# Convert datetime to type int
all_data_mod['release_date_int'] = pd.to_datetime(all_data_mod['release_date'])
all_data_mod['release_date_int'] = all_data_mod['release_date_int'].astype(int)

# Use all_data_mod for clusterables

feature = ['Drama', 'Documentary', 'Family', 'Comedy', 'Crime',
           'Horror', 'Science Fiction', 'Action', 'Adventure',
           'War', 'Thriller', 'Romance', 'Fantasy', 'Mystery',
           'Western', 'Foreign', 'History', 'Music', 'Animation',
           'TV Movie', 'ab', 'af', 'am', 'ar', 'ay', 'bg', 'bm',
           'bn', 'bo', 'bs', 'ca', 'cn', 'cs', 'cy', 'da', 'de',
           'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy',
           'gl', 'he', 'hi', 'hr', 'hu', 'hy', 'id_lang', 'is', 'it',
           'iu', 'ja', 'jv', 'ka', 'kk', 'kn', 'ko', 'ku', 'ky', 'la',
           'lb', 'lo', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
           'nb', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'qu', 'ro',
           'ru', 'rw', 'sh', 'si', 'sk', 'sl', 'sm', 'sq', 'sr', 'sv',
           'ta', 'te', 'tg', 'th', 'tl', 'tr', 'uk', 'ur', 'uz', 'vi',
           'wo', 'xx', 'zh', 'zu', 'release_date_int', "vote_average",
           "revenue"]

cluster_labels = all_data_mod.loc[:, feature]
cluster_labels = cluster_labels.dropna()

n_clusters = get_n_clusters(cluster_labels)
clusters = get_cluster_labels(n_clusters, cluster_labels).labels_

clusters_series = pd.Series(clusters, index=cluster_labels.index)
all_data_clustered = pd.concat([all_data_mod.loc[cluster_labels.index, :], clusters_series], axis=1)
all_data_clustered.rename(columns={0: 'clusters'}, inplace=True)

# Initial set of movie output
movie_list_1 = get_movies(all_data_clustered, n_clusters)


# Update clusters after exhausting current list
def process_response(return_cluster, curr_data):
    start = True
    for i in return_cluster:
        if start:
            update_data = curr_data[curr_data['clusters'] == i]
            start = False
        else:
            update_data = update_data.append(curr_data[curr_data['clusters'] == i])

    update_labels = update_data.loc[:, feature]
    up_n_clusters = get_n_clusters(update_labels)
    update_km = get_cluster_labels(up_n_clusters, update_labels)
    update_clu = update_km.labels_

    update_clusters_series = pd.Series(update_clu, index=update_labels.index)
    update_data_clustered = pd.concat([all_data_mod.loc[update_labels.index, :], update_clusters_series], axis=1)
    update_data_clustered.rename(columns={0: 'clusters'}, inplace=True)

    movie_list_update = get_movies(update_data_clustered, up_n_clusters)

    return update_data_clustered, movie_list_update


sample_return_cluster = [0, 3, 4]
current_set, movie_list_new = process_response(sample_return_cluster, all_data_clustered)
=======
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
>>>>>>> b007eed5dcb74d64fb96da31ea78e4a5cc87d6e9
