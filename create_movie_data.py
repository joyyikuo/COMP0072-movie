import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
import random
import math
from sklearn.neighbors import NearestNeighbors


# Filter out adult films from all_data
all_data = pd.read_csv('movies_metadata.csv')
all_data = all_data[all_data['adult'] == 'False']
all_data.reset_index(drop=True, inplace=True)
extra_features=pd.read_csv('extra_features_updated.csv')
all_data['id']=all_data['id'].astype('int64')
all_data=pd.merge(all_data,extra_features,on='id')


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
           "revenue",'main_cast_id','director_code','overview_0', 
           'overview_1','overview_2','overview_3','overview_4',
           'overview_5','overview_6','overview_7','overview_8',
           'overview_9','overview_10','overview_11','overview_12',
           'overview_13','overview_14','overview_15','overview_16',
           'overview_17','overview_18','overview_19']

# function defining
# function for creating list of unique items
def unique_items(in_list):
    unique_list = []
    for x in in_list:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

def get_n_clusters(c_labels):
    k_range = range(1, 4)
    cluster_errors = []

    for k in k_range:
        clust = KMeans(n_clusters=k, random_state=3234)
        clust.fit(c_labels)
        cluster_errors.append(clust.inertia_)

    min_error = np.where((cluster_errors == min(cluster_errors)))[0][0] + 1

    return min_error

def get_cluster_labels(num_clusters, labels):
    km = KMeans(n_clusters=num_clusters, random_state=234523)

    return km.fit(labels)

def create_data():
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

    # Join dummie variables for original language of film
    lang_1h = pd.get_dummies(all_data_mod['original_language'])
    lang_1h.rename(columns={'id': 'id_lang'}, inplace=True)
    all_data_mod = all_data_mod.join(lang_1h)

    # Convert datetime to type int
    # all_data_mod['release_date_int'] = pd.to_datetime(all_data_mod['release_date'])
    all_data_mod['release_date_int'] = all_data_mod['release_date'].str.split('-').str[0]
    all_data_mod['release_date_int'] = all_data_mod['release_date_int'].fillna(0)
    all_data_mod['release_date_int'] = all_data_mod['release_date_int'].astype('int64')
    all_data_mod['release_date_int'] = all_data_mod['release_date_int'].astype(float)

    all_data_mod.to_csv("updated_movies_metadata.csv")

    return all_data_mod

def get_clusters (all_data_mod):
    # Use all_data_mod for clusterables

    cluster_labels = all_data_mod.loc[:, feature]
    cluster_labels = cluster_labels.dropna()

    n_clusters = get_n_clusters(cluster_labels)
    print(n_clusters)

    # temp n_clusters, should use n_clusters after refining k-optimisation
    n_cluster_temp = 4
    clusters = get_cluster_labels(n_cluster_temp, cluster_labels).labels_

    clusters_series = pd.Series(clusters, index=cluster_labels.index)
    all_data_clustered = pd.concat([all_data_mod.loc[cluster_labels.index, :], clusters_series], axis=1)
    all_data_clustered.rename(columns={0: 'clusters'}, inplace=True)

    all_data_clustered.to_csv("all_data_clustered.csv")

    return "output all_data_clustered"

all_data_mod = create_data()
get_clusters (all_data_mod)
