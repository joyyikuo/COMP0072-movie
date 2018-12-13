import pandas as pd
import numpy as np
import re
# from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

import random

import math
from sklearn.neighbors import NearestNeighbors

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


def get_movies(in_data, cluster_n):
    movie_titles = []
    for i in cluster_n:
        curr_list = in_data[in_data['clusters'] == i]
        movie_titles.append([curr_list.loc[random.sample(list(curr_list.index), 1), 'original_title'].item(), i])
    # select random cluster from list of selected clusters for an additional movie if returned clusters is odd
    if len(cluster_n) % 2 != 0:
        curr_list = in_data[in_data['clusters'] == random.choice(cluster_n)]
        movie_titles.append([curr_list.loc[random.sample(list(curr_list.index), 1), 'original_title'].item(), i])

    return movie_titles


def get_data():
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

    # Use all_data_mod for clusterables

    cluster_labels = all_data_mod.loc[:, feature]
    cluster_labels = cluster_labels.dropna()

    n_clusters = get_n_clusters(cluster_labels)

    # temp n_clusters, should use n_clusters after refining k-optimisation
    n_cluster_temp = 4
    clusters = get_cluster_labels(n_cluster_temp, cluster_labels).labels_

    clusters_series = pd.Series(clusters, index=cluster_labels.index)
    all_data_clustered = pd.concat([all_data_mod.loc[cluster_labels.index, :], clusters_series], axis=1)
    all_data_clustered.rename(columns={0: 'clusters'}, inplace=True)

    # Initial set of movie output
    input_cluster = range(n_clusters + 1)
    movie_list_1 = get_movies(all_data_clustered, input_cluster)

    return movie_list_1, all_data_clustered, all_data_mod


# Update clusters after exhausting current list
# Will not be using this
def process_response_recluster(return_cluster, curr_data, all_data_mod):
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


# Iterate to get more movies from the selected clusters
def process_response_iterate(curr_data, return_movie):
    return_cluster = return_movie[:, 1].astype(int)
    movie_list_update = get_movies(curr_data, return_cluster)

    return movie_list_update


# Get knn of chosen movies
def get_knn(chosen_movie, k, all_data_clustered):
    recommendation = []

    for i in range(len(chosen_movie)):
        movie_details = [chosen_movie[i, 0], chosen_movie[i, 1].astype(int)]
        filtered_df = all_data_clustered.loc[all_data_clustered['clusters'] == movie_details[1], feature]
        cluster_df = all_data_clustered.loc[all_data_clustered['clusters'] == movie_details[1], :]
        cluster_df = cluster_df.reset_index(drop=True)

        k = k + 1  # use k+1 as original title will be included in k as well
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(filtered_df)
        distances, indices = nbrs.kneighbors(filtered_df)

        get_item = cluster_df.loc[cluster_df['original_title'] == movie_details[0]].index

        get_indices = indices[get_item][0]

        for j in range(1, len(cluster_df.loc[get_indices, 'original_title'])):
            recommendation.append([cluster_df.loc[get_indices[j], 'original_title'], movie_details[1]])

    return recommendation

# sample_return_movies = np.array([['Heartbeeps', 0], ['Planet of the Apes', 2]])
# sample_return_movies_2 = np.array([['Jungle Cat', 0], ['Arrival', 2]])

# movie_list_1, all_data_clustered, all_data_mod = get_data()
# movie_list_new = process_response_iterate(all_data_clustered, sample_return_movies)
# recommendation_fin = get_knn(sample_return_movies_2, 2, all_data_clustered)
