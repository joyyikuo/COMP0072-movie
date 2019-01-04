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
           "revenue",'main_cast_id','director_code','overview_0', 
           'overview_1','overview_2','overview_3','overview_4',
           'overview_5','overview_6','overview_7','overview_8',
           'overview_9','overview_10','overview_11','overview_12',
           'overview_13','overview_14','overview_15','overview_16',
           'overview_17','overview_18','overview_19']

# Filter out adult films from all_data
all_data_mod = pd.read_csv("updated_movies_metadata.csv")
all_data_clustered = pd.read_csv("all_data_clustered.csv")


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
    # Initial set of movie output
    n_clusters = 3
    input_cluster = range(n_clusters + 1)
    movie_list_1 = get_movies(all_data_clustered, input_cluster)

    return movie_list_1


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
def get_knn(chosen_movie, k):
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

sample_return_movies = np.array([['Heartbeeps', 0], ['Toy Story', 2]])
sample_return_movies_2 = np.array([['Dracula: Dead and Loving It', 0], ['Pocahontas', 2]])

movie_list_1 = get_data()
print(movie_list_1)
movie_list_new = process_response_iterate(all_data_clustered, sample_return_movies)
recommendation_fin = get_knn(sample_return_movies_2, 2)
print(recommendation_fin)
