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
