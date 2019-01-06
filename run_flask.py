import pandas as pd
from flask import Flask , render_template, request
from collections import Counter
from movie_clusters import *
import random

app = Flask(__name__)

mov_1 = "_"
mov_2 = "_"
return_liked_movies = []
new_list_count = 1

all_data_clustered = pd.read_csv("all_data_clustered.csv")

movie_list = get_data()

#####################  SUPPORTING FUNCTIONS  #####################

def chooseNewTitles():
	global movie_list
	global new_list_count
	global return_liked_movies

	#check if list is empty
	if movie_list:
		#if not empty
		randomRow = random.choice(movie_list)
	else:

		movie_list = process_response_iterate(all_data_clustered, np.array(return_liked_movies))
		return_liked_movies = []
		randomRow = random.choice(movie_list)

		new_list_count = new_list_count + 1

	opt_1_name = randomRow[0]
	opt_1_custer = randomRow[1]

	movie_list.remove(randomRow)

	opt_2 = random.choice(movie_list)
	opt_2_name = opt_2[0]

	movie_list.remove(opt_2)

	return opt_1_name, opt_2_name

def chooseTitleOnDecisions():
	modeClusterNum = Counter(data).most_common(1)[0][0]
	return modeClusterNum


def getClusterNum(title):
	return all_data_clustered.loc[all_data_clustered['original_title'] == title, 'clusters'].iloc[0]

def getTitleName(cluster):
	return all_data_clustered.loc[all_data_clustered['clusters'] == cluster, 'original_title'].iloc[0]

def format_rec(recommendation_list):
	clean_list = []
	for rec in recommendation_list:
		movie_name = "'" + str(rec[0]) + "'"
		clean_list.append(movie_name)
	return ' and '.join(clean_list)

#####################  FLASK METHODS  #####################
@app.route('/')
def homePage():
	return render_template("movie_temp.html", your_rec = "Nothing yet! Get a recommendation!")

'''
The purpose of this method is to grab the newest user selected movies and then propose new movies to recommend to them.
Once they have selected about 2 - 3 rounds of preferences the app will redirect them to the home page and propose movie suggestions
'''
@app.route('/recommendations/', methods=['POST', 'GET'])
def recommendationsPage():
	#global variables
	global mov_1
	global mov_2
	global return_liked_movies
	global new_list_count

	num_rounds = 2
   
	if request.method == 'POST':
		#If the reommendation form has previously been initiated determine which round of recommendations the user is one
		if 'opt_1' in request.form or 'opt_2' in request.form:
			if 'opt_1' in request.form:

				if new_list_count < num_rounds:
					#Append to the selected movie list
					return_liked_movies.append([mov_1,getClusterNum(mov_1)])
					mov_1, mov_2  =  chooseNewTitles() #opt_1

				elif new_list_count == num_rounds:
					new_list_count = new_list_count + 1
					return_liked_movies.append([mov_1,getClusterNum(mov_1)])
					recommendation_fin = get_knn(np.array(return_liked_movies), 2)
					rtrn_rec = format_rec(recommendation_fin)

					#renders the home page - with the proposed recommendations
					return render_template("movie_temp.html", your_rec = rtrn_rec)


				#if the user has already selected 3 movies and wanted to continue to get another recommendation, 
				#we will ask them two go throw 2 more rounds of movie selections
				else:
					new_list_count = 1
					return_liked_movies.clear()
					return_liked_movies.append([mov_1,getClusterNum(mov_1)])
					mov_1, mov_2  =  chooseNewTitles() #opt_1

			elif 'opt_2' in request.form:

				if new_list_count < num_rounds:
					#Append to the selected movie list					
					return_liked_movies.append([mov_2,getClusterNum(mov_2)])
					mov_1, mov_2  =  chooseNewTitles() #opt_2

				elif new_list_count == num_rounds:
					new_list_count = new_list_count + 1
					return_liked_movies.append([mov_2,getClusterNum(mov_2)])
					recommendation_fin = get_knn(np.array(return_liked_movies), 2)
					rtrn_rec = format_rec(recommendation_fin)

					#renders the home page - with the proposed recommendations
					return render_template("movie_temp.html", your_rec = rtrn_rec)
				
				#if the user has already selected 3 movies and wanted to continue to get another recommendation, 
				#we will ask them two go throw 2 more rounds of movie selections
				else:
					new_list_count = 1
					return_liked_movies.clear()
					return_liked_movies.append([mov_2,getClusterNum(mov_2)])
					mov_1, mov_2  =  chooseNewTitles() #opt_2

	#Else propose 2 new movies for the user to choose between
	else:
		if new_list_count < num_rounds:
			mov_1, mov_2 = chooseNewTitles() #else

	#Pull the youtube trailers for the movies the user will be chosing between	
	youtube_url_1 = "https://www.youtube.com/embed?listType=search;list=" + mov_1
	youtube_url_2 = "https://www.youtube.com/embed?listType=search;list=" + mov_2

	#Render the recommendation html page
	return render_template("recommendations.html", mov_name_1 = youtube_url_1, mov_name_2 = youtube_url_2, name_1 = mov_1, name_2 = mov_2)	


if __name__=='__main__':
    app.run(debug = True)

