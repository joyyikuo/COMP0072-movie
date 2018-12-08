import pandas as pd
from flask import Flask , render_template, request
from collections import Counter

app = Flask(__name__)

mov_1 = ""
mov_2 = ""
likedOptionsName = []
likedOptionsCluster = []

df = pd.read_csv("CSV/clustered_Data.csv")


#####################  SUPPORTING FUNCTIONS  #####################

def chooseNewTitles():
	randomRow = df.sample(n=1)
	opt_1_name = randomRow['original_title'].item()
	opt_1_custer = randomRow['Cluster'].item()

	opt_2_name = df[df.Cluster != opt_1_custer].sample(n=1)['original_title'].item()

	return opt_1_name, opt_2_name

#DETERMINE LOGIC...for right now I'm going to select the mode cluster number
def chooseTitleOnDecisions():
	modeClusterNum = Counter(data).most_common(1)[0][0]
	


def getClusterNum(title):
	df.loc[df['original_title'] == title, 'Cluster'].iloc[0]

def getTitleName(cluster):
	df.loc[df['Cluster'] == cluster, 'original_title'].iloc[0]


#####################  FLASK METHODS  #####################
@app.route('/')
def homePage():
	return render_template("movie_temp.html")


@app.route('/recommendations/')
def recommendationsPage():

	"""
	Go to the Recommendation page
	Show two random movie trailers - user will click on the option that they prefer
	Send the selected option to python so that it can append to the likedOptions list
	Get the cluster # of the last likedOption and get another movie from the same cluster and compare it to a movie from a different cluster
    """

	if 'submit_button' in request.form:
		if request.form['submit_button'] == 'Option 1':
			likedOptionsName.append(mov_1)
			likedOptionsCluster.append(getClusterNum(mov_1))

		elif request.form['submit_button'] == 'Option 2':
			likedOptionsName.append(mov_2)
			likedOptionsCluster.append(getClusterNum(mov_1))

		#define new mov_1 and mov_2



	#this is the first time the user is selecting an option
	else:
		if mov_1 == "" or mov_2 == "":
			#if either movie one or movie 2 do not have values then we will randomly assign movie titles to these variables
			mov_1, mov_2 = chooseNewTitles()
	
	youtube_url_1 = "https://www.youtube.com/embed?listType=search;list=" + mov_1
	youtube_url_2 = "https://www.youtube.com/embed?listType=search;list=" + mov_2

	return render_template("recommendations.html", mov_name_1 = youtube_url_1, mov_name_2 = youtube_url_2)	


if __name__=='__main__':
    app.run(debug = True)

df['treated'] = 0
new_df = df[df['agent_id']==i]
if new_df['got_software'].astype(str).str.contains('1').any():
	df.loc[df['agent_id'] == i, 'treated'] = 1
