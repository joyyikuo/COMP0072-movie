# COMP0072-movie

COMP0072 Movie Recommendation Project (18-19)

Team Members: Yiran Du, Petros Hadjikkos, Yi Kuo, Durga Ravindra, Dimitris Samouil


The purpose of this app is to recommend movies to the user based on their choices to a short quiz. 
Often times when people use medias like Netflix, they are suggested movies based on a specific genre or other characteristics. 
However, we believe that our users should be recommended movies based on multiple characteristics of movies for more quality movie recommendations. 
Users also don't want to have to scroll 100s of different options. 
That's why we are recommending a few options which we believe will really appeal to the user's preferences.
After about 2 - 3 rounds of a user selecting their movie prefence when asked to choose between two, the application will choose the cluster that was most often selected and then find similar movies (through KNN) to suggest to the user.   

This application incorportates NLTK, KMeans, KNN, Flask, and HTML.

To run this application follow the following steps:
  1. Download all the files in this github repository. Ensure that the html files are in a folder called 'templates'.
  2. Then, run the python file titled 'extra_features_updated.py'. This will generated the features we have engineered for our model.
  3. Run the python file titled 'create_movie_data.py'. This will generate an updated verision of the originial data with clusters.
  4. Now we are ready to run the application. Run the file titled 'run_flask.py'. 
  5. Once the brower loads, begin the recommedation process by clicking the red button titled 'Get Recommendation'.
  
    
Enjoy,

Team Anonymous
