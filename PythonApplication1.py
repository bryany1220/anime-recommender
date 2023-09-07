
#step 1: import necessary libraries https://www.kaggle.com/code/yonatanrabinovich/anime-recommendations-project/notebook#Preprocessing-and-Data-Analysis-%F0%9F%92%BB
import os
import numpy as np
import pandas as pd
import warnings
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

#step 2: change display settings
pd.options.display.max_columns
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

#step 3: extract data  #https://www.youtube.com/watch?v=lbG_VNLUPBQ
mainpath = r"C:\Users\bryanyu\Downloads\animeprojectdata"
ratingspath = r"C:\Users\bryanyu\Downloads\animeprojectdata\rating.csv"
animepath = r"C:\Users\bryanyu\Downloads\animeprojectdata\anime.csv"

#step 4: check data
rating_df = pd.read_csv(ratingspath)
print(rating_df.head())
print("\n", "="*39, "\n")
anime_df = pd.read_csv(animepath)
print(anime_df.head())
print("\n")

#step 5: gather data info
print(f"anime set (row, col): {anime_df.shape}\n\nrating set (row, col): {rating_df.shape}\n\n")
print("Anime:\n")
print(anime_df.info())
print("\n","="*39,"\n\nRating:\n")
print(rating_df.info())

#step 6: data cleaning
print ("Anime missing values (%):\n")
print (round ((anime_df.isnull().sum().sort_values(ascending = False)) / (len(anime_df.index)), 4) * 100) #get to see empty/null values on top
print("\n\n", "="*39, "\n\nRatings missing values (%):\n")
print (round ((rating_df.isnull().sum().sort_values(ascending = False)) / (len(rating_df.index)), 4) * 100) #get to see empty/null values on top, surprisingly nothing is null here
print(anime_df['type'].mode())
print(anime_df['genre'].mode()) #most rated in database is a hentai TV show apparently :0
print("\n\n") #note: only type and genre have null values
anime_df = anime_df[~np.isnan(anime_df["rating"])] #clear out anime with 0 rating
topgenre = anime_df['genre'].dropna().mode().values[0]
toptype = anime_df['type'].dropna().mode().values[0]
anime_df['genre'].fillna(topgenre, inplace = True)
anime_df['type'].fillna(toptype, inplace = True)
print(anime_df.isnull().sum()) #sanity check
lamby = lambda a :(np.nan if a == -1 else a)
#turn -1 ratings into NaN in dataset
rating_df['rating'] = rating_df['rating'].apply(lamby)
print(rating_df.head(10))

#step 7: anime recommendations building
charlie = ''
boolie = False

while boolie == False :
    charlie = input("Enter 't' for TV shows or 'm' for movies\n")
    if charlie == "t": #set to TV series by default
        anime_df = anime_df[anime_df['type'] == 'TV']
        boolie = True
    elif charlie == "m":
        anime_df = anime_df[anime_df['type'] == 'Movie']
        boolie = True
    else:
        print("An error occured. Please try again.\n")
        boolie = False

rated_anime = rating_df.merge(anime_df, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', '']) #combine ratings with respective anime
rated_anime =rated_anime[['user_id', 'name', 'rating']]
rated_anime_adj= rated_anime[rated_anime.user_id <= 7500] #only first 7500 users to rate anime will be considered for adjusted table
print(rated_anime_adj.head())


#step 8: create a pivot table to help see ratings of all anime (i guess?)
pivot1 = rated_anime_adj.pivot_table(index = ['user_id'], columns = ['name'], values = 'rating')
print(pivot1.head()) #prints data for first five users' ratings
pivot2 = pivot1.apply(lambda b: (b - np.mean(b)) / (np.max(b) - np.min(b)), axis = 1)
pivot2.fillna(0, inplace = True) #fill N/a with rating of 0
pivot2 = pivot2.T #easier on the eyes somehow?
pivot2 = pivot2.loc[:, (pivot2 != 0).any(axis = 0)] #display column if not zero
pivot3 = sp.sparse.csr_matrix(pivot2.values)

#step 9: create similarity model with cosine model
anime_sim = cosine_similarity(pivot3)
anime_sim_df = pd.DataFrame(anime_sim, index = pivot2.index, columns = pivot2.index)
print(anime_sim_df)

#step 10: build the actual function
def recommend(title, numrecs) :
    i = 1
    print('Recommended because you watched {}:\n'.format(title))
    for anime in anime_sim_df.sort_values(by = title, ascending = False).index[1:numrecs + 1]:
        print(f"#{i}: {anime}, {round(anime_sim_df[anime][title] * 100, 2)}% match")
        i += 1

#step 11: use the function
basis = input("Enter anime title:\n")
num_recs = int(input("How many anime title recommendations would you like? (Min 1, Max 25):\n"))
if num_recs < 1: #minimum value check
    num_recs = 1
elif num_recs > 25: #maximum value check
    num_recs = 25
try:
    recommend(basis, num_recs)
except KeyError:
    print("Sorry, we don't have", basis, "in our database yet\n")


