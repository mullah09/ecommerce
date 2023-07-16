# Data processing
import pandas as pd
from itertools import groupby

# Nearest Neighbor
from sklearn.neighbors import NearestNeighbors

# Data
import csv
from .models import Product, Color

# miscelaneous
from munch import DefaultMunch
import re


def key_func(k):
    return k['item']

def getMatrix():
    df=""
    userDict={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    cartItem={}
    productList=[]
    with open('shoesSales2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                cartItem[row[0]]=[row[4],row[5],row[6],row[7],row[8],row[9]]
                productList.append(transform_to_dummy_obj(row, line_count))
                line_count += 1

    df = pd.DataFrame(cartItem).T.rename(columns=userDict)
    return df, productList

#get recommendation from updated data
def get_recommendation(df, df1, userId, num_recommendation, productList):

    recommended_movies = []
    for m in df[df[userId] == 0].index.tolist():

        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(userId)]
        recommended_movies.append((m, predicted_rating))

    sorted_rm = sorted(recommended_movies, key=lambda x:x[1], reverse=True)
    sorted_rm = sorted_rm[:num_recommendation]
    sorted_rm = [i[0] for i in sorted_rm]
    finalList = [e for e in productList if e.title in set(sorted_rm)][:num_recommendation]
    
    return(finalList)

#update reccomendation data using KNN
def recommender(df, userId):

    number_neighbors = 3
    df1 = df.copy()

    # print(df)

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
   
    user_index = df.columns.tolist().index(userId)

    # t: movie_title, m: the row number of t in df
    for m,t in list(enumerate(df.index)):
    
        # find movies without ratings by userId
        if df.iloc[m, user_index] == 0:
            sim_movies = indices[m].tolist()
            movie_distances = distances[m].tolist()
            # remove the product itself from neighbor list
            if m in sim_movies:
                id_movie = sim_movies.index(m)
                sim_movies.remove(m)
                movie_distances.pop(id_movie) 
            # if Some product have all 0 value which are considered the same movies by NearestNeighbors(). 
            # Then,even the product itself cannot be included in the indices. 
            # so, take off the farthest movie in the list.
            else:
                sim_movies = sim_movies[:number_neighbors-1]
                movie_distances = movie_distances[:number_neighbors-1]
  
            movie_similarity = [1-x for x in movie_distances]
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0
            # print(sim_movies)

            # for each similar movie
            for s in range(0, len(movie_similarity)):
                # check if the rating of a similar movie is zero
                if df.iloc[sim_movies[s], user_index] == 0:
                    # if the rating is zero, ignore the rating 
                    # and the similarity in calculating the predicted rating
                    # print(len(movie_similarity_copy) == (number_neighbors - 1))
                    if len(movie_similarity_copy) == (number_neighbors - 1):
                        movie_similarity_copy.pop(s)
                    else:
                        movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))
                # if the rating is not zero, use the rating and similarity in the calculation
                else:
                    nominator = nominator + movie_similarity[s]*df.iloc[sim_movies[s],user_index]

            if len(movie_similarity_copy) > 0:
                predicted_r = nominator/sum(movie_similarity_copy) if sum(movie_similarity_copy) > 0 else 0
            else:
                predicted_r = 0

            df1.iloc[m,user_index] = predicted_r
    return df1



def coll_filter_anon():
    userId = "anon"
    num_recommended_movies = 4

    df, productList = getMatrix()

    df = df.assign(anon=[0] * df.shape[0])
    df1 = recommender(df, userId)
    # print(get_recommendation(df, df1, userId, num_recommended_movies, productList))
    
    # print(test)
    # for product in productList:
    #     print(product.productattribute_set.first.image)
    
    return(get_recommendation(df, df1, userId, num_recommended_movies, productList))

           
def col_filter(userId):
    num_recommended_movies = 4

    df, productList = getMatrix()
    df1 = recommender(df, userId)
    
    return(get_recommendation(df, df1, userId, num_recommended_movies, productList))

print(col_filter)

def transform_to_dummy_obj(row, id):
    return DefaultMunch.fromDict({'title':row[0],
                'id':id,
                'slug': "dummy",
                'productattribute_set':{
                    'first':{
                        'image': re.findall(r'"([^"]*)"', row[3])[0],
                        'price':row[1],
                        'color':{'title': 'black'}
                    }
                }
            })