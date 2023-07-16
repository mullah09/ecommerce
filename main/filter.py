#logging module
import logging
# Data processing
import pandas as pd
from itertools import groupby
# Nearest Neighbor
from sklearn.neighbors import NearestNeighbors

#result
from sklearn.metrics import mean_absolute_error, f1_score, mean_squared_error
from math import sqrt

# Data
from .models import CartOrderItems, Product, User


def key_func(k):
    return k['item']

def getMatrix():
    cartItemData = CartOrderItems.objects.all()
    productData = Product.objects.all()
    user = User.objects.all()

    productList = [item.title for item in productData]
    userDict = {x:item.id for x,item in enumerate(user)}
    userDictLen = len(userDict)

    cartItemList = [{'item':i.item, 'user':i.order.user.id} for i in cartItemData]
    cartItemList = sorted(cartItemList, key=key_func)
    cartItemList = {key:list(value) for key,value in groupby(cartItemList, key_func)}

    cartItem = {}

    for j in productList:
        try:
            temp = [item['user'] for item in cartItemList[j]]
            cartItem[j] = [temp.count(item) for item in userDict.values()]
        except(KeyError):
            cartItem[j] = [0] * userDictLen

    df = pd.DataFrame(cartItem).T.rename(columns=userDict)
    return df

#get recommendation from updated data
def get_recommendation(df, df1, userId, num_recommendation):

    recommended_itemprs = []
    for m in df[df[userId] == 0].index.tolist():

        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(userId)]
        recommended_itemprs.append((m, predicted_rating))

    sorted_rm = sorted(recommended_itemprs, key=lambda x:x[1], reverse=True)
    sorted_rm = sorted_rm[:num_recommendation]
    
    return([i[0] for i in sorted_rm])

#update reccomendation data using KNN
def recommender(df, userId):

    number_neighbors = 3
    df1 = df.copy()

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
   
    user_index = df.columns.tolist().index(userId)

    # t: itempr_title, m: the row number of t in df
    for m,t in list(enumerate(df.index)):
    
        # find itemprs without ratings by userId
        if df.iloc[m, user_index] == 0:
            sim_itemprs = indices[m].tolist()
            itempr_distances = distances[m].tolist()
            # remove the product itself from neighbor list
            if m in sim_itemprs:
                id_itempr = sim_itemprs.index(m)
                sim_itemprs.remove(m)
                itempr_distances.pop(id_itempr) 
            # if Some product have all 0 value which are considered the same itemprs by NearestNeighbors(). 
            # Then,even the product itself cannot be included in the indices. 
            # so, take off the farthest itempr in the list.
            else:
                sim_itemprs = sim_itemprs[:number_neighbors-1]
                itempr_distances = itempr_distances[:number_neighbors-1]
  
            itempr_similarity = [1-x for x in itempr_distances]
            itempr_similarity_copy = itempr_similarity.copy()
            nominator = 0

            # for each similar itempr
            for s in range(0, len(itempr_similarity)):
                # check if the rating of a similar itempr is zero
                if df.iloc[sim_itemprs[s], user_index] == 0:
                    # if the rating is zero, ignore the rating 
                    # and the similarity in calculating the predicted rating
                    if len(itempr_similarity_copy) == (number_neighbors - 1):
                        itempr_similarity_copy.pop(s)
                    else:
                        itempr_similarity_copy.pop(s-(len(itempr_similarity)-len(itempr_similarity_copy)))
                # if the rating is not zero, use the rating and similarity in the calculation
                else:
                    nominator = nominator + itempr_similarity[s]*df.iloc[sim_itemprs[s],user_index]

            if len(itempr_similarity_copy) > 0:
                predicted_r = nominator/sum(itempr_similarity_copy) if sum(itempr_similarity_copy) > 0 else 0
            else:
                predicted_r = 0

            df1.iloc[m,user_index] = predicted_r
    return df, df1

def calculate_metrics(df_original, df_predicted, userId):
    y_true = df_original[userId].values
    y_pred = df_predicted[userId].values
    
    # Debug: Print the true and predicted values
    print("True Values:", y_true)
    print("Predicted Values:", y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    y_true_binary = (y_true > 0.5).astype(int)
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true_binary, y_pred_binary)
    rmse = sqrt(mean_squared_error(y_true, y_pred))

    return mae, f1, rmse
          
def col_filter(userId):
    num_recommended_itemprs = 4

    df = getMatrix()
    df_original, df_predicted = recommender(df, userId)
    
    mae, f1, rmse = calculate_metrics(df_original, df_predicted, userId)
    
    # Setup logging
    logging.basicConfig(filename='result.log', level=logging.INFO)
    
    # Log the results
    logging.info(f'User ID: {userId}')
    logging.info(f'MAE: {mae}')
    logging.info(f'F1-Score: {f1}')
    logging.info(f'RMSE: {rmse}')
    
    return(get_recommendation(df, df_predicted, userId, num_recommended_itemprs))