import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


def load_data_from_csv():
    data = pd.read_csv('C:/Users/xmuxer/OneDrive/College/django-apps/e-commerce/ecommerce/main/data_updated.csv')
    print(data.head())
    
    user_ids = data['CustomerID'].unique()
    descriptions = data['Description'].unique()
    
    matrix = pd.DataFrame(0, index=descriptions, columns=user_ids, dtype=int)
    
    for _, row in data.iterrows():
        matrix.at[row['Description'], row['CustomerID']] += row['Quantity']
    
    return matrix


def recommender(df, customerID):
    number_neighbors = 10  # Increased the number of neighbors
    df1 = df.copy()
    knn = NearestNeighbors(metric='cosine', p=1, algorithm='brute')  # Changed metric to  with p=1
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)
    
    try:
        user_index = df.columns.get_loc(customerID)
    except KeyError:
        print(f"CustomerID {customerID} not found in the dataset.")
        return df, pd.DataFrame()
    
    for m in range(len(df.index)):
        if df.iloc[m, user_index] == 0:
            sim_items = indices[m].tolist()
            item_distances = distances[m].tolist()
            sim_items = sim_items[:number_neighbors-1]
            item_distances = item_distances[:number_neighbors-1]

            item_similarity = [np.exp(-x) for x in item_distances]  # Exponential weighting
            nominator = 0
            for s, sim_item in enumerate(sim_items):
                nominator += item_similarity[s] * df.iloc[sim_item, user_index]

            predicted_rating = nominator / sum(item_similarity) if sum(item_similarity) > 0 else 0
            df1.iloc[m, user_index] = predicted_rating

    return df, df1


def get_recommendation(df, df1, customerID, num_recommendation):
    recommended_items = []
    
    try:
        customer_index = df.columns.get_loc(customerID)
    except KeyError:
        print(f"CustomerID {customerID} not found in the dataset.")
        return []

    for m in df[df.iloc[:, customer_index] == 0].index.tolist():
        index_df = df.index.get_loc(m)
        predicted_rating = df1.iloc[index_df, customer_index]
        recommended_items.append((m, predicted_rating))

    sorted_recommendations = sorted(recommended_items, key=lambda x: x[1], reverse=True)
    sorted_recommendations = sorted_recommendations[:num_recommendation]
    
    return [i[0] for i in sorted_recommendations]


def calculate_metrics(df_original, df_predicted, customerID):
    try:
        user_index = df_original.columns.get_loc(customerID)
    except KeyError:
        print(f"CustomerID {customerID} not found in the dataset.")
        return None, None

    y_true = df_original.iloc[:, user_index].values
    y_pred = df_predicted.iloc[:, user_index].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    
    return mae, rmse


def coll_filter(customerID):
    df = load_data_from_csv()
    df_original, df_predicted = recommender(df, customerID)
    
    if df_predicted.empty:
        return []
    
    mae, rmse = calculate_metrics(df_original, df_predicted, customerID)
    
    logging.basicConfig(filename='result.log', level=logging.INFO)
    logging.info(f'Customer ID: {customerID}')
    logging.info(f'MAE: {mae}')
    logging.info(f'RMSE: {rmse}')
    
    return get_recommendation(df_original, df_predicted, customerID, num_recommendation=4)


# Replace 17850.0 with the appropriate customer ID
recommendations = coll_filter(customerID=13047.0)
print(recommendations)
