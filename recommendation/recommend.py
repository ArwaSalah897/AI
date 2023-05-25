from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# load restaurant data
df = pd.read_csv("restaurant_data.csv")

@app.route('/recommend', methods=['POST'])
def recommend_restaurant():
    # get user input for preferred cuisine and budget
    preferred_cuisine = request.json['cuisine']
    preferred_budget = request.json['budget']

    # filter restaurants based on user preferences
    filtered_df = df[(df["cuisine"] == preferred_cuisine) & (df["price_range"] == preferred_budget)]

    # create a pivot table for collaborative filtering
    pivot_df = filtered_df.pivot_table(index='user_id', columns='restaurant_id', values='rating')

    # fill missing values with 0
    pivot_df = pivot_df.fillna(0)

    # convert pivot table to matrix
    matrix = pivot_df.to_numpy()

    # calculate cosine similarity between restaurants
    similarity_matrix = cosine_similarity(matrix)

    # create a kNN model for collaborative filtering
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(matrix)

    # recommend a restaurant based on collaborative filtering
    user_preferences = pivot_df.loc[1].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_preferences, n_neighbors=3)
    recommended_restaurants = []
    for i in range(len(indices[0])):
        index = indices[0][i]
        recommended_restaurants.append(filtered_df['name'].iloc[index])
    return jsonify({'restaurants': recommended_restaurants})

if __name__ == '__main__':
    app.run(debug=True)