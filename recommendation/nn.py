import pandas as pd
from flask import Flask, jsonify, request
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

app = Flask(__name__)

# load data
data = pd.read_csv('C:/Users/dell/Desktop/test/recommendation/product_data.csv')

# encode user and product ids
user_ids = data['user_id'].unique()
user_id_map = {id: i for i, id in enumerate(user_ids)}
data['user_id'] = data['user_id'].map(user_id_map)

product_names = data['product_name'].unique()
product_id_map = {name: i for i, name in enumerate(product_names)}
data['product_id'] = data['product_name'].map(product_id_map)
# create train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# define model
model = Sequential([
    Dense(32, input_dim=2, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(product_names),activation='softmax')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# train model
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[['user_id', 'product_id']].values.reshape(-1, 2), train_data['rating'].values)).shuffle(len(train_data)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data[['user_id', 'product_id']].values.reshape(-1, 2), test_data['rating'].values)).batch(batch_size)
model.fit(train_dataset, epochs=1000, validation_data=test_dataset)

@app.route('/users', methods=['GET'])
def get_users():
    users = []
    for id in user_ids:
        user = {'user_id': id}
        users.append(user)
    return jsonify(users)
# define prediction function
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json['user_id']
    user_id_encoded = user_id_map[user_id]
    user_input = tf.convert_to_tensor([[user_id_encoded, product_id] for product_id in range(len(product_names))])
    product_ratings = model.predict(user_input)
    product_ratings = product_ratings.tolist()
    product_names_with_ratings_and_images = [(product_names[i], rating, image_url) for i, (rating, image_url) in enumerate(zip(product_ratings, data['image_url'].values.tolist()))]
    product_names_with_ratings_and_images = sorted(product_names_with_ratings_and_images, key=lambda x: x[1], reverse=True)
    recommendations = []
    for product_name, rating, image_url in product_names_with_ratings_and_images:
        recommendation = {'product_name': product_name, 'image_url': image_url}
        recommendations.append(recommendation)
    return jsonify(recommendations)

# run app

if __name__ == "__main__":
    app.run(debug=True)