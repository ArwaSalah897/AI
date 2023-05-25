import pandas as pd
from flask import Flask, jsonify, request
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import psycopg2

app = Flask(__name__)

# establish connection to PostgreSQL database
conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_username",
    password="your_password"
)

# load data from PostgreSQL
data = pd.read_sql_query("SELECT user_id, product_name, rating, product_image_url FROM your_table", conn)

# encode user and product ids
user_ids = data['user_id'].unique()
user_id_map = {id: i for i, id in enumerate(user_ids)}
data['user_id'] = data['user_id'].map(user_id_map)

product_names = data['product_name'].unique()
product_id_map = {name: i for i, name in enumerate(product_names)}
data['product_id'] = data['product_name'].map(product_id_map)

product_image_urls = data[['product_id', 'product_image_url']].drop_duplicates().set_index('productID')

# create train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# define model
model = Sequential([
    Dense(32, input_dim=2, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(product_names), activation='softmax')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# train model
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[['user_id', 'product_id']].values.reshape(-1, 2), train_data['rating'].values)).shuffle(len(train_data)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data[['user_id', 'product_id']].values.reshape(-1, 2), test_data['rating'].values)).batch(batch_size)
model.fit(train_dataset, epochs=1000, validation_data=test_dataset)

# define prediction function
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json['user_id']
    user_id_encoded = user_id_map[user_id]
    user_input = tf.convert_to_tensor([[user_id_encoded, product_id] for product_id in range(len(product_names))])
    product_ratings = model.predict(user_input)
    product_ratings = product_ratings.tolist()
    product_names_with_ratings = [(product_names[i], rating, product_image_urls.loc[i]['product_imageSorry, it seems like my previous response was cut off. Here's the complete modified code to show the name and image of the recommended product:


import pandas as pd
from flask import Flask, jsonify, request
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import psycopg2

app = Flask(__name__)

# establish connection to PostgreSQL database
conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_username",
    password="your_password"
)

# load data from PostgreSQL
data = pd.read_sql_query("SELECT user_id, product_name, rating, product_image_url FROM your_table", conn)

# encode user and product ids
user_ids = data['user_id'].unique()
user_id_map = {id: i for i, id in enumerate(user_ids)}
data['user_id'] = data['user_id'].map(user_id_map)

product_names = data['product_name'].unique()
product_id_map = {name: i for i, name in enumerate(product_names)}
data['product_id'] = data['product_name'].map(product_id_map)

product_image_urls = data[['product_id', 'product_image_url']].drop_duplicates().set_index('product_id')

# create train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# define modelmodel = Sequential([
    Dense(32, input_dim=2, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(product_names), activation='softmax')
])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# train model
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[['user_id', 'product_id']].values.reshape(-1, 2), train_data['rating'].values)).shuffle(len(train_data)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data[['user_id', 'product_id']].values.reshape(-1, 2), test_data['rating'].values)).batch(batch_size)
model.fit(train_dataset, epochs=1000, validation_data=test_dataset)

# define prediction function
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json['user_id']
    user_id_encoded = user_id_map[user_id]
    user_input = tf.convert_to_tensor([[user_id_encoded, product_id] for product_id in range(len(product_names))])
    product_ratings = model.predict(user_input)
    product_ratings = product_ratings.tolist()
    product_names_with_ratings = [(product_names[i], rating, product_image_urls.loc[i]['product_image_url']) for i, rating in enumerate(product_ratings)]
    product_names_with_ratings = sorted(product_names_with_ratings, key=lambda x: x[1], reverse=True)
    recommendations```
@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json['user_id']
    user_id_encoded = user_id_map[user_id]
    user_input = tf.convert_to_tensor([[user_id_encoded, product_id] for product_id in range(len(product_names))])
    product_ratings = model.predict(user_input)
    product_ratings = product_ratings.tolist()
    product_names_with_ratings = [(product_names[i], rating, product_image_urls.loc[i]['product_image_url']) for i, rating in enumerate(product_ratings)]
    product_names_with_ratings = sorted(product_names_with_ratings, key=lambda x: x[1], reverse=True)
    recommendations = [{'name': name, 'rating': rating, 'image_url': image_url} for name, rating, image_url in product_names_with_ratings]
    return jsonify(recommendations)