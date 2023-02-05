import pandas as pd
import sqlite3
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD

# Connect to the database and load the shopping history and preferences data into a pandas DataFrame
conn = sqlite3.connect('ecommerce.db')
df = pd.read_sql_query("SELECT user_id, product_id, rating FROM shopping_history_preferences", conn)

# Convert the DataFrame into a Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)

# Train a Singular Value Decomposition (SVD) model
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Fit the model to the entire dataset
algo.fit(data.build_full_trainset())

# Use the trained model to make recommendations for a given user
user_id = df['user_id'].iloc[0]
product_ids = df['product_id'].unique()
predictions = []
for product_id in product_ids:
    prediction = algo.predict(user_id, product_id)
    predictions.append((product_id, prediction[3]))

# Sort the predictions by the predicted rating and show the top N recommendations
N = 10
predictions.sort(key=lambda x: x[1], reverse=True)
recommended_product_ids = [product_id for product_id, _ in predictions[:N]]
print("Recommended product IDs: ", recommended_product_ids)
