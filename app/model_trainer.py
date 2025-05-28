import os
import pandas as pd
from sklearn.cluster import KMeans
import joblib

data_dir = "/app/output/retail_batches"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
all_data = pd.concat([pd.read_json(f, lines=True) for f in files])

X = all_data[['Quantity', 'UnitPrice']].fillna(0)

# Train 3 models based on portions of data
split_size = len(X) // 3
model_1 = KMeans(n_clusters=3).fit(X[:split_size])
model_2 = KMeans(n_clusters=3).fit(X[:2*split_size])
model_3 = KMeans(n_clusters=3).fit(X)

joblib.dump(model_1, '/app/model_1.pkl')
joblib.dump(model_2, '/app/model_2.pkl')
joblib.dump(model_3, '/app/model_3.pkl')
