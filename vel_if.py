import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from random import sample
df = pd.read_csv('setD_90_mult.csv', parse_dates=['time'])
fraud = df[df['fraud'] == 1]
norm = df[df['fraud'] == 0]

data = pd.concat([fraud, norm[norm['index'].isin(sample(list(norm['index'].unique()), 10000))]])
#data['time_diff'] = data.groupby(['index'])['time'].diff().fillna(pd.Timedelta(seconds=0))
#data['time_diff'] = data['time_diff'] / pd.to_timedelta(1, unit='s')

grouped = data.groupby(['index'], as_index=False).agg({'fraud': 'first', 'location': 'nunique', 'store': {'nunique', 'count'}, 
                                            # 'time': {'min', 'max', 'mean', 'median',}, 
                                             'time_diff': {'max', 'mean', 'median'}})
grouped['time_diff'] = (grouped['time_diff']-grouped['time_diff'].min())/(grouped['time_diff'].max()-grouped['time_diff'].min())

grouped.columns = grouped.columns.droplevel(0)

grouped.columns = ['index', 'fraud', 'uniqueloc', 'uniquestore', 'numstore', 'time_diff_max', 'time_diff_mean', 'time_diff_median']

features = ['uniqueloc', 'uniquestore', 'numstore', 'time_diff_max', 'time_diff_mean', 'time_diff_median']
# Extract the feature values

# Create an Isolation Forest model
isolation_forest = IsolationForest(contamination=0.03, random_state=42)  # Adjust contamination as needed

# Fit the model to the data
isolation_forest.fit(grouped[features])

# Predict anomalies (-1) and inliers (1)
predictions = isolation_forest.predict(grouped[features])

# Add the predictions as a new column to the 'grouped' dataframe
grouped['anomaly_prediction'] = predictions

# Print the count of anomalies
print("Number of anomalies:", np.sum(predictions == -1))
print(grouped.groupby(['fraud', 'anomaly_prediction']).size())
print(grouped[grouped['anomaly_prediction'] == -1].head(50))