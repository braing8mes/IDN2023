import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from random import sample
df = pd.read_csv('setD_90_mult.csv', parse_dates=['time'])
# print(data.head(50))
fraud = df[df['fraud'] == 1]
norm = df[df['fraud'] == 0]
data = pd.concat([fraud, norm[norm['index'].isin(sample(list(norm['index'].unique()), 3000))]])
data['time_diff'] = data.groupby(['index'])['time'].diff().fillna(pd.Timedelta(seconds=0))
data['time_diff'] = data['time_diff'] / pd.to_timedelta(1, unit='s')

grouped = data.groupby(['index'], as_index=False).agg({'fraud': 'first', 'location': 'nunique', 'store': {'nunique', 'count'}, 
                                            # 'time': {'min', 'max', 'mean', 'median',}, 
                                             'time_diff': {'max', 'mean', 'median'}})

features = ['location', 'store', 'time_diff']
X = grouped[features]
y = np.array(grouped['fraud']).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
# print(f'parameters: {svm.get_params()}')
print('Accuracy of SVM classifier on training set: {:.3f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.3f}'
     .format(svm.score(X_test, y_test)))
print(f'F1 score: {f1_score(y_test, y_pred)}')
print(pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).head(50))