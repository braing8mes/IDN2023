import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from random import sample
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 25, num = 3)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 5]
# Method of selecting samples for training each tree
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

df = pd.read_csv('setD_90_mult.csv', parse_dates=['time'])
fraud = df[df['fraud'] == 1]
norm = df[df['fraud'] == 0]

data = pd.concat([fraud, norm[norm['index'].isin(sample(list(norm['index'].unique()), 4000))]])
data['time_diff'] = data.groupby(['index'])['time'].diff().fillna(pd.Timedelta(seconds=0))
data['time_diff'] = data['time_diff'] / pd.to_timedelta(1, unit='s')

grouped = data.groupby(['index'], as_index=False).agg({'fraud': 'first', 'location': 'nunique', 'store': {'nunique', 'count'}, 
                                            # 'time': {'min', 'max', 'mean', 'median',}, 
                                             'time_diff': {'max', 'mean', 'median'}})

grouped['time_diff'] = (grouped['time_diff']-grouped['time_diff'].min())/(grouped['time_diff'].max()-grouped['time_diff'].min())
grouped.columns = grouped.columns.droplevel(0)

grouped.columns = ['index', 'fraud', 'uniqueloc', 'uniquestore', 'numstore', 'time_diff_max', 'time_diff_mean', 'time_diff_median']

features = ['uniqueloc', 'uniquestore', 'numstore', 'time_diff_max', 'time_diff_mean', 'time_diff_median']
X = grouped[features]
y = np.array(grouped['fraud']).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
y_pred = rf_random.predict(X_test)
y_prob = rf_random.predict_proba(X_test)[:, 1]

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
conf_matrix_rearranged = np.array([[tp, fn], [fp, tn]])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rearranged, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Feature Importance Plot
feature_importance = rf_random.best_estimator_.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(features, feature_importance)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
average_precision = average_precision_score(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.title("Precision-Recall Curve")
plt.plot(recall, precision, color='darkorange', lw=2, label=f'Avg Precision = {average_precision:.2f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper right")
plt.show()







"""
print(f'parameters: {rf_random.best_params_}')
print('Accuracy of forest classifier on training set: {:.3f}'
     .format(rf_random.score(X_train, y_train)))
print('Accuracy of forest classifier on test set: {:.3f}'
     .format(rf_random.score(X_test, y_test)))
print(f'F1 score: {f1_score(y_test, y_pred)}')
print(pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).head(50))"""