import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from random import sample
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
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
features = ['location', 'store', 'time_diff']
X = grouped[features]
y = np.array(grouped['fraud']).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

# Confusion Matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
conf_matrix_rearranged = np.array([[tp, fn], [fp, tn]])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rearranged, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
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

# Classification Report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

"""print(f'parameters: {svm.get_params()}')
print('Accuracy of SVM classifier on training set: {:.3f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.3f}'
     .format(svm.score(X_test, y_test)))
print(f'F1 score: {f1_score(y_test, y_pred)}')
# print(pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).head(200))"""