import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
df = pd.read_csv('setD_90_mult.csv', parse_dates=['time'])
fraud = df[df['fraud'] == 1]
print(len(fraud))
norm = df[df['fraud'] == 0]

data = pd.concat([fraud, norm[norm['index'].isin(sample(list(norm['index'].unique()), 5000))]])
data['time_diff'] = data.groupby(['index'])['time'].diff().fillna(pd.Timedelta(seconds=0))
data['time_diff'] = data['time_diff'] / pd.to_timedelta(1, unit='s')

grouped = data.groupby(['index'], as_index=False).agg({'fraud': 'first', 'location': 'nunique', 'store': {'nunique', 'count'}, 
                                            # 'time': {'min', 'max', 'mean', 'median',}, 
                                             'time_diff': {'max', 'mean', 'median'}})

grouped['time_diff'] = (grouped['time_diff']-grouped['time_diff'].min())/(grouped['time_diff'].max()-grouped['time_diff'].min())
features = ['location', 'store', 'time_diff']
X = grouped[features]
y = np.array(grouped['fraud']).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = torch.sigmoid(self.hidden_layer(x))
        return x

# Define the input size and hidden size for the model
input_size = 6
hidden_size = 3

# Define a function for plotting the training process
def plot_training_progress(loss_values):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, marker='o', linestyle='-', color='b', markersize=2, label='Training Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Create the model instance
model = BinaryClassifier(input_size, hidden_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Training the model
num_epochs = 3000

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from random import sample

# ... (your data preprocessing code) ...

# Define a function for plotting the training process
def plot_training_progress(loss_values):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_values, marker='o', linestyle='-', color='b', markersize=2, label='Training Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Training the model
num_epochs = 3000
loss_values = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    # Append the loss for plotting
    loss_values.append(loss.item())

print("Training finished!")

plot_training_progress(loss_values)

# Convert X_test to a PyTorch tensor
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# Evaluation
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor)
    y_pred = (y_pred_prob >= 0.5).float().view(-1)

# Calculate accuracy
accuracy = (y_pred == torch.tensor(y_test, dtype=torch.float32)).float().mean()
print(f"Accuracy: {accuracy.item()}")
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
pr_auc = average_precision_score(y_test, y_pred_prob)
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {roc_auc}")
print(f"AUC-PR: {pr_auc}")

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Rearrange confusion matrix
conf_matrix_rearranged = np.array([[tp, fn], [fp, tn]])

# Visualize the rearranged confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rearranged, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()






