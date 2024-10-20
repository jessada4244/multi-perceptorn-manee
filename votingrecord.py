import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load training dataset
train_file_path = 'Votingrecordtrain.csv'  # Adjust the path if necessary
df_train = pd.read_csv(train_file_path)

# Apply Label Encoding to all columns in the training dataset
le = LabelEncoder()

# Encode class labels (first column)
df_train[df_train.columns[0]] = le.fit_transform(df_train[df_train.columns[0]])

# Encode all features in training data
for column in df_train.columns[1:]:
    df_train[column] = le.fit_transform(df_train[column])

# Split data into features (X) and target (y)
X_train = df_train.drop(columns=[df_train.columns[0]])  # Drop first column assuming it's the class
y_train = df_train[df_train.columns[0]]  # Target labels are in the first column

# Normalize the training data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)

# Create the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, learning_rate_init=0.01, random_state=42)

# Train the model
mlp.fit(X_train_normalized, y_train)

# Load test dataset
test_file_path = 'Votingrecordtest.csv'  # Adjust the path if necessary
df_test = pd.read_csv(test_file_path)

# Apply Label Encoding to the test dataset
df_test[df_test.columns[0]] = le.fit_transform(df_test[df_test.columns[0]])

# Encode all features in test data
for column in df_test.columns[1:]:
    df_test[column] = le.fit_transform(df_test[column])

# Adjust test data to match the structure of the training data
X_test = df_test.drop(columns=[df_test.columns[0]])  # Drop the first column from the test dataset

# Normalize the test data using the same scaler as for the training data
X_test_normalized = scaler.transform(X_test.values)  # Use .values to ignore column names

# Predict using the trained model
y_test_pred = mlp.predict(X_test_normalized)

# Calculate the accuracy on the training data
y_train_pred = mlp.predict(X_train_normalized)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate the accuracy on the test data
test_accuracy = accuracy_score(df_test[df_test.columns[0]], y_test_pred)

# Plot the datasets and accuracy
plt.figure(figsize=(18, 6))

# Original training dataset
plt.subplot(1, 5, 1)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap='viridis')
plt.title('Original Training Dataset')

# Original test dataset
plt.subplot(1, 5, 2)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test_pred, cmap='viridis')
plt.title('Original Test Dataset')

# Normalized training dataset
plt.subplot(1, 5, 3)
plt.scatter(X_train_normalized[:, 0], X_train_normalized[:, 1], c=y_train, cmap='plasma')
plt.title('Normalized Training Dataset')

# Normalized test dataset
plt.subplot(1, 5, 4)
plt.scatter(X_test_normalized[:, 0], X_test_normalized[:, 1], c=y_test_pred, cmap='coolwarm')
plt.title('Normalized Test Dataset')

# Accuracy plot for both training and test datasets
plt.subplot(1, 5, 5)
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy])
plt.title('Model Accuracy (Train vs Test)')
plt.ylim(0, 1)

# Annotate the accuracy values
plt.annotate(f'Train: {train_accuracy*100:.2f}%', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.annotate(f'Test: {test_accuracy*100:.2f}%', xy=(0.55, 0.9), xycoords='axes fraction', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
