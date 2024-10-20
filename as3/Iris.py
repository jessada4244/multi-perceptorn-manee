import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Import training and test CSV files
train_file_path = 'Iristrain.csv'  # Specify the path of the training CSV file
test_file_path = 'Iristest.csv'    # Specify the path of the test CSV file

# Read training and test datasets
df_train = pd.read_csv(train_file_path, header=None)  # Read training file without header
df_test = pd.read_csv(test_file_path, header=None)    # Read test file without header

# Step 2: Convert class column to numeric for both training and test datasets
df_train['class'] = df_train[4].astype('category').cat.codes  # Column 5 as class for training
df_test['class'] = df_test[4].astype('category').cat.codes    # Column 5 as class for test

# Step 3: Split features (X) and target (y) for both datasets
X_train = df_train.drop([4, 'class'], axis=1)  # Drop class column from training
y_train = df_train['class']

X_test = df_test.drop([4, 'class'], axis=1)    # Drop class column from test
y_test = df_test['class']

# Step 4: Normalize both training and test data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Step 5: Create and train the Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_normalized, y_train)

# Step 6: Predict and calculate accuracy for the test dataset
y_pred_train = tree.predict(X_train_normalized)  # Predictions on training data
train_accuracy = accuracy_score(y_train, y_pred_train)  # Training accuracy

y_pred_test = tree.predict(X_test_normalized)  # Predictions on test data
test_accuracy = accuracy_score(y_test, y_pred_test)  # Test accuracy

print(f'Training Accuracy: {train_accuracy*100:.2f}%')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')

# Step 7: Plot datasets and accuracy
plt.figure(figsize=(18, 5))

# Plot Original Training Dataset
plt.subplot(1, 5, 1)
plt.scatter(X_train[0], X_train[1], c=y_train, cmap='viridis')
plt.title('Original Train Dataset')

# Plot Normalized Training Dataset
plt.subplot(1, 5, 2)
plt.scatter(X_train_normalized[:, 0], X_train_normalized[:, 1], c=y_train, cmap='plasma')
plt.title('Normalized Train Dataset')

# Plot Original Test Dataset
plt.subplot(1, 5, 3)
plt.scatter(X_test[0], X_test[1], c=y_test, cmap='viridis')
plt.title('Original Test Dataset')

# Plot Normalized Test Dataset
plt.subplot(1, 5, 4)
plt.scatter(X_test_normalized[:, 0], X_test_normalized[:, 1], c=y_test, cmap='coolwarm')
plt.title('Normalized Test Dataset')

# Plot Model Accuracy (Train vs Test)
plt.subplot(1, 5, 5)
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy])
plt.title('Model Accuracy (Train vs Test)')
plt.ylim(0, 1)

# Annotate the accuracy values
plt.annotate(f'Train: {train_accuracy*100:.2f}%', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.annotate(f'Test: {test_accuracy*100:.2f}%', xy=(0.55, 0.9), xycoords='axes fraction', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
