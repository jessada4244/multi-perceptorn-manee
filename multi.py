import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Import the training data from a CSV file
train_data = pd.read_csv('Iristrain.csv')

# Encode the species column to numerical values for training data
label_encoder = LabelEncoder()
train_data['species'] = label_encoder.fit_transform(train_data['species'])

# Split the training data into features (X_train) and target (y_train)
X_train = train_data.drop('species', axis=1)
y_train = train_data['species']

# Create and train the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Import the test data from a CSV file
test_data = pd.read_csv('Iristest.csv')

# Encode the species column to numerical values for test data
test_data['species'] = label_encoder.transform(test_data['species'])

# Split the test data into features (X_test) and target (y_test)
X_test = test_data.drop('species', axis=1)
y_test = test_data['species']

# Make predictions on the test data
y_pred = mlp.predict(X_test)

# Output the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], yticklabels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Iris Dataset')
plt.show()
