import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Prepare the training data
train_data = pd.DataFrame({
    'sepal_length': [5, 4.5, 4.4, 5, 5.1, 4.8, 5.1, 4.6, 5.3, 5, 5.5, 6.1, 5.8, 5, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9],
    'sepal_width': [3.5, 2.3, 3.2, 3.5, 3.8, 3, 3.8, 3.2, 3.7, 3.3, 2.6, 3, 2.6, 2.3, 2.7, 3, 2.9, 2.9, 2.5, 2.8, 3.1, 3.1, 2.7, 3.2, 3.3, 3, 2.5, 3, 3.4, 3],
    'petal_length': [1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.4, 4.6, 4, 3.3, 4.2, 4.2, 4.2, 4.3, 3, 4.1, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5, 5.2, 5.4, 5.1],
    'petal_width': [0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.2, 1.4, 1.2, 1, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2, 2.3, 1.8],
    'species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']
})

# Encode the species column to numerical values for training data
label_encoder = LabelEncoder()
train_data['species'] = label_encoder.fit_transform(train_data['species'])

# Split the training data into features (X_train) and target (y_train)
X_train = train_data.drop('species', axis=1)
y_train = train_data['species']

# Create and train the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Prepare the test data to include all three classes (Iris-setosa, Iris-versicolor, Iris-virginica)
test_data = pd.DataFrame({
    'sepal_length': [5.1, 4.9, 4.7, 7, 6.4, 6.9, 6.3, 5.8, 7.1, 6.5],
    'sepal_width': [3.5, 3, 3.2, 3.2, 3.2, 3.1, 2.9, 2.7, 3, 3],
    'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 5.6, 5.1, 5.9, 5.8],
    'petal_width': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.8, 1.9, 2.1, 2.2],
    'species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']
})

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
