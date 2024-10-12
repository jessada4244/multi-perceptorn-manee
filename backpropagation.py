import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# เตรียมข้อมูลฝึก (Train Data)
train_data = pd.DataFrame({
    'sepal_length': [5, 4.5, 4.4, 5, 5.1, 4.8, 5.1, 4.6, 5.3, 5, 5.5, 6.1, 5.8, 5, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9],
    'sepal_width': [3.5, 2.3, 3.2, 3.5, 3.8, 3, 3.8, 3.2, 3.7, 3.3, 2.6, 3, 2.6, 2.3, 2.7, 3, 2.9, 2.9, 2.5, 2.8, 3.1, 3.1, 2.7, 3.2, 3.3, 3, 2.5, 3, 3.4, 3],
    'petal_length': [1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.4, 4.6, 4, 3.3, 4.2, 4.2, 4.2, 4.3, 3, 4.1, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5, 5.2, 5.4, 5.1],
    'petal_width': [0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.2, 1.4, 1.2, 1, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2, 2.3, 1.8],
    'species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']
})

# แยก features และ labels
train_labels = train_data['species']
train_data = train_data.drop('species', axis=1)

# แปลง labels ให้เป็นตัวเลข
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# แปลงข้อมูลให้เป็น numpy array
train_data = train_data.values

# เตรียมข้อมูลทดสอบ (Test Data)
test_data = pd.DataFrame({
    'sepal_length': [5.1, 4.9, 4.7, 7, 6.4, 6.9, 6.3, 5.8, 7.1, 6.5],
    'sepal_width': [3.5, 3, 3.2, 3.2, 3.2, 3.1, 2.9, 2.7, 3, 3],
    'petal_length': [1.4, 1.4, 1.3, 4.7, 4.5, 4.9, 5.6, 5.1, 5.9, 5.8],
    'petal_width': [0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.8, 1.9, 2.1, 2.2],
    'species': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica', 'Iris-virginica']
})

# แยก features และ labels ของ test data
test_labels = test_data['species']
test_data = test_data.drop('species', axis=1)

# แปลง labels ของ test data เป็นตัวเลข
test_labels_encoded = label_encoder.transform(test_labels)

# แปลงข้อมูล test data ให้เป็น numpy array
test_data = test_data.values

# สร้างโมเดล MLP
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))  # Hidden layer
model.add(Dense(10, activation='relu'))               # Hidden layer
model.add(Dense(3, activation='softmax'))             # Output layer

# กำหนด loss function, optimizer และ metric ที่ใช้ในการฝึก
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ฝึกโมเดล (พร้อม backpropagation อัตโนมัติ)
model.fit(train_data, train_labels_encoded, epochs=100, batch_size=5, verbose=1)

# ทำนายผลลัพธ์จากชุดทดสอบ
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

# สร้าง confusion matrix
cm = confusion_matrix(test_labels_encoded, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], rotation=45)
plt.yticks(tick_marks, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# ใส่ตัวเลขลงในช่อง confusion matrix
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
