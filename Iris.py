import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Import CSV
file_path = 'Iristest.csv'  # ระบุ path ของไฟล์ CSV ที่ต้องการ
df = pd.read_csv(file_path, header=None)  # อ่านไฟล์ CSV โดยไม่มี header

# ตรวจสอบข้อมูลใน DataFrame
print(df.head())

# Step 2: แปลงค่าคลาสเป็นตัวเลข
df['class'] = df[4].astype('category').cat.codes  # ใช้คอลัมน์ที่ 5 เป็น class

# Step 3: แบ่งข้อมูลเป็น features (X) และ target (y)
X = df.drop([4, 'class'], axis=1)  # เอาคอลัมน์ class ออก
y = df['class']

# Step 4: แบ่งข้อมูลเป็น training และ testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4.1: ทำการ Normalize ข้อมูล
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Step 5: สร้างโมเดล MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, learning_rate_init=0.01, random_state=42)

# Step 6: ฝึกโมเดล
mlp.fit(X_train_normalized, y_train)

# Step 7: ทำนายผลและคำนวณค่า accuracy
y_pred = mlp.predict(X_test_normalized)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')

# Step 8: Plot ข้อมูล dataset และประกาศค่า Accuracy
plt.figure(figsize=(15, 5))

# Plot Original Dataset
plt.subplot(1, 4, 1)
plt.scatter(X[0], X[1], c=y, cmap='viridis')
plt.title('Original Dataset')

# Plot Normalized Train Dataset
plt.subplot(1, 4, 2)
plt.scatter(X_train_normalized[:, 0], X_train_normalized[:, 1], c=y_train, cmap='plasma')
plt.title('Normalized Train Dataset')

# Plot Normalized Test Dataset
plt.subplot(1, 4, 3)
plt.scatter(X_test_normalized[:, 0], X_test_normalized[:, 1], c=y_test, cmap='coolwarm')

# Plot Accuracy
plt.subplot(1, 4, 4)
plt.bar(['Accuracy'], [accuracy])
plt.title('Model Accuracy')
plt.ylim(0, 1)

plt.annotate(f'Accuracy: {accuracy*100:.2f}%', 
             xy=(0.1, 0.9), 
             xycoords='axes fraction', 
             fontsize=12, 
             color='black',
             bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
