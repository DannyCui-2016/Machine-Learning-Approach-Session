import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# ======================================
# 1. 加载数据
# ======================================
df = pd.read_csv("/Users/kc/Desktop/Machine-Learning-Approach-Session/section 25/kc_house_data.csv")

# 只选择几个重要的特征
features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition", "grade",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated"
]

X = df[features].fillna(0)
y = df["price"]

# ======================================
# 2. 划分训练集测试集
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# 3. 数据标准化
# ======================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================
# 4. 构建深度学习模型
# ======================================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ======================================
# 5. 模型训练
# ======================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64
)

# ======================================
# 6. 画出训练曲线
# ======================================
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training History")
plt.show()

# ======================================
# 7. 使用模型预测
# ======================================
example = np.array([[3, 2, 1800, 5000, 1, 0, 0, 3, 7, 1500, 300, 1990, 0]])
example = scaler.transform(example)

pred_price = model.predict(example)
print("预测房价：$", pred_price[0][0])
plt.savefig("house_price_prediction.png")
