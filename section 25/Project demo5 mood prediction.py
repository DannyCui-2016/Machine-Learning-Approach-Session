


import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. 构造模拟生活行为数据（可替换成真实数据）
# =====================================
# 特征：[睡觉时间(小时24制), 起床时间, 咖啡杯数, 是否午睡(1/0), 夜间精神(1高/0低)]
X = np.array([
    [22, 6, 1, 1, 0], [23, 7, 2, 1, 0], [21, 6, 0, 1, 0],
    [1, 9, 3, 0, 1],  [2, 10, 4, 0, 1], [0, 8, 3, 0, 1],
    [23, 7, 1, 1, 0], [22, 5, 2, 1, 0], [3, 11, 4, 0, 1],
    [4, 12, 5, 0, 1], [21, 6, 1, 1, 0], [1, 9, 4, 0, 1],
])

# 标签：0 = 早起型，1 = 夜猫子
y = np.array([0,0,0,1,1,1,0,0,1,1,0,1])

# =====================================
# 2. 数据拆分 + 标准化
# =====================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 3. 构建分类模型
# =====================================
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")  # 二分类输出
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# =====================================
# 4. 训练
# =====================================
model.fit(X_train, y_train, epochs=30, verbose=0)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n模型准确率: {acc*100:.2f}%")

# =====================================
# 5. 新样本预测
# =====================================
def predict_type(data):
    data = scaler.transform([data])
    pred = model.predict(data)[0][0]
    return "夜猫子 🦉" if pred>0.5 else "早起型 ☀️"

# 测试一个新输入
print("\n预测测试:")
print(predict_type([22, 6, 1, 1, 0]))
print(predict_type([2, 10, 4, 0, 1]))

