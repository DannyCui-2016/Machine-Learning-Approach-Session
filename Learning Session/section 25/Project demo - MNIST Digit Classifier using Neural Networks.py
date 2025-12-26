import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1) 加载 MNIST 手写数字数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("训练集形状：", x_train.shape)
print("测试集形状：", x_test.shape)

# 2) 数据标准化：像素从 0-255 变成 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3) Label one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4) 构建最基础的神经网络模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")  # 输出10类（0-9）
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5) 训练模型
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 6) 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率：{test_acc:.4f}")

# 7) 可视化训练过程
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")
plt.savefig("training_history.png")

plt.show()
