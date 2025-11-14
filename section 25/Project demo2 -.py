import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 预处理：缩放到 0~1 之间，并展开成 784 维向量
x_train = x_train.reshape((60000, 28*28)) / 255.0
x_test = x_test.reshape((10000, 28*28)) / 255.0

# 3. 搭建最简单的神经网络（2 层）
model = models.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])

# 4. 编译模型
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. 开始训练
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# 6. 测试集准确率
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# 7. 展示预测效果
plt.figure(figsize=(4,4))
plt.imshow(x_test[0].reshape(28,28), cmap="gray")
plt.title(f"Prediction: {model.predict(x_test[:1]).argmax()}")
plt.show()
plt.savefig('mnist_prediction.png')