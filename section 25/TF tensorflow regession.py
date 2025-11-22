import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.datasets import fetch_california_housing

# 加载数据
housing = fetch_california_housing()
X = housing.data
y = housing.target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=[X_train.shape[1]]),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)  # 回归任务只有1个输出
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)


loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)


sample = X_test[:5]
pred = model.predict(sample)

print("Predicted:", pred.flatten())
print("Actual:", y_test[:5])

