import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ========== 1. Load Data ==========
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize images (0~255 → 0~1)
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# ========== 2. Build Model ==========
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# ========== 3. Compile Model (using Adam) ==========
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ========== 4. Train ==========
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32
)

# ========== 5. Evaluate ==========
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# ========== 6. Plot Training Curve ==========
plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
plt.savefig('training_accuracy_curve.png')