import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ============================== 
# 1. Load MNIST dataset
# ============================== 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape for CNN input: (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# ============================== 
# 2. Build CNN Model
# ============================== 
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 classes
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ============================== 
# 3. Train model
# ============================== 
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=64
)

# ============================== 
# 4. Plot accuracy curve
# ============================== 
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training Accuracy")
plt.legend()
plt.show()

# ============================== 
# 5. Evaluate model
# ============================== 
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# ============================== 
# 6. Predict a sample
# ============================== 
index = 0
sample = x_test[index].reshape(1, 28, 28, 1)
prediction = model.predict(sample)

print("Model predicted:", np.argmax(prediction))
plt.imshow(x_test[index].reshape(28,28), cmap="gray")
plt.title(f"Prediction: {np.argmax(prediction)}")
plt.show()
plt.savefig("mnist_prediction.png")