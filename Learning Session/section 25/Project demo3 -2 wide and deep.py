import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# ==========================
# 1. Load Data
# ==========================
df = pd.read_csv("/Users/kc/Desktop/Machine-Learning-Approach-Session/section 25/kc_house_data.csv")

features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition", "grade",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated"
]

X = df[features].fillna(0)
y = df["price"].values

# Train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================
# 2. Build Wide & Deep Model
# ==========================
wide = layers.Input(shape=(X_train.shape[1],), name="wide_input")

deep = layers.Dense(128, activation="relu")(wide)
deep = layers.BatchNormalization()(deep)
deep = layers.Dropout(0.3)(deep)

deep = layers.Dense(64, activation="relu")(deep)
deep = layers.BatchNormalization()(deep)
deep = layers.Dropout(0.2)(deep)

deep = layers.Dense(32, activation="relu")(deep)

# merge wide + deep
combined = layers.concatenate([wide, deep])

output = layers.Dense(1)(combined)

model = models.Model(inputs=wide, outputs=output)

model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()


# ==========================
# 3. Train Model
# ==========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=80,
    batch_size=64
)


# ==========================
# 4. Plot Training Curve
# ==========================
plt.plot(history.history["loss"], label="Training MSE")
plt.plot(history.history["val_loss"], label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.legend()
plt.show()


# ==========================
# 5. Evaluate
# ==========================
test_loss, test_rmse = model.evaluate(X_test, y_test)
print("Test RMSE:", test_rmse)


# ==========================
# 6. Predict
# ==========================
example = np.array([[3, 2, 1800, 5000, 1, 0, 0, 3, 7, 1500, 300, 1990, 0]])
example_scaled = scaler.transform(example)

pred = model.predict(example_scaled)
print("Predicted price:", pred[0][0])
plt.savefig("house_price_prediction_wide_and_deep.png")