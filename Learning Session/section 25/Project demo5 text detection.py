import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. Generate fake OCR data
# =====================================
def generate_fake_data(num=2000):
    X = []
    y = []
    chars = list("abcdefghijklmnopqrstuvwxyz ")
    
    for _ in range(num):
        text = "".join(np.random.choice(chars, size=np.random.randint(3, 12)))
        y.append(text)
        
        img = np.random.rand(32, 128, 1) * 0.5
        img[:, :len(text)*10, :] += 0.5
        X.append(img)
    
    return np.array(X), y

X, y_texts = generate_fake_data(1000)

# character maps
vocab = list("abcdefghijklmnopqrstuvwxyz ")
char_to_id = {c: i+1 for i, c in enumerate(vocab)}  # 0 = blank
id_to_char = {i+1: c for i, c in enumerate(vocab)}

def text_to_ids(t):
    return [char_to_id[c] for c in t]

y = [text_to_ids(t) for t in y_texts]
max_len = max(len(t) for t in y)

y_padded = tf.keras.preprocessing.sequence.pad_sequences(
    y, maxlen=max_len, padding="post"
)

label_lengths = np.array([len(t) for t in y], dtype=np.int32)

# =====================================
# 2. Build CRNN Model (CNN + BiLSTM)
# =====================================
inputs = layers.Input(shape=(32, 128, 1))

x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool2D(2)(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPool2D(2)(x)

shape = x.shape
x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)

x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
outputs = layers.Dense(len(vocab) + 1)(x)  # no softmax for CTC

model = Model(inputs, outputs)

# =====================================
# 3. Custom training with CTC loss
# =====================================
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(images, labels, label_len):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)

        # CTC loss uses time-major: [max_time, batch, num_classes]
        time_major = tf.transpose(logits, [1, 0, 2])

        logit_len = tf.fill([tf.shape(images)[0]], tf.shape(time_major)[0])

        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=time_major,
            label_length=label_len,
            logit_length=logit_len,
            logits_time_major=True,
            blank_index=0
        )

        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

# =====================================
# 4. Training Loop
# =====================================
batch_size = 32
epochs = 5

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0

    for i in range(0, len(X), batch_size):
        images = X[i:i+batch_size]
        labels = y_padded[i:i+batch_size]
        lengths = label_lengths[i:i+batch_size]

        batch_loss = train_step(images, labels, lengths).numpy()
        epoch_loss += batch_loss

    print("Loss:", epoch_loss)


# =====================================
# 5. Prediction (Greedy CTC decode)
# =====================================
def ctc_decode(logits):
    logits = tf.nn.softmax(logits, axis=-1)
    pred = tf.argmax(logits, axis=-1).numpy()
    text = ""
    prev = -1
    for p in pred:
        if p != prev and p != 0:
            text += id_to_char.get(p, "")
        prev = p
    return text


idx = 10
test_img = X[idx:idx+1]
logits = model(test_img)
pred_text = ctc_decode(logits[0])

print("True:", y_texts[idx])
print("Pred:", pred_text)

plt.imshow(test_img[0].reshape(32,128), cmap="gray")
plt.title(f"Pred: {pred_text}")
plt.show()
