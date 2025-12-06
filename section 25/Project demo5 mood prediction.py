import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ===================================
# 1. Toy Dataset (小样本示例数据)
# ===================================
sentences = [
    "I love this movie",
    "This film is amazing",
    "What a great day",
    "I am very happy",
    "This product is bad",
    "I hate this movie",
    "This is terrible",
    "I am very sad"
]

labels = [1,1,1,1,0,0,0,0]  # 1=positive, 0=negative

# ===================================
# 2. Tokenization
# ===================================
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(sentences)

X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X, maxlen=6)

y = np.array(labels)

# ===================================
# 3. Build Model
# ===================================
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=6),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ===================================
# 4. Train
# ===================================
model.fit(X, y, epochs=15)

# ===================================
# 5. Predict
# ===================================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=6)
    pred = model.predict(seq)[0][0]
    return "Positive 😀" if pred>0.5 else "Negative ☹️"

print(predict_sentiment("I really love it"))
print(predict_sentiment("This feels bad"))
print(predict_sentiment("Amazing product"))
print(predict_sentiment("I hate everything"))
