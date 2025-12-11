import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# ============================
# 1. 构造生活食物数据
# ============================
foods = [
    "苹果", "香蕉", "西红柿", "胡萝卜", "鸡胸肉", "西兰花", "酸奶",
    "薯条", "炸鸡", "可乐", "汉堡", "饼干", "蛋糕", "巧克力"
]

labels = [
    0,0,0,0,0,0,0,   # 健康 0
    1,1,1,1,1,1,1    # 不健康 1
]

label_name = ["健康🥗","不健康🍟"]

# ============================
# 2. Tokenizer 文本数字化
# ============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(foods)

X = tokenizer.texts_to_sequences(foods)
X = pad_sequences(X, maxlen=3)
y = np.array(labels)

# ============================
# 3. LSTM 分类模型
# ============================
model = Sequential([
    Embedding(input_dim=50, output_dim=16, input_length=3),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# ============================
# 4. 训练
# ============================
model.fit(X, y, epochs=50, verbose=0)
print("训练完成 ✔")

# ============================
# 5. 预测函数
# ============================
def classify_food(name):
    seq = tokenizer.texts_to_sequences([name])
    seq = pad_sequences(seq, maxlen=3)
    pred = model.predict(seq)[0][0]
    return label_name[1 if pred>0.5 else 0]

# 测试
print(classify_food("苹果"))
print(classify_food("炸鸡"))
print(classify_food("酸奶"))
print(classify_food("蛋糕"))
