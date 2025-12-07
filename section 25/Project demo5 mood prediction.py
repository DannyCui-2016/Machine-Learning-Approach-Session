import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# =============================
# 1. 小型训练数据样本
# =============================
items = [
    "塑料瓶", "矿泉水瓶", "啤酒瓶", "饮料瓶", "易拉罐", "快递纸盒", "玻璃瓶",
    "果皮", "剩饭", "菜叶", "茶叶渣", "骨头", "西瓜皮", "香蕉皮",
    "电池", "荧光灯", "药品", "油漆桶", "废旧灯泡",
    "灰土", "纸巾", "烟头", "尘土", "口罩"
]

labels = [
    0,0,0,0,0,0,0,   # 可回收物 0
    1,1,1,1,1,1,1,   # 厨余垃圾 1
    2,2,2,2,2,       # 有害垃圾 2
    3,3,3,3,3        # 其他垃圾 3
]

label_names = ["可回收物♻", "厨余垃圾🍃", "有害垃圾☣", "其他垃圾🗑"]

# =============================
# 2. Text Tokenize
# =============================
tokenizer = Tokenizer()
tokenizer.fit_on_texts(items)

X = tokenizer.texts_to_sequences(items)
X = pad_sequences(X, maxlen=3)
y = np.array(labels)

# =============================
# 3. Build & Train model
# =============================
model = Sequential([
    Embedding(input_dim=50, output_dim=16, input_length=3),
    LSTM(32),
    Dense(4, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=50, verbose=0)

print("训练完成！")

# =============================
# 4. 预测函数
# =============================
def classify(item):
    seq = tokenizer.texts_to_sequences([item])
    seq = pad_sequences(seq, maxlen=3)
    pred = model.predict(seq)[0]
    return label_names[np.argmax(pred)]

# 测试
print(classify("矿泉水瓶"))
print(classify("剩饭"))
print(classify("电池"))
print(classify("口罩"))
