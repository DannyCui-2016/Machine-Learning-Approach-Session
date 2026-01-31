from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 数据集
texts = [
    "I love this",
    "This is great",
    "I hate this",
    "This is terrible"
]

labels = [1, 1, 0, 0]  # 1=positive, 0=negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# 测试
test = ["I love it"]
X_test = vectorizer.transform(test)
prediction = model.predict(X_test)

print("预测结果:", "😊 Positive" if prediction[0] == 1 else "😢 Negative")


