import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# download nltk data (only first time)
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Step 1: Sample dataset ---
data = {
    "review": [
        "This movie was amazing! I loved the acting and story.",
        "Terrible movie, waste of time.",
        "The plot was dull and predictable.",
        "Fantastic visuals and great soundtrack!",
        "I didn’t enjoy the movie, very boring.",
        "Absolutely wonderful experience, 10/10!",
        "Not bad, but could have been better.",
        "Awful acting, I fell asleep halfway.",
        "Loved every minute of it!",
        "Poorly written and directed."
    ],
    "sentiment": [
        "positive", "negative", "negative", "positive", "negative",
        "positive", "neutral", "negative", "positive", "negative"
    ]
}

df = pd.DataFrame(data)

# --- Step 2: Preprocessing ---
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

df["clean_text"] = df["review"].apply(preprocess)

# --- Step 3: Vectorization ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# --- Step 4: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Step 5: Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Step 7: Try custom text ---
def predict_sentiment(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    return prediction

print("\n🎬 Try custom examples:")
examples = [
    "What a fantastic movie! I enjoyed it a lot.",
    "This was a complete disaster.",
    "Just okay, not great but not bad."
]

for ex in examples:
    print(f"{ex} → {predict_sentiment(ex)}")
