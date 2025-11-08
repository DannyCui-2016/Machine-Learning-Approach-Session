import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download resources (only first time)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

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
        "Poorly written and directed.",
        "Superb film! The characters felt real.",
        "Disgusting script, horrible acting.",
        "Mediocre movie with decent visuals.",
        "Brilliant! I’d watch it again.",
        "One of the worst films ever made.",
        "Incredible performance by the lead actor.",
        "Boring and way too long.",
        "Heartwarming and touching story.",
        "Pathetic plot and poor execution.",
        "Cinematic masterpiece with great pacing."
    ],
    "sentiment": [
        "positive", "negative", "negative", "positive", "negative",
        "positive", "neutral", "negative", "positive", "negative",
        "positive", "negative", "neutral", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive"
    ]
}

df = pd.DataFrame(data)
df = df[df["sentiment"] != "neutral"]  # ✅ Remove neutral *after* DataFrame exists

# --- Step 2: Preprocessing ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(filtered)

df["clean_text"] = df["review"].apply(preprocess)

# --- Step 3: Vectorization ---
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

# --- Step 4: Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Step 5: Train ---
model = MultinomialNB()
model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred))

# --- Step 7: Try examples ---
def predict_sentiment(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

print("\n🎬 Try custom examples:")
examples = [
    "What a fantastic movie! I enjoyed it a lot.",
    "This was a complete disaster.",
    "Just okay, not great but not bad."
]
for ex in examples:
    print(f"{ex} → {predict_sentiment(ex)}")
