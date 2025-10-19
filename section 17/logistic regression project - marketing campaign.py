# Logistic Regression - Predict if customer buys a product


# =======================
# 1. Import libraries
# =======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =======================
# 2. Create a realistic dataset
# =======================
np.random.seed(42)  # for reproducibility

n = 300  # total number of samples

# Create random but realistic features
age = np.random.randint(20, 60, n)                  # age between 20 and 60
income = np.random.randint(30000, 120000, n)        # annual income
has_children = np.random.choice([0, 1], n)          # binary: 0=no children, 1=has children
visited_website = np.random.choice([0, 1], n)       # binary: visited marketing site
ad_clicked = np.random.choice([0, 1], n)            # binary: clicked online ad

# Construct target variable (buy or not)
# We create a probabilistic relationship:
# older + higher income + visited website + clicked ad → higher purchase probability
prob_buy = (
    0.2 * (age / 60)
    + 0.4 * (income / 120000)
    + 0.3 * visited_website
    + 0.5 * ad_clicked
    - 0.1 * has_children
)
# Convert to 0/1 using a threshold
buy_product = (prob_buy + np.random.normal(0, 0.1, n) > 0.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Age': age,
    'Income': income,
    'Has_Children': has_children,
    'Visited_Website': visited_website,
    'Ad_Clicked': ad_clicked,
    'Buy_Product': buy_product
})

print("✅ Sample of dataset:")
print(df.head(), "\n")

# =======================
# 3. Split dataset into train and test
# =======================
X = df[['Age', 'Income', 'Has_Children', 'Visited_Website', 'Ad_Clicked']]
y = df['Buy_Product']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =======================
# 4. Train logistic regression model
# =======================
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# =======================
# 5. Make predictions
# =======================
y_pred = log_model.predict(X_test)

# =======================
# 6. Evaluate model
# =======================
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("🎯 Accuracy:", round(acc, 3))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =======================
# 7. Visualize coefficients
# =======================
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='green')
plt.xlabel("Coefficient value")
plt.title("Feature importance (Logistic Regression)")
plt.gca().invert_yaxis()
plt.show()
plt.savefig("feature_importance_logistic_regression.png")
