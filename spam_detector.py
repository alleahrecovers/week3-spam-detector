
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)

# Define money-related words
MONEY_WORDS = ["money", "cash", "dollars", "free", "prize", "win"]

# Define urgency-related words
URGENCY_WORDS = ["urgent", "immediately", "now", "today", "act fast"]

# Extract six features from an email
def extract_features(email):
    words = email.split()

    word_count = len(words)

    exclamation_count = email.count("!")

    money_count = sum(
        email.lower().count(word) for word in MONEY_WORDS
    )

    all_caps_count = sum(
        1 for word in words
        if len(word) > 1 and word.isupper()
    )

    has_url = int(bool(re.search(
        r"https?://|www\.",
        email,
        re.IGNORECASE
    )))

    urgency_count = sum(
        email.lower().count(word) for word in URGENCY_WORDS
    )

    return [
        word_count,
        exclamation_count,
        money_count,
        all_caps_count,
        has_url,
        urgency_count
    ]


# Load the email dataset
df = pd.read_csv("data/emails.csv")

# Extract features from every email
X = np.array([
    extract_features(email)
    for email in df["email"]
])

# Separate the labels
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create and train the classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Display evaluation results
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nPrecision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))

# Display confusion matrix visually
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Spam"]
)

cm_display.plot()
plt.title("Spam Detector Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()


# Display feature importance
feature_names = [
    "Word Count",
    "Exclamation Marks",
    "Money Words",
    "All Caps",
    "Has URL",
    "Urgency Words"
]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.coef_[0]
})

importance["Absolute Importance"] = importance["Importance"].abs()

importance = importance.sort_values(
    by="Absolute Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)
