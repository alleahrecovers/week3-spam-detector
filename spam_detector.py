
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
