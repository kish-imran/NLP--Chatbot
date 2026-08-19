# ============================================================
# NLP CHATBOT - MODEL TRAINING
# ============================================================

import os
import re
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "chatbot.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "chatbot_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

RESPONSES_PATH = os.path.join(
    MODEL_DIR,
    "responses.pkl"
)


# ============================================================
# 2. CREATE MODEL FOLDER
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 60)
print("NLP CHATBOT - MODEL TRAINING")
print("=" * 60)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):

    print("\nERROR!")
    print("Dataset not found:")
    print(DATASET_PATH)

    print("\nMake sure your CSV is here:")
    print("D:\\NLP-Chatbot\\dataset\\chatbot.csv")

    exit()


df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ============================================================
# 5. SHOW COLUMNS
# ============================================================

print("\nDataset columns:")
print(list(df.columns))


# ============================================================
# 6. CHECK YOUR ACTUAL COLUMNS
# ============================================================

if "User Utterance" not in df.columns:

    print("\nERROR!")
    print("The column 'User Utterance' was not found.")

    print("\nYour CSV columns are:")
    print(list(df.columns))

    exit()


if "Bot Response" not in df.columns:

    print("\nERROR!")
    print("The column 'Bot Response' was not found.")

    print("\nYour CSV columns are:")
    print(list(df.columns))

    exit()


# ============================================================
# 7. CREATE TEXT COLUMN
# ============================================================

print("\nCreating 'text' column...")

df["text"] = df["User Utterance"].astype(str)


# ============================================================
# 8. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["text"] = df["text"].apply(clean_text)


# ============================================================
# 9. REMOVE EMPTY TEXT
# ============================================================

df = df[
    df["text"].str.len() > 0
].copy()


# ============================================================
# 10. CREATE INTENT
# ============================================================

print("Creating 'intent' column...")


def create_intent(text):

    text = text.lower()

    words = text.split()

    # -----------------------------
    # Greeting
    # -----------------------------

    if any(word in words for word in [
        "hello",
        "hi",
        "hey",
        "salam"
    ]):
        return "greeting"


    # -----------------------------
    # Goodbye
    # -----------------------------

    if any(word in words for word in [
        "bye",
        "goodbye"
    ]):
        return "goodbye"


    # -----------------------------
    # Thanks
    # -----------------------------

    if (
        "thank you" in text
        or "thanks" in text
        or "thank" in words
    ):
        return "thanks"


    # -----------------------------
    # Name
    # -----------------------------

    if (
        "your name" in text
        or "who are you" in text
        or "what are you" in text
    ):
        return "name"


    # -----------------------------
    # How are you
    # -----------------------------

    if (
        "how are you" in text
        or "how r u" in text
    ):
        return "how_are_you"


    # -----------------------------
    # Help
    # -----------------------------

    if "help" in words:

        return "help"


    # -----------------------------
    # Weather
    # -----------------------------

    if (
        "weather" in text
        or "temperature" in text
    ):
        return "weather"


    # -----------------------------
    # Time
    # -----------------------------

    if "time" in words:

        return "time"


    # -----------------------------
    # Date
    # -----------------------------

    if (
        "date" in words
        or "today" in words
        or "tomorrow" in words
    ):
        return "date"


    # -----------------------------
    # Price
    # -----------------------------

    if (
        "price" in text
        or "cost" in text
        or "expensive" in text
        or "cheap" in text
    ):
        return "price"


    # -----------------------------
    # Problem
    # -----------------------------

    if (
        "problem" in text
        or "error" in text
        or "issue" in text
        or "not working" in text
    ):
        return "problem"


    # -----------------------------
    # Positive
    # -----------------------------

    if any(word in words for word in [
        "good",
        "great",
        "excellent",
        "awesome",
        "amazing"
    ]):
        return "positive"


    # -----------------------------
    # Negative
    # -----------------------------

    if any(word in words for word in [
        "bad",
        "terrible",
        "worst",
        "hate"
    ]):
        return "negative"


    # -----------------------------
    # General
    # -----------------------------

    return "general"


df["intent"] = df["text"].apply(
    create_intent
)


# ============================================================
# 11. DISPLAY CREATED COLUMNS
# ============================================================

print("\nCreated columns successfully!")

print("\nFirst 10 records:")

print(
    df[
        [
            "text",
            "intent",
            "Bot Response"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 12. INTENT DISTRIBUTION
# ============================================================

print("\nIntent distribution:")

print(
    df["intent"].value_counts()
)


# ============================================================
# 13. CHECK INTENTS
# ============================================================

number_of_intents = df["intent"].nunique()

print(
    "\nTotal intents:",
    number_of_intents
)


if number_of_intents < 2:

    print("\nERROR!")
    print("At least two intents are required for training.")

    exit()


# ============================================================
# 14. PREPARE DATA
# ============================================================

X = df["text"]

y = df["intent"]


# ============================================================
# 15. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# 16. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF feature shape:",
    X_train_tfidf.shape
)


# ============================================================
# 17. CREATE MODEL
# ============================================================

print("\nCreating Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


# ============================================================
# 18. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed!")


# ============================================================
# 19. TEST MODEL
# ============================================================

print("\nTesting model...")

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 20. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 21. CREATE RESPONSE DATABASE
# ============================================================

print("\nCreating response database...")

responses = {}

for intent in df["intent"].unique():

    response_list = df[
        df["intent"] == intent
    ]["Bot Response"].tolist()

    responses[intent] = response_list


# ============================================================
# 22. SAVE MODEL
# ============================================================

print("\nSaving chatbot model...")

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# 23. SAVE VECTORIZER
# ============================================================

print("Saving TF-IDF vectorizer...")

with open(
    VECTORIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ============================================================
# 24. SAVE RESPONSES
# ============================================================

print("Saving responses...")

with open(
    RESPONSES_PATH,
    "wb"
) as file:

    pickle.dump(
        responses,
        file
    )


# ============================================================
# 25. SAVE PROCESSED DATASET
# ============================================================

PROCESSED_DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "chatbot_processed.csv"
)

df.to_csv(
    PROCESSED_DATASET_PATH,
    index=False
)


# ============================================================
# 26. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nCreated files:")

print(
    "Model:",
    MODEL_PATH
)

print(
    "Vectorizer:",
    VECTORIZER_PATH
)

print(
    "Responses:",
    RESPONSES_PATH
)

print(
    "Processed Dataset:",
    PROCESSED_DATASET_PATH
)

print("\nTotal records:", len(df))

print(
    "Total intents:",
    number_of_intents
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nYou can now run your chatbot.")

print("=" * 60)