import os
import re
import joblib

from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# Model paths
# -------------------------------------------------

MODEL_PATH = "model/chatbot_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


# -------------------------------------------------
# Text cleaning
# -------------------------------------------------

def clean_text(text):

    text = text.lower()

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


# -------------------------------------------------
# Load model
# -------------------------------------------------

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "Chatbot model not found! "
        "Run train_model.py first."
    )


if not os.path.exists(VECTORIZER_PATH):

    raise FileNotFoundError(
        "Vectorizer not found! "
        "Run train_model.py first."
    )


model_data = joblib.load(MODEL_PATH)

vectorizer = joblib.load(VECTORIZER_PATH)


questions = model_data["questions"]

answers = model_data["answers"]

question_vectors = model_data["vectors"]


# -------------------------------------------------
# Generate chatbot response
# -------------------------------------------------

def get_response(user_input):

    cleaned_input = clean_text(user_input)

    if cleaned_input == "":
        return "Please type something."


    # Convert user input into TF-IDF vector
    user_vector = vectorizer.transform(
        [cleaned_input]
    )


    # Calculate similarity
    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )


    # Find highest score
    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]


    # Minimum confidence
    if best_score < 0.20:

        return (
            "Sorry, I don't understand that yet. "
            "Please ask another question."
        )


    return answers[best_match_index]