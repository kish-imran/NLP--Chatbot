# 🤖 NLP Chatbot

## 📌 Project Description

This project is an **NLP-based Chatbot** developed using Python and Machine Learning. The chatbot understands user messages, identifies the user's intent using **TF-IDF Vectorization and Logistic Regression**, and provides appropriate responses.

The project demonstrates important Natural Language Processing concepts including text preprocessing, feature extraction, model training, testing, and prediction.

---

## 🎯 Objectives

* Build a simple NLP chatbot using Python.
* Understand user input using Natural Language Processing.
* Classify user messages into different intents.
* Train a Machine Learning model.
* Use TF-IDF for text feature extraction.
* Use Logistic Regression for intent classification.
* Save the trained model for future predictions.

---

## 🛠️ Technologies Used

* Python
* Natural Language Processing (NLP)
* Machine Learning
* Pandas
* NumPy
* Scikit-learn
* Joblib
* TF-IDF Vectorizer
* Logistic Regression
* VS Code

---

## 📂 Project Structure

```text
NLP-Chatbot/
│
├── dataset/
│   └── chatbot_dataset.csv
│
├── model/
│   ├── chatbot_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── chatbot.py
├── main.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The chatbot uses a CSV dataset named:

```text
chatbot_dataset.csv
```

The dataset contains two main columns:

```text
text
intent
```

Example:

```csv
text,intent
Hello,greeting
Hi,greeting
What is your name?,name
How are you?,status
Bye,goodbye
Thank you,thanks
```

---

## 🧠 Machine Learning Algorithm

### TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts text messages into numerical features that the Machine Learning model can understand.

### Logistic Regression

**Logistic Regression** is used to classify the user's message into an appropriate intent such as:

* Greeting
* Name
* Status
* Goodbye
* Thanks

---

## ⚙️ How the Project Works

```text
User Message
     ↓
Text Input
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Model
     ↓
Intent Prediction
     ↓
Chatbot Response
```

---

## 🚀 Installation

First, make sure Python is installed.

Open the VS Code terminal and install the required libraries:

```bash
pip install pandas numpy scikit-learn joblib
```

You can also install the requirements using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Before running the chatbot, train the Machine Learning model:

```bash
python train_model.py
```

After successful training, the following files will be created automatically:

```text
model/
├── chatbot_model.pkl
└── tfidf_vectorizer.pkl
```

---

## ▶️ Run the Chatbot

After training the model, run:

```bash
python main.py
```

or:

```bash
python chatbot.py
```

---

## 💬 Example Conversation

```text
==================================================
NLP CHATBOT
==================================================

You: Hello
Bot: Hello! How can I help you?

You: What is your name?
Bot: I am an NLP chatbot.

You: How are you?
Bot: I am doing great!

You: Thank you
Bot: You're welcome!

You: Bye
Bot: Goodbye! Have a nice day.
```

---

## 📈 Model Evaluation

The project evaluates the trained model using:

* Accuracy Score
* Classification Report

The training script displays the model accuracy after testing.

Example:

```text
MODEL RESULTS
============================================================

Accuracy: 90.00%
```

The actual accuracy depends on the size and quality of the dataset.

---

## 📁 Important Files

### `train_model.py`

Responsible for:

* Loading the dataset
* Cleaning the data
* Splitting training and testing data
* Creating TF-IDF features
* Training Logistic Regression
* Testing the model
* Saving the model

### `chatbot.py`

Responsible for chatbot prediction and response generation.

### `main.py`

Used to run the chatbot application.

### `dataset/chatbot_dataset.csv`

Contains the training examples and intents.

### `model/`

Stores the trained Machine Learning model and TF-IDF vectorizer.

---

## 🔮 Future Improvements

The project can be improved by adding:

* More training data
* More intents
* Better text preprocessing
* Voice input
* Voice output
* GUI interface
* Web interface
* Chat history
* Deep Learning models
* Transformer-based NLP models

---

## 🎓 Learning Outcomes

Through this project, we learn how to:

1. Work with text datasets.
2. Perform basic NLP processing.
3. Convert text into numerical features.
4. Train a Machine Learning classifier.
5. Evaluate model performance.
6. Save and load trained models.
7. Build a basic chatbot using Python.

---

## 👨‍💻 Author

**Kishwar Imran**

AI / Machine Learning Student & Developer

---

## 📜 License

This project is created for **educational and learning purposes**.
