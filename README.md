# 🛡️ SafeSpeak - AI Hate Speech Detection using BERT

An AI-powered web application that detects hate speech in real time using a fine-tuned BERT model.

SafeSpeak helps students, educators, researchers, and online communities identify harmful language and promote healthier digital conversations.

---

## 🚀 Features

- 🤖 Fine-tuned BERT model for hate speech detection
- ⚡ Real-time text analysis
- 📊 Confidence score for every prediction
- 🎨 Modern responsive Bootstrap interface
- 🖥️ Flask backend
- 🧠 Hugging Face Transformers
- ☁️ Ready for deployment

---

## 🛠 Tech Stack

### Frontend

- HTML5
- CSS3
- Bootstrap
- JavaScript

### Backend

- Flask
- PyTorch
- Hugging Face Transformers

### AI Model

- bert-base-uncased
- Fine-tuned Binary Classifier

---

## 📂 Project Structure

```text
hate-speech-detector/

│
├── backend/
│   ├── app.py
│   ├── model.py
│   ├── requirements.txt
│   ├── model/
│   │     └── bert_model.pt
│   ├── templates/
│   ├── static/
│   └── training/
│
├── frontend/
│
├── README.md
└── .gitignore
```

---

## 🧠 How It Works

1. User enters text.
2. Text is tokenized using BertTokenizer.
3. The fine-tuned BERT model predicts the class.
4. Softmax converts logits into probabilities.
5. The application displays:

- Prediction
- Confidence Score

---

## ⚙ Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd hate-speech-detector/backend
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

## Trained Model

The trained BERT model (`bert_model.pt`) is not included in this repository because it exceeds GitHub's file size limit.

To run the application:

1. Train the model using the notebook in `backend/training/`.
2. Save the weights backend/model/bert_model.pt
3.Run the flask application

## 📊 Sample Prediction

Input

```
I hate everyone from that community.
```

Output

```
Prediction:
🚨 Hate Speech Detected

Confidence:
98.42%
```

---

## 🔮 Future Improvements

- Multi-class hate speech classification
- Explainable AI using SHAP/LIME
- REST API
- User authentication
- Dashboard for analytics
- Docker support

---

## 👨‍💻 Author

Bhavya

GitHub:
(Add GitHub profile)

LinkedIn:
(Add LinkedIn profile)
