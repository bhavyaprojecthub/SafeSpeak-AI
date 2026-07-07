# SafeSpeak – AI-Powered Hate Speech Detection using BERT

SafeSpeak is an AI-powered web application that detects hate speech in real time using a fine-tuned BERT Transformer model. Users can enter text through an intuitive web interface and receive instant predictions, demonstrating the practical application of Natural Language Processing (NLP) for text classification.


## Features

- AI-powered hate speech detection using a fine-tuned BERT Transformer model
- Real-time text classification through an interactive web interface
- Prediction confidence score for each classification
- Clean and responsive user interface for seamless user experience
- Efficient text preprocessing and tokenization before prediction

## Tech Stack

### Frontend

- HTML5
- CSS3
- Bootstrap
- JavaScript

### Backend

- Python
- Flask
- PyTorch
- Hugging Face Transformers

### AI Model

- BERT (`bert-base-uncased`)
- Fine-tuned for binary hate speech classification

## Project Architecture

### Overview

SafeSpeak follows a client-server architecture that integrates a web interface with a fine-tuned BERT Transformer model for real-time hate speech detection. The frontend collects user input, the Flask backend processes the request, and the AI model performs text classification before returning the prediction and confidence score to the user.

### Workflow

1. User Input
   - The user enters text into the web application.

2. Request Handling
   - The frontend sends the input text to the Flask backend through an HTTP POST request.

3. Text Preprocessing
   - The backend tokenizes the input using the BERT tokenizer, converting the text into a format suitable for the model.

4. Model Inference
   - The tokenized input is passed to the fine-tuned BERT model, which predicts the appropriate class.

5. Prediction Generation
   - The model returns the predicted label along with its confidence score.

6. Response
   - The Flask backend sends the prediction back to the frontend.

7. Result Display
   - The web interface displays the predicted class and confidence score to the user.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.10 or later
- pip
- Git

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd SafeSpeak/backend
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

The trained model weights (`bert_model.pt`) are not included in this repository because the file exceeds GitHub's size limitations.

To run the application locally:

1. Train the model using the notebook available in `backend/training/`.
2. Save the trained model as:

```text
backend/model/bert_model.pt
```

3. Start the Flask application:

```bash
python app.py
```

## Sample Prediction

### Input

```text
I hate everyone from that community.
```

### Output

```text
Prediction:
Hate Speech Detected

Confidence:
98.42%
```

## Future Improvements

- Multi-class hate speech classification
- Explainable AI using SHAP/LIME
- REST API
- User authentication
- Dashboard for analytics
- Docker support

