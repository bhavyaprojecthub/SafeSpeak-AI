from flask import Flask, request, render_template
import os
import torch
from transformers import BertTokenizer
from model import BERTClassifier
import torch.nn.functional as F

app = Flask(__name__)

# Load model and tokenizer
# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute path of the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model
MODEL_PATH = os.path.join(BASE_DIR, "model", "bert_model.pt")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BERTClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict(text):
    with torch.no_grad():
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)

        # Convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

        prediction = prediction.item()
        confidence = confidence.item() * 100

        result = (
            "Hate Speech Detected"
            if prediction == 1
            else "No Hate Speech Detected"
        )

        return result, round(confidence, 2)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/demo", methods=["GET", "POST"])
def demo():
    prediction = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]
        prediction, confidence = predict(text)

    return render_template(
        "demo.html",
        prediction=prediction,
        confidence=confidence
    )
if __name__ == "__main__":
    app.run(debug=True)