from flask import Flask, request, render_template
import os
import torch
from transformers import BertTokenizer
from model import BERTClassifier
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Load model and tokenizer
# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute path of the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "bert_model.pt")

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")

    downloaded_path = hf_hub_download(
        repo_id="bhavyaprojecthub/safespeak-bert-model",
        filename="bert_model.pt",
        local_dir=MODEL_DIR,
    )

    MODEL_PATH = downloaded_path

print("Loading model...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BERTClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")
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