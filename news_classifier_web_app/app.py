import os
import gdown
from flask import Flask, render_template, request
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Initialize the Flask application
app = Flask(__name__)

# Model files and their corresponding Google Drive links
files = {
    'config.json': 'https://drive.google.com/uc?id=1XTo2MZOvTQ55jSwM6G2yOs4OMiKWegEr',
    'model.safetensors': 'https://drive.google.com/uc?id=1MMxuq-zVxTc8au6VJaS37a5cQhQA75M4',
    'special_tokens_map.json': 'https://drive.google.com/uc?id=1v3SiKX_I5Hjb64cdvPRhiRCBI_UtPyxy',
    'tokenizer_config.json': 'https://drive.google.com/uc?id=1ndbo1nEOmBhBVKxz5Ibd0Tjt5WySCMqq',
    'vocab.txt': 'https://drive.google.com/uc?id=1I3QuOmeKpAISpukH9xtbldhqZ4xcyUgp',
}

# Path to save the model locally
model_save_path = "./bert_model/news_classifier_model"

# Create the directory if it doesn't exist
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Download the model files if they do not exist
for filename, link in files.items():
    file_path = os.path.join(model_save_path, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        gdown.download(link, file_path, quiet=False)
    else:
        print(f"{filename} already exists.")

# Set the model directory
model_path = "./bert_model/news_classifier_model"


# Define the label-to-topic mapping
label_map = {
    0: 'Politics',
    1: 'Sports',
    2: 'Technology',
    3: 'Entertainment',
    4: 'Business',
}

# Load pretrained model
if os.path.exists(model_path):
    print("Loading pretrained model...")
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
else:
    print("Error: No pretrained model found. Please provide a pretrained model first.")
    model = None

# Move model to CPU
device = torch.device("cpu")
if model:
    model.to(device)

# Prediction function
def predict_topic(text):
    if model is None:
        return "Error: No model loaded."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_topic = label_map.get(predicted_class, "Unknown Topic")
    return predicted_topic

# Home route (index page)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Read the incoming JSON data
        data = request.get_json()  # Now we use request.get_json() instead of form['text']
        text = data.get('text', '')  # Get 'text' from the received JSON

        predicted_topic = predict_topic(text)
        return {'topic': predicted_topic}  # Return a JSON response

if __name__ == '__main__':
    app.run(debug=True)
