import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests
from urllib.parse import urlparse
from sklearn.metrics import precision_recall_curve
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "saved_fake_news_model.pt"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
print(f"✅ Using device: {device}")

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_image(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image_path = "downloaded_image.jpg"
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return image_path
    except:
        pass
    return None

def get_text_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :] 

def optimize_post_training_threshold(outputs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, outputs)
    if len(thresholds) == 0:
        return np.percentile(outputs, 75)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_threshold = thresholds[f1_scores.argmax()]
    return max(best_threshold, np.percentile(outputs, 75))

def apply_temperature_scaling(output, temp=1.5):
    return torch.sigmoid(output / temp).item()

def verify_source(news_source):
    trusted_sources = ["BBC", "Reuters", "NY Times"]
    return 1 if any(src in news_source for src in trusted_sources) else 0

def weighted_voting(prob, news_source):
    source_weight = 0.05
    return prob * (1 - source_weight) + verify_source(news_source) * source_weight

class MultiModalModel(nn.Module):
    def __init__(self, text_dim=128):
        super(MultiModalModel, self).__init__()
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.image_fc = models.resnet18(weights="IMAGENET1K_V1")
        self.image_fc.fc = nn.Linear(self.image_fc.fc.in_features, 128)
        self.fc = nn.Linear(1, 1)

    def forward(self, text_feat, image):
        text_out = self.text_fc(text_feat)
        image_out = self.image_fc(image)
        cos_sim = torch.nn.functional.cosine_similarity(text_out, image_out, dim=1).unsqueeze(1)
        return self.fc(cos_sim)

model = MultiModalModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

validation_outputs = [0.6, 0.7, 0.3, 0.5, 0.8]
validation_labels = [1, 1, 0, 0, 1]
global_threshold = optimize_post_training_threshold(validation_outputs, validation_labels)

def predict_fake_news_text(text):
    with torch.no_grad():
        text_feat = get_text_embedding(text)
        text_feat_reduced = nn.Linear(768, 128).to(device)(text_feat)
        dummy_image = torch.zeros((1, 3, 128, 128)).to(device)
        output = model(text_feat_reduced, dummy_image)
        prob = torch.sigmoid(output).item()
        score = weighted_voting(prob, text)
        print(f"🍭 Prediction: {'FAKE NEWS' if score > global_threshold else 'REAL NEWS'} (score: {score:.2f})")

def predict_fake_news_image(image_input):
    with torch.no_grad():
        image_path = download_image(image_input) if is_valid_url(image_input) else image_input
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        dummy_text = "This is a placeholder text."
        text_feat = get_text_embedding(dummy_text)
        text_feat_reduced = nn.Linear(768, 128).to(device)(text_feat)
        output = model(text_feat_reduced, image_tensor)
        prob = torch.sigmoid(output).item()
        print(f"🍭 Prediction: {'FAKE NEWS' if prob > global_threshold else 'REAL NEWS'} (score: {prob:.2f})")

while True:
    user_input = input("\n🔎 Enter 'text' for news classification, 'image' for image verification, or 'exit' to quit: ").strip().lower()
    if user_input == "text":
        text = input("📝 Enter a news headline: ")
        predict_fake_news_text(text)
    elif user_input == "image":
        image_url = input("📸 Enter an image URL or local path: ")
        predict_fake_news_image(image_url)
    elif user_input == "exit":
        print("🚀 Exiting. Have a great day!")
        break
    else:
        print("❌ Invalid input. Please enter 'text', 'image', or 'exit'.")
