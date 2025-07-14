import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from transformers import AutoTokenizer
from PIL import Image

MODEL_PATH = "saved_fake_news_model.pt"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

fake_text_path = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\fake newss\News _dataset\Fake.csv"
true_text_path = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\fake newss\News _dataset\True.csv"

image_folders = {
    "train": r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\fake newss\merged_dataset\train",
    "test": r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\fake newss\merged_dataset\test",
    "valid": r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\fake newss\merged_dataset\valid"
}

def load_text_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path, dtype=str, low_memory=False)
    true_df = pd.read_csv(true_path, dtype=str, low_memory=False)
    fake_df["label"] = 1
    true_df["label"] = 0
    df = pd.concat([fake_df, true_df]).reset_index(drop=True)
    text_data = df["text"].dropna().tolist()
    labels = df["label"].tolist()
    return text_data, labels

def load_image_data(image_folders):
    image_paths, labels = [], []
    for label, category in enumerate(["real", "fake"]):
        category_path = os.path.join(image_folders["train"], category)
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                image_paths.append(os.path.join(category_path, file))
                labels.append(label)
    return image_paths, labels

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, image_paths, labels, transform=None):
        self.text_data = text_data
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")["input_ids"].squeeze(0).to(device)
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros((3, 128, 128))
        return tokenized.float(), image, label

class MultiModalModel(nn.Module):
    def __init__(self, text_dim=128):
        super(MultiModalModel, self).__init__()
        self.text_fc = nn.Sequential(nn.Linear(text_dim, 128), nn.LayerNorm(128), nn.ReLU())
        self.image_fc = models.resnet18(weights="IMAGENET1K_V1")
        self.image_fc.fc = nn.Linear(self.image_fc.fc.in_features, 128)
        self.fc = nn.Linear(1, 1)

    def forward(self, text, image):
        text_out = self.text_fc(text)
        image_out = self.image_fc(image)
        cos_sim = torch.nn.functional.cosine_similarity(text_out, image_out, dim=1).unsqueeze(1)
        return self.fc(cos_sim)

def train_model(model, train_loader, device, epochs=15):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for text, image, label in train_loader:
            text = text.to(device)
            image = image.to(device)
            label = label.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(text, image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model

text_data, text_labels = load_text_data(fake_text_path, true_text_path)
image_paths, image_labels = load_image_data(image_folders)

text_data = text_data[:len(image_paths)]
text_labels = text_labels[:len(image_paths)]
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = MultiModalDataset(text_data, image_paths, text_labels, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = MultiModalModel().to(device)
model = train_model(model, loader, device, epochs=15)

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
