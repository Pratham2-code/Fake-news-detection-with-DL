import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- 1. Preprocessing Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text_scratch(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

# --- 2. Data Loading & Cleaning (WELFake_Dataset) ---
print("Loading and cleaning WELFake_Dataset.csv...")
# WELFake has columns: title, text, label
data = pd.read_csv('/content/WELFake_Dataset.csv')

# Handle missing values which are common in this dataset
data = data.dropna(subset=['title', 'text'])

data['content'] = (data['title'] + " " + data['text']).apply(clean_text_scratch)
y_labels = data['label'].values

# --- 3. Feature Extraction ---
print("Vectorizing features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_processed = vectorizer.fit_transform(data['content']).toarray()

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_processed, y_labels, test_size=0.2, random_state=42)

# --- 4. PyTorch Dataset & Model ---
train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train_s, dtype=torch.float32).view(-1,1))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

class FakeNewsTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5000, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_torch = FakeNewsTorch().to(device)
opt = optim.Adam(model_torch.parameters(), lr=0.001)
crit = nn.BCELoss()

# --- 5. Training Loop ---
print(f"Training on {device}...")
for e in range(5):
    model_torch.train()
    curr_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model_torch(xb), yb)
        loss.backward()
        opt.step()
        curr_loss += loss.item()
    print(f"Epoch {e+1}/5 | Loss: {curr_loss/len(train_dl):.4f}")

# --- 6. Save ---
torch.save(model_torch.state_dict(), 'pytorch_model_welfake.pth')
with open('vectorizer_welfake.pkl', 'wb') as f: pickle.dump(vectorizer, f)
print("PyTorch artifacts for WELFake saved successfully.")

#-------- 7. Evaluation ---
model_torch.eval()
with torch.no_grad():
    # Convert predictions to a numpy array for sklearn metrics
    y_pred_probs = model_torch(torch.tensor(X_test_s, dtype=torch.float32).to(device)).cpu().numpy()
    y_pred = (y_pred_probs > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test_s, y_pred, target_names=['Fake', 'Real']))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test_s, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()