import os
import json
import re
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI()

# Enable CORS for communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL ARCHITECTURE ---
class FakeNewsTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5000, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

model_dl = FakeNewsTorch()
vectorizer = None

try:
    if os.path.exists('pytorch_model_welfake.pth') and os.path.exists('vectorizer_welfake.pkl'):
        model_dl.load_state_dict(torch.load('pytorch_model_welfake.pth', map_location='cpu'))
        model_dl.eval()
        with open('vectorizer_welfake.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Local Model Loaded Successfully")
    else:
        print("! Local model files missing. Using Gemini only.")
except Exception as e:
    print(f"! Load Error: {e}")

GEMINI_API_KEY = "AIzaSyD_IwqmearAFknTLaUFUuUiAkrVbQ8O3fA"
client = genai.Client(api_key=GEMINI_API_KEY)

class PredictRequest(BaseModel):
    text: str
    engine: str

# --- ENDPOINTS ---
@app.get("/")
async def serve_index():
    path = os.path.join("frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "index.html not found in 'frontend' folder"}

@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        if req.engine == "dl" and vectorizer:
            vec = vectorizer.transform([req.text]).toarray()
            with torch.no_grad():
                prob = model_dl(torch.tensor(vec, dtype=torch.float32)).item()
            is_fake = prob > 0.5
            return {
                "prediction": "Fake News" if is_fake else "Real News",
                "confidence": round(prob if is_fake else (1-prob), 2),
                "reason": "Analyzed patterns using your local PyTorch model.",
                "model_used": "PyTorch ANN (Local)"
            }
        else:
            prompt = f"Analyze: {req.text[:800]}. Return ONLY JSON: {{\"prediction\": \"Real News\", \"confidence\": 0.95, \"reason\": \"text\"}}"
            resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            
            # Safety Check: Extract JSON and handle errors
            match = re.search(r'\{.*\}', resp.text, re.DOTALL)
            if not match:
                raise ValueError("Could not find JSON in Gemini response")
            
            data = json.loads(match.group())
            data["model_used"] = "Gemini 1.5 Flash (Cloud)"
            return data
    except Exception as e:
        print(f"Server Error: {e}")
        return {"prediction": "Error", "confidence": 0, "reason": str(e), "model_used": "System Recovery"}

# Static file serving
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    # RUNNING ON PORT 5000
    print("\n--- VerifyAI Neural Dashboard Online ---")
    print("Click here: http://127.0.0.1:5000\n")
    uvicorn.run(app, host="127.0.0.1", port=5000)