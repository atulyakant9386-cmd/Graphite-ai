from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# Enable CORS: restrict in production to your domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend index.html on root path
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

model_path = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
model_url = "https://huggingface.co/atulyakant9/tinyllama-model/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"

# Download model only if not present
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded.")

# Load the model
try:
    model = Llama(model_path=model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None  # Handle failure gracefully

# API endpoint for generating AI response
@app.get("/generate")
async def generate(prompt: str):
    if model is None:
        return {"error": "Model not loaded, try again later."}
    try:
        output = model(prompt=prompt, max_tokens=50)
        return {"response": output['choices'][0]['text']}
    except Exception as e:
        return {"error": f"Generation error: {e}"}
