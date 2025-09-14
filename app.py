from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# Allow CORS for frontend requests (adjust allowed origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use exact frontend URLs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at root path
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

model_path = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
model_url = "https://huggingface.co/atulyakant9/tinyllama-model/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"

# Download model if missing
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded successfully.")

# Load model
model = Llama(model_path=model_path)

# API endpoint generating the AI response
@app.get("/generate")
async def generate(prompt: str):
    output = model(prompt=prompt, max_tokens=50)
    return {"response": output['choices'][0]['text']}
