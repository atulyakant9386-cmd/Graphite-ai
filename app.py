from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# Enable CORS to allow frontend requests from any domain (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve your frontend index.html on the root path
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

model_path = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
model_url = "https://huggingface.co/atulyakant9/tinyllama-model/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"

# Download the model if it does not exist
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded.")

# Load the model
model = Llama(model_path=model_path)

# Backend API endpoint to generate AI response
@app.get("/generate")
async def generate_response(prompt: str):
    output = model(prompt=prompt, max_tokens=50)
    return {"response": output['choices'][0]['text']}
