from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the old frontend index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Model download and loading
model_path ="Qwen2-1.5B-Instruct.IQ1_M.gguf"
model_url ="https://huggingface.co/atulyakant9/Qwen2-1.5B-Instruct.IQ1_M.gguf/blob/main/Qwen2-1.5B-Instruct.IQ1_M.gguf"

if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded successfully.")

try:
    model = Llama(model_path=model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# API endpoint to generate AI response
@app.get("/generate")
async def generate(prompt: str):
    if model is None:
        return {"error": "Model is not ready. Please try again later."}
    try:
        output = model(prompt=prompt, max_tokens=50)
        return {"response": output['choices'][0]['text']}
    except Exception as e:
        return {"error": f"Error generating response: {e}"}
