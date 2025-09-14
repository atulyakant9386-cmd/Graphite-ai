from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# Serve index.html from project root
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Model file info
model_path = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
model_url = "https://huggingface.co/atulyakant9/tinyllama-model/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"

# Download model if missing
if not os.path.exists(model_path):
    print("Downloading model from Hugging Face...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

# Load LLaMA model
model = Llama(model_path=model_path)

@app.get("/generate")
async def generate_response(prompt: str):
    output = model(prompt=prompt, max_tokens=50)
    return {"response": output['choices'][0]['text']}

