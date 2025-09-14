from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# Enable CORS for requests from any origin (adjust "allow_origins" for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend index.html at root
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Model file and URL
model_path = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
model_url = "https://huggingface.co/atulyakant9/tinyllama-model/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"

# Download model if not exist
if not os.path.exists(model_path):
    print("[*] Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("[*] Model downloaded.")

# Load llama model
try:
    model = Llama(model_path=model_path)
    print("[*] Model loaded successfully.")
except Exception as e:
    print(f"[!] Error loading model: {e}")
    model = None  # So we can detect backend not ready

@app.get("/generate")
async def generate_response(prompt: str):
    if model is None:
        return {"error": "Model not loaded or backend not ready. Please try again later."}
    try:
        output = model(prompt=prompt, max_tokens=50)
        return {"response": output['choices'][0]['text']}
    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}

# # Uncomment below to test backend integration quickly with dummy response:
# @app.get("/generate")
# async def generate_response(prompt: str):
#     return {"response": f"Echo backend says: {prompt}"}

