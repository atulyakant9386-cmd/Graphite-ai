
from fastapi import FastAPI
import os
import urllib.request
from llama_cpp import Llama

app = FastAPI()

# Model path and Hugging Face download URL
model_path = "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"
model_url = "https://huggingface.co/atulyakant9/tinyllama-model/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf"

# Download the model if it does not exist locally
if not os.path.exists(model_path):
    print("Downloading model from Hugging Face...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

# Load the LLaMA model
model = Llama(model_path=model_path)

@app.get("/")
async def root():
    return {"message": "Graphite AI chatbot server running."}

@app.get("/generate")
async def generate_response(prompt: str):
    # Generate text from prompt with model
    output = model(prompt=prompt, max_tokens=50)
    return {"response": output['choices'][0]['text']}
