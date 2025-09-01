from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from llama_cpp import Llama
import os

app = FastAPI()

# Serve index.html using absolute path
@app.get("/")
async def read_index():
    return FileResponse(r"C:\Users\HP WORLD\OneDrive\Documents\ch-1\ch1 practice set\index.html")

# Load TinyLlama 1.1B Q3_K_S GGUF model (update path if needed)
llm = Llama(model_path=r"C:\Users\HP WORLD\OneDrive\Documents\ch-1\ch1 practice set\tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf", n_ctx=512)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        print("Received message:", request.message)
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": request.message}],
            max_tokens=100
        )
        reply = result['choices'][0]['message']['content']
        print("Model reply:", reply)
        return {"response": reply}
    except Exception as e:
        print("Model error:", e)
        return {"error": "AI backend error: " + str(e)}
