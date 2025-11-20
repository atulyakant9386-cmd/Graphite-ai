from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

llm = Llama(model_path=r"C:\Users\HP WORLD\OneDrive\Documents\ch-1\ch1 practice set\llama-2-7b-chat.Q4_0.gguf")


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": request.message}],
        max_tokens=100
    )
    reply = result['choices'][0]['message']['content']
    return {"response": reply}

