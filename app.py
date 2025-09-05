from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# Load your tinyllama model once on startup
model = Llama(model_path="tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")

# Define the request body schema
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100  # optional with default

# Define the response schema
class GenerateResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        # Call tinyllama to generate text
        output = model.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )
        # Extract generated text
        generated = output.choices[0].text
        return GenerateResponse(generated_text=generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
