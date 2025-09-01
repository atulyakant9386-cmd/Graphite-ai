import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response from the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt from the response to only return the generated part
    response = response[len(prompt):].strip()
    return response

print("Chatbot: Hi there! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "bye"]:
        print("Chatbot: Goodbye!")
        break

    reply = generate_response(user_input)
    print("Chatbot:", reply)
