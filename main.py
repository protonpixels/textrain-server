from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Allow your Vue app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
print("Loading model...")
model_name = "primelaunch34v/textrain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


class SuggestionRequest(BaseModel):
    text: str
    n_suggestions: int = 5
    temperature: float = 0.8


class SuggestionResponse(BaseModel):
    suggestions: list
    probabilities: list


@app.get("/")
def root():
    return {"message": "Kalama Next Word API is running", "model": model_name}


@app.post("/suggest", response_model=SuggestionResponse)
def suggest(request: SuggestionRequest):
    """Get next word suggestions"""
    if not request.text.strip():
        return SuggestionResponse(
            suggestions=["The", "I", "We", "This", "It"],
            probabilities=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

    inputs = tokenizer(request.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :] / request.temperature
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, request.n_suggestions)

    suggestions = []
    probabilities = []

    for prob, idx in zip(top_probs, top_indices):
        word = tokenizer.decode([idx]).strip()
        if word and len(word) > 1:
            suggestions.append(word)
            probabilities.append(float(prob))

    return SuggestionResponse(
        suggestions=suggestions[:request.n_suggestions],
        probabilities=probabilities[:request.n_suggestions]
    )


# For local testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)