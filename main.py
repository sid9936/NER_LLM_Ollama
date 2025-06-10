from fastapi import FastAPI, Request
from pydantic import BaseModel
import spacy
import requests

# Initialize spaCy model
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# Define input format
class Prompt(BaseModel):
    text: str

# API endpoint
@app.post("/process/")
def process_prompt(prompt: Prompt):
    text = prompt.text

    # Named Entity Recognition using spaCy
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("Detected Entities:", entities)

    # Call local LLM using Ollama REST API
    try:
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": text}
        )
        llm_output = ollama_response.json()["response"]
    except Exception as e:
        print("LLM Error:", str(e))
        llm_output = "Error contacting LLM."

    print("LLM Response:", llm_output)

    return {
        "entities": entities,
        "llm_response": llm_output
    }
