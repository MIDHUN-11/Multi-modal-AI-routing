from fastapi import FastAPI
import requests

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.get("/")
def home():
    return {"message": "AI Router Running"}

@app.post("/ai")
def ask_ai(prompt: str):

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    return response.json()