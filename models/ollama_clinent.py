import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def call_model(prompt, model):

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    return response.json()