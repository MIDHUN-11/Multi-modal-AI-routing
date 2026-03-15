from classifier.predict import classify_prompt
from models.ollama_client import call_model


def route_prompt(prompt: str):

    task = classify_prompt(prompt)

    if task == "coding":
        model = "llama3"

    elif task == "math":
        model = "phi3"

    else:
        model = "mistral"

    return call_model(prompt, model)