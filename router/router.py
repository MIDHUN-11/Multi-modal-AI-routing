from classifier.predict import classify_prompt
from models.ollama_client import call_model

routing_table = {
    "coding": {
        "primary": "llama3",
        "fallback": "mistral"
    },
    "math": {
        "primary": "phi3",
        "fallback": "llama3"
    },
    "general": {
        "primary": "mistral",
        "fallback": "llama3"
    }
}


def route_prompt(prompt: str):

    task = classify_prompt(prompt)

    primary_model = routing_table[task]["primary"]
    fallback_model = routing_table[task]["fallback"]

    try:
        print(f"Using primary model: {primary_model}")

        return call_model(prompt, primary_model)

    except Exception as e:

        print(f"Primary model failed: {e}")
        print(f"Switching to fallback: {fallback_model}")

        return call_model(prompt, fallback_model)