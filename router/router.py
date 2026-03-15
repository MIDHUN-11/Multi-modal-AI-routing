from classifier.predict import classify_prompt
from models.ollama_client import call_model
from metrics.latency import track_latency

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

    primary = routing_table[task]["primary"]
    fallback = routing_table[task]["fallback"]

    try:

        response, latency = track_latency(call_model, prompt, primary)

        print(f"task={task} model={primary} latency={latency:.2f}s")

        return response

    except Exception as e:

        print(f"Primary model failed: {e}")

        response, latency = track_latency(call_model, prompt, fallback)

        print(f"fallback model={fallback} latency={latency:.2f}s")

        return response