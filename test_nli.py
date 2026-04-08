from transformers import pipeline

print("Loading NLI model...")
nli_pipeline = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small", device="cpu")

tests = [
    {
        "name": "Entailment",
        "premise": "The boiling point of water is 100 degrees Celsius.",
        "hypothesis": "Water boils at 100 degrees Celsius."
    },
    {
        "name": "Contradiction",
        "premise": "The boiling point of water is 100 degrees Celsius.",
        "hypothesis": "Water boils at 50 degrees Celsius."
    },
    {
        "name": "Neutral",
        "premise": "The boiling point of water is 100 degrees Celsius.",
        "hypothesis": "The sky is blue."
    }
]

for t in tests:
    res = nli_pipeline({"text": t["premise"], "text_pair": t["hypothesis"]})
    print(f"{t['name']}: {res}")
