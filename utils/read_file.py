import json

def load_json(config: str) -> dict:
    with open(config, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data