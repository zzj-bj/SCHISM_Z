import json


def safe_json_loads(text: str):
    """Charge du texte JSON en dict, tolère l’absence d’accolades."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not text.strip().startswith("{"):
            text = "{\n" + text.strip().rstrip(",") + "\n}"
            return json.loads(text)
        raise