import json
import tiktoken

def safe_json_loads(text: str):
    """Charge du texte JSON en dict, tolère l’absence d’accolades."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not text.strip().startswith("{"):
            text = "{\n" + text.strip().rstrip(",") + "\n}"
            return json.loads(text)
        raise

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_call_cost(prompt_parts: dict, model: str = "gpt-4o"):
    """
    prompt_parts can be either:
      - dict { "name_part": "text" }
      - message list [{ "role": "...", "content": "..."}, ...]
    
    Retrurns the number of input tokens.
    """
    input_tokens = 0

    if isinstance(prompt_parts, dict):
        # Dict case
        for name, text in prompt_parts.items():
            tokens = count_tokens(str(text), model)
            #print(tokens)
            input_tokens += tokens

    elif isinstance(prompt_parts, list):
        # case of message list -> OpenAI chat format (role + content)
        for i, message in enumerate(prompt_parts, start=1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            tokens = count_tokens(str(content), model)
            #print(tokens)
            # print(f"[{role}] → {tokens} tokens")
            input_tokens += tokens

    else:
        raise TypeError("prompt_parts must be dict or list")

    # print(f"\nTotal Tokens : {input_tokens}")
    return input_tokens