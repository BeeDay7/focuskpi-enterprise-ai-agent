import os
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
model = os.getenv("LLM_MODEL")
base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")

print("Model:", model)
print("Base URL:", base_url)
print("Key loaded:", bool(api_key))

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly: OpenAI connection works."
        }
    ],
    "temperature": 0.1,
}

response = httpx.post(
    f"{base_url}/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=60,
)

print("HTTP:", response.status_code)
print(response.text[:3000])