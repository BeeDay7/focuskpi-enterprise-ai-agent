import os
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if not api_key:
    print("ERROR: LLM_API_KEY was not found in .env")
    raise SystemExit(1)

response = httpx.get(
    "https://api.openai.com/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}",
    },
    timeout=30,
)

print("HTTP:", response.status_code)

if response.status_code != 200:
    print("ERROR:")
    print(response.text[:1000])
    raise SystemExit(1)

models = response.json().get("data", [])

print("\nAVAILABLE MODELS:\n")

for model in sorted(models, key=lambda item: item.get("id", "")):
    print(model.get("id", ""))