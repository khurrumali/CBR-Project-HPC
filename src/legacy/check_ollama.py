import os
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


def main() -> int:
    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    try:
        llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        reply = llm.invoke("Reply with exactly: OLLAMA_OK")
        text = getattr(reply, "content", str(reply))
        print(f"Model: {model}")
        print(f"Base URL: {base_url}")
        print(f"Reply: {text}")
        return 0
    except Exception as exc:
        print(f"Ollama check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
