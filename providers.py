import requests
from google import genai
from openai import OpenAI

import config


def ask_openai(message):
    """Send one message to OpenAI and return the answer as text."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.responses.create(
        model=config.OPENAI_MODEL,
        instructions=config.SYSTEM_PROMPT,
        input=message,
    )
    return response.output_text


def ask_gemini(message):
    """Send one message to Google Gemini and return the answer as text."""
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=message,
        config={"system_instruction": config.SYSTEM_PROMPT},
    )
    return response.text


def ask_ollama(message):
    """Send one message to a model running locally in Ollama."""
    # Ollama has no API key: it is a plain HTTP server running on your own machine.
    try:
        response = requests.post(
            config.OLLAMA_BASE_URL + "/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                "stream": False,
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach Ollama at " + config.OLLAMA_BASE_URL + ". Is Ollama running?"
        )

    response.raise_for_status()
    return response.json()["message"]["content"]
