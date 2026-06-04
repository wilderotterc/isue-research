"""
M5 — LLM Caller
Submits the prompt to gpt-4o-mini at temperature=0.
Records model name, version string, and token usage.
"""

import os
from dotenv import load_dotenv
import openai

load_dotenv()

LLM_MODEL   = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS  = 400

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def call_llm(system_prompt: str, user_message: str) -> dict:
    """
    Call the LLM and return the response with metadata.

    Returns a dict containing:
        answer:           the generated text
        model_name:       model string (e.g. "gpt-4o-mini")
        model_version:    full version string from API response
        prompt_tokens:    tokens used in prompt
        completion_tokens: tokens generated
        total_tokens:     total tokens used
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
    )

    return {
        "answer":             response.choices[0].message.content,
        "model_name":         LLM_MODEL,
        "model_version":      response.model,
        "prompt_tokens":      response.usage.prompt_tokens,
        "completion_tokens":  response.usage.completion_tokens,
        "total_tokens":       response.usage.total_tokens,
    }

if __name__ == "__main__":
    result = call_llm(
        system_prompt="You are a helpful assistant answering questions about US student financial aid. Answer using ONLY the information in the context below. If the context does not contain enough information to answer the question, say so clearly.",
        user_message="What is the maximum Pell Grant for 2025-26?"
    )
    print(f"Model: {result['model_version']}")
    print(f"Tokens: {result['total_tokens']} ({result['prompt_tokens']} prompt + {result['completion_tokens']} completion)")
    print(f"Answer: {result['answer']}")