from typing import Dict, List, Optional, Union

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from atia.config import settings

# Initialize the OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_completion(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    temperature: float = settings.openai_temperature,
    max_tokens: int = settings.openai_max_tokens,
    model: str = settings.openai_model,
) -> str:
    """
    Get a completion from the OpenAI API with retry logic.

    Args:
        prompt: The user prompt
        system_message: The system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        model: The OpenAI model to use

    Returns:
        The generated text
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def get_json_completion(
    prompt: str,
    system_message: str = "You are a helpful AI assistant.",
    temperature: float = settings.openai_temperature,
    max_tokens: int = settings.openai_max_tokens,
    model: str = settings.openai_model,
) -> Dict:
    """
    Get a JSON completion from the OpenAI API with retry logic.

    Args:
        prompt: The user prompt
        system_message: The system message
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        model: The OpenAI model to use

    Returns:
        The generated JSON as a Python dictionary
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        # Return the JSON content
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling OpenAI API for JSON completion: {e}")
        raise


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Get an embedding from the OpenAI API with retry logic.

    Args:
        text: The text to embed
        model: The embedding model to use

    Returns:
        The embedding as a list of floats
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise