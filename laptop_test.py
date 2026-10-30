
import os
import time
from ollama import Client


# Read the laptop Ollama address from an environment variable
OLLAMA_LAPTOP_URL = os.getenv("OLLAMA_LAPTOP_URL")

if not OLLAMA_LAPTOP_URL:
    raise RuntimeError(
        "OLLAMA_LAPTOP_URL environment variable is not set"
    )


# Create an Ollama client
client = Client(
    host=OLLAMA_LAPTOP_URL,
    timeout=60
)

def ask_laptop(prompt, system_prompt=None):  
    start_time = time.perf_counter()
    print(f"Request started at: {start_time:.3f} seconds") 
    
    response = client.chat(
        model = 'gemma:2b',
        messages = [
            {
                'role': 'system',
                'content': system_prompt or 'You are BIMO, a cheerful offline assistant. Keep responses under 2 sentences.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    end_time = time.perf_counter()
    print(f"Response received at: {end_time:.3f} seconds")
    print(f"Total request time: {end_time - start_time:.2f} seconds")

    answer = response['message']['content'].strip()
    return answer


if __name__ == "__main__":
    question = "How are you? Introduce yourself briefly."
    answer = ask_laptop(question)

    print("\n[BIMO AI]:", answer)
