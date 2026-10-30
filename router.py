import os
import re
import ollama
import requests

import time
from ollama import Client

bimo_token = os.getenv("BIMO_TOKEN")
url_address = "http://localhost:8000/ask"

if not bimo_token:
    raise RuntimeError("BIMO_TOKEN is not set!")

headers = {
    "x-bimo-token": bimo_token
}

# defining the Ollama client for Gemma
OLLAMA_LAPTOP_URL = os.getenv("OLLAMA_LAPTOP_URL")

if not OLLAMA_LAPTOP_URL:
    raise RuntimeError(
        "OLLAMA_LAPTOP_URL environment variable is not set"
    )


# Create an Ollama client
client = Client(
    host=OLLAMA_LAPTOP_URL,
    timeout=20
)

# ---------------------- METHODS DEFINITION ----------------------

def needs_realtime(prompt):
    trigger_words = ["today", "now", "weather", "news", "current", "latest", "price", "tomorrow", "yesterday"]

    words = re.findall(r"\b[a-z]+\b", prompt.lower())
    
    return any(word in words for word in trigger_words)

def ask_cloud(prompt):
    # send this prompt to my server and give me its response
    response = requests.post(
        url_address,
        json={"prompt": prompt},
        headers=headers,
        timeout=8
    )
    
    response.raise_for_status()
    data = response.json()

    # check if the 'answer' key is present in the response
    if not isinstance(data, dict) or "answer" not in data:
        raise ValueError("Invalid cloud response: missing 'answer'")

    return data["answer"]

def ask_local(prompt, system_prompt=None):   
    # send prompt to gemma 
    response = ollama.chat(
        model = 'qwen2.5:1.5b',
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
    answer = response['message']['content'].strip()
    return answer

def ask_laptop(prompt, system_prompt=None):
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
    answer = response['message']['content'].strip()
    return answer


def answer_question(prompt):
    if needs_realtime(prompt):
        try:
            start_time = time.perf_counter()
            print(f"Request started at: {start_time:.3f} seconds")
            
            answer = ask_cloud(prompt)
            end_time = time.perf_counter()

            print("[BIMO AI]: Cloud response")

            print(f"Response received at: {end_time:.3f} seconds")
            print(f"Total request time: {end_time - start_time:.2f} seconds")
            return answer

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Cloud failed: {e}")
            print("Trying local Gemma...")

            try:
                start_time = time.perf_counter()

                answer = ask_local(prompt, system_prompt="You have no internet access. Do not guess live information. Admit when you cannot check.")

                end_time = time.perf_counter()
                print("[BIMO AI]: Local fallback")
                print(f"Response received at: {end_time:.3f} seconds")
                print(f"Total request time: {end_time - start_time:.2f} seconds")
                return answer
            except Exception as local_error:
                return f"Cloud and local answers both failed: {local_error}"

    try:
        start_time = time.perf_counter()
        print(f"Request started at: {start_time:.3f} seconds")
                    
        answer = ask_laptop(prompt)
        end_time = time.perf_counter()

        print("[BIMO AI]: Laptop response")
        print(f"Response received at: {end_time:.3f} seconds")
        print(f"Total request time: {end_time - start_time:.2f} seconds")     
        return answer

    except Exception as e:
        print(f"Laptop ollama failed: {e}")
        print("Trying local model on pi...")

        try:
            start_time = time.perf_counter()
            answer = ask_local(
                prompt,
                system_prompt="You have no internet access. Do not guess live information. Admit when you cannot check."
            )
            end_time = time.perf_counter()
            print("[BIMO AI]: Local fallback")
            print(f"Response received at: {end_time:.3f} seconds")
            print(f"Total request time: {end_time - start_time:.2f} seconds")     
            return answer

        except Exception as local_error:
            return f"Laptop and Pi answers both failed: {local_error}"

if __name__ == "__main__":
    while True:
        prompt = input("You: ")

        if prompt.lower() == "quit":
            break

        answer = answer_question(prompt)
        print("BIMO:", answer)