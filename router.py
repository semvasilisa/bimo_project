import os
import re
import requests

import time
from ollama import Client

from httpx import Timeout
import re

bimo_token = os.getenv("BIMO_TOKEN")
url_address = "http://192.168.0.102:8000/ask"

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
    timeout=Timeout(10.0, connect=2.0) # 2s max connect timeout, 10s max waits to receive data after connecting
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
        timeout=(2, 10) # 2s max connect timeout, 10s max read timeout
    )
    
    response.raise_for_status()
    data = response.json()

    # check if the 'answer' key is present in the response
    if not isinstance(data, dict) or "answer" not in data:
        raise ValueError("Invalid cloud response: missing 'answer'")

    return data["answer"]

def ask_laptop(prompt, system_prompt=None):
    response = client.chat(
            model = 'gemma:2b',
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt or 'You are BIMO, a cheerful voice assistant. Reply in plain, natural spoken sentences. Do not use lists, markdown, headings, or emojis. Keep your answer to at most two short sentences. Do not volunteer information about yourself, your identity, your purpose, your creators, or any organization associated with you unless the user explicitly asks.'
                }, 
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={'num_predict': 100}
        )
    answer = response['message']['content'].strip()
    return limit_sentences(answer)

def limit_sentences(text, max_sentences=2):
    # split after sentence-ending punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return " ".join(sentences[:max_sentences])


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
            return limit_sentences(answer)

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Cloud failed: {e}")
            print("Trying local Gemma...")

            try:
                start_time = time.perf_counter()

                answer = ask_laptop(prompt, system_prompt="You cannot check live information. Do not guess. Admit when you cannot verify something. Reply in plain spoken sentences, without lists, markdown, or emojis. Keep your answer under two short sentences.")

                end_time = time.perf_counter()
                print("[BIMO AI]: Laptop fallback")
                print(f"Response received at: {end_time:.3f} seconds")
                print(f"Total request time: {end_time - start_time:.2f} seconds")
                return limit_sentences(answer)
            except Exception as local_error:
                return "Sorry, I can't reach my main assistant right now. Please try again shortly."

    try:
        start_time = time.perf_counter()
        print(f"Request started at: {start_time:.3f} seconds")
                    
        answer = ask_laptop(prompt)
        end_time = time.perf_counter()

        print("[BIMO AI]: Laptop response")
        print(f"Response received at: {end_time:.3f} seconds")
        print(f"Total request time: {end_time - start_time:.2f} seconds")     
        return limit_sentences(answer)

    except Exception as e:
        print(f"Laptop ollama failed: {e}")
        return "Sorry, I can't reach my main assistant right now. Please try again shortly."

if __name__ == "__main__":
    while True:
        prompt = input("You: ")

        if prompt.lower() == "quit":
            break

        answer = answer_question(prompt)
        print("BIMO:", answer)