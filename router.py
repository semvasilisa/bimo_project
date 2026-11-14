import os
import re
import requests

import time
from ollama import Client

from httpx import Timeout

ROUTER_FAILURE_SENTENCE = "Sorry, I can't reach my main assistant right now. Please try again shortly."
FILLER_START = re.compile(r"^(sure|certainly|of course|absolutely|okay)\b[!,.:]?\s*", re.IGNORECASE)

bimo_token = os.getenv("BIMO_TOKEN")

if not bimo_token:
    raise RuntimeError("BIMO_TOKEN is not set!")

headers = {
    "x-bimo-token": bimo_token
}


def get_url_list(name):
    """Read a comma-separated list of addresses from an environment variable."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set!")
    return [u.strip() for u in value.split(",") if u.strip()]


# Addresses of the server on the laptop: home first, hotspot second
server_urls = get_url_list("BIMO_SERVER_URLS")

# One Ollama client per laptop address
laptop_clients = [
    Client(
        host=u,
        timeout=Timeout(10.0, connect=2.0)  # 2s max connect, 10s max wait for data
    )
    for u in get_url_list("OLLAMA_LAPTOP_URLS")
]

# ---------------------- METHODS DEFINITION ----------------------

def needs_realtime(prompt):
    trigger_words = ["today", "now", "weather", "news", "current", "latest", "price", "tomorrow", "yesterday"]

    words = re.findall(r"\b[a-z]+\b", prompt.lower())
    
    return any(word in words for word in trigger_words)

def ask_cloud(prompt):
    last_error = None

    for url in list(server_urls):
        try:
            response = requests.post(
                url,
                json={"prompt": prompt},
                headers=headers,
                timeout=(2, 10)  # 2s max connect timeout, 10s max read timeout
            )
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict) or "answer" not in data:
                raise ValueError("Invalid cloud response: missing 'answer'")

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"[CLOUD] {url} failed: {e}")
            last_error = e
            continue

        server_urls.remove(url)
        server_urls.insert(0, url)
        return data["answer"]

    raise last_error

def laptop_chat(**kwargs):
    last_error = None

    for laptop_client in list(laptop_clients):
        try:
            response = laptop_client.chat(**kwargs)
        except Exception as e:
            print(f"[OLLAMA] {laptop_client._client.base_url} failed: {e}")
            last_error = e
            continue

        laptop_clients.remove(laptop_client)
        laptop_clients.insert(0, laptop_client)
        return response

    raise last_error

def ask_laptop(prompt, system_prompt=None):
    response = laptop_chat(
            model = 'gemma:2b',
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt or (
                        'You are BIMO, a cheerful voice assistant. '
                        'Reply in plain, natural spoken sentences. '
                        'Do not use lists, markdown, headings, or emojis. '
                        'Answer in no more than 40 words. '
                        'Use one, two or three short, grammatically complete sentences. '
                        'Use simple words and complete your final thought. '
                        'Never end with an unfinished phrase or a dangling word. '
                        'Before finishing, check that your final sentence makes grammatical sense. '
                        'Do not volunteer information about yourself, your identity, '
                        'your purpose, your creators, or any organization associated '
                        'with you unless the user explicitly asks. '
                        "Start directly with the answer. Never begin with 'Sure' or 'Here's'."
                    )
                }, 
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={'num_predict': 200}
        )
    answer = response['message']['content'].strip()

    done_reason = response.get('done_reason', 'unknown')

    print(f"[OLLAMA] Done reason: {done_reason}")

    if done_reason == 'length':
        print("[OLLAMA] Response reached token limit. Removing unfinished sentence.")

        # Keep only complete sentences.
        sentences = re.findall(r'.*?[.!?](?=\s|$)', answer)

        answer = " ".join(sentences).strip()

    if not answer:
        return "Sorry, I couldn't complete my answer. Please try again."

    answer = FILLER_START.sub("", answer)
    answer = answer[:1].upper() + answer[1:]
    return limit_sentences(answer)

def limit_sentences(text, max_sentences=3, max_words=40):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    selected = []
    total_words = 0

    for sentence in sentences[:max_sentences]:
        word_count = len(sentence.split())

        # Always keep the first sentence, even if it exceeds the budget.
        if not selected:
            selected.append(sentence)
            total_words += word_count
            continue

        # Keep subsequent sentences only if they fit within the budget.
        if total_words + word_count <= max_words:
            selected.append(sentence)
            total_words += word_count
        else:
            break

    return " ".join(selected)


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
                return ROUTER_FAILURE_SENTENCE

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
        return ROUTER_FAILURE_SENTENCE

if __name__ == "__main__":
    while True:
        prompt = input("You: ")

        if prompt.lower() == "quit":
            break

        answer = answer_question(prompt)
        print("BIMO:", answer)