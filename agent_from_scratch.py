import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

key = "sk-4ba79f849b3941dba380fa0d3641055d"
short_mem = []  # Keep last 2 dialogues
long_mem = []   # Store all vectors
encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

def build_prompt(q):
    prompt = "Recent chat:\n" + "\n".join([f"Q:{q}\nA:{a}" for q,a in short_mem[-3:]])
    if long_mem:
        q_vec = encoder.encode(q)
        scores = [np.dot(q_vec, m) for m in long_mem]
        best_idx = np.argmax(scores)
        prompt += f"\nRelated memory:\n{long_mem[best_idx]}"
    return prompt + f"\nNew question: {q}"

def get_api_response(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def update_memories(q, resp):
    short_mem.append((q, resp))
    long_mem.append(encoder.encode(f"Q:{q} A:{resp}"))
    if len(short_mem) > 3: 
        short_mem.pop(0)

def agent(q):
    prompt = build_prompt(q)
    resp = get_api_response(prompt)
    update_memories(q, resp)
    return resp

# Test run
while True:
    q = input("You: ")
    if q == "quit": break
    print("Bot:", agent(q))