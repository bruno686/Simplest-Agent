from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import json
import requests
from langchain_community.tools import DuckDuckGoSearchRun

key = "sk-4ba79f849b3941dba380fa0d3641055d"
short_mem = []  # Keep last 2 dialogues
long_mem = []   # Store all vectors
encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

# Tools we want to call
tools = {
    "calculator": lambda expr: eval(expr),  # Simple eval for mathematical expressions
    "search": lambda query: search(query),  # Call real search function below
}

def build_prompt(q):
    """
    Build a prompt that explicitly asks the model to respond with tool calls in JSON format.
    """
    prompt = "Recent chat:\n" + "\n".join([f"Q:{q}\nA:{a}" for q,a in short_mem[-3:]])
    if long_mem:
        q_vec = encoder.encode(q)
        scores = [np.dot(q_vec, m) for m in long_mem]
        best_idx = np.argmax(scores)
        prompt += f"\nRelated memory:\n{long_mem[best_idx]}"
    
    # Tell the model to reason step by step and provide tool calls in JSON format.
    prompt += """
    Please reason step by step. 
    1. The "thinking" phase: You think carefully about the user's problem
    2. Tool Invocation Phase: Select the tool that can be invoked and output the parameters required by the corresponding tool
    If you think a tool (like 'calculator' or 'search') is needed, respond with a JSON object:
    {
        "tool": "<tool_name>",
        "params": "<parameters>"
    }
    <tool_name> only has calculator and search. You don't generate anything else
    For example, if you need to calculate, respond with something like:
    {"tool": "calculator", "params": "3 + 4"}
    """
    return prompt + f"\nNew question: {q}"

def get_api_response(prompt):
    """
    Get the response from the model and return its content.
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def update_memories(q, resp):
    """
    Update the memory with the question and response.
    """
    short_mem.append((q, resp))
    long_mem.append(encoder.encode(f"Q:{q} A:{resp}"))
    if len(short_mem) > 3: 
        short_mem.pop(0)

def extract_tool_request(response):
    """
    Extract and return tool request from the model's response in JSON format.
    """
    try:
        return json.loads(response)  # Expecting a JSON tool request
    except json.JSONDecodeError:
        return None  # No tool request found

def call_tool(tool_name, params):
    """
    Call the appropriate tool based on the tool name and parameters.
    """
    if tool_name in tools:
        # Call the tool and get the result
        return tools[tool_name](params)
    return None

def search(query):
    """
    Simulating a search functionality. Replace with actual search API call as needed.
    """
    # Use DuckDuckGo search API for simulation (this can be replaced with Google or another search engine)
    search = DuckDuckGoSearchRun()
    result = search.invoke(query)
    return result

def agent(q):
    """
    Main agent function to process the input and either respond directly or call tools.
    """
    prompt = build_prompt(q)
    resp = get_api_response(prompt)
    print(f"LLM response: {resp}")  # Debugging step to see what the model returns

    tool_request = extract_tool_request(resp)

    if tool_request:
        tool_name = tool_request.get("tool")
        params = tool_request.get("params")
        if tool_name and params:
            tool_result = call_tool(tool_name, params)
            # return f"Tool result: {tool_result}"
            resp = get_api_response(f"Organize the tool results: {tool_result} according to the question: {q}.")

    # If no tool was requested, just return the model's response
    update_memories(q, resp)
    return resp

# Test run
while True:
    q = input("You: ")
    if q == "quit": break
    print("Bot:", agent(q))
