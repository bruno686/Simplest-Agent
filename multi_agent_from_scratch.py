import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

key = "sk-4ba79f849b3941dba380fa0d3641055d"

class DeepSeekAgent:
    def __init__(self, name, system_prompt=None):
        self.name = name
        self.short_mem = []  # Keep last 3 dialogues
        self.long_mem = []   # Store memory vectors and texts
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        self.system_prompt = system_prompt or "You are a helpful assistant"

    def build_prompt(self, conversation_history):
        # Construct recent dialogue context
        recent_dialogue = "\n".join(
            [f"{msg['speaker']}: {msg['content']}" 
             for msg in conversation_history[-3:]]
        )
        
        # Long-term memory retrieval
        current_topic = conversation_history[-1]["content"]
        if self.long_mem:
            current_vec = self.encoder.encode(current_topic)
            scores = [np.dot(current_vec, mem["vector"]) for mem in self.long_mem]
            best_match = self.long_mem[np.argmax(scores)]["text"]
            memory_section = f"\n[Relevant Memory]\n{best_match}"
        else:
            memory_section = ""
            
        return (
            f"{self.system_prompt}\n"
            f"[Dialogue History]\n{recent_dialogue}"
            f"{memory_section}\n"
            f"Please respond to {conversation_history[-1]['speaker']}..."
        )

    def respond(self, conversation_history):
        prompt = self.build_prompt(conversation_history)
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        ).choices[0].message.content
        
        # Update memory
        self.update_memory(conversation_history[-1]["content"], response)
        return response

    def update_memory(self, last_question, response):
        # Maintain short-term memory
        self.short_mem.append((last_question, response))
        if len(self.short_mem) > 3:
            self.short_mem.pop(0)
            
        # Store long-term memory
        memory_text = f"Q: {last_question}\nA: {response}"
        self.long_mem.append({
            "text": memory_text,
            "vector": self.encoder.encode(memory_text)
        })

def multi_agent_chat(agents, rounds=5, initial_prompt="Hello!"):
    history = [{"speaker": "User", "content": initial_prompt}]
    
    for _ in range(rounds):
        current_agent = agents[len(history) % len(agents)]
        response = current_agent.respond(history)
        
        history.append({
            "speaker": current_agent.name,
            "content": response
        })
        
        print(f"\n{current_agent.name}: {response}")
        print("-"*50)

if __name__ == "__main__":
    # Initialize two agents with different styles
    scientist = DeepSeekAgent("Dr.Smith", 
        "You are a research scientist. Respond with critical thinking and cite academic references.")
    
    philosopher = DeepSeekAgent("Philosopher_Jane", 
        "You are a philosopher. Analyze questions through the lens of existentialism and phenomenology.")
    
    # Conduct 5 rounds of dialogue
    multi_agent_chat(
        agents=[scientist, philosopher],
        rounds=5,
        initial_prompt="What is the meaning of consciousness?"
    )