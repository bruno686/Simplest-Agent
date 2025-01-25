import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

key = "sk-4ba79f849b3941dba380fa0d3641055d"

class DeepSeekAgent:
    def __init__(self, name, system_prompt=None, short_mem_size=3):
        self.name = name
        self.short_mem_size = short_mem_size  # Number of recent dialogues to keep (hyperparameter)
        self.short_mem = []  # Keep the most recent short_mem_size dialogues
        self.long_mem = []   # Store memory vectors and texts
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        self.system_prompt = system_prompt or "You are a helpful assistant"

    def build_prompt(self, conversation_history):
        # Construct recent dialogue context
        recent_dialogue = "\n".join(
            [f"{msg['speaker']}: {msg['content']}" 
             for msg in conversation_history[-self.short_mem_size:]]
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
            
        """
        Return Example:
        You represent Florida, a key swing state with a large elderly and Latino population. You need to respond to the views of other states and analyze the U.S. election from Florida's perspective.
            [Dialogue History]
            Moderator: The 2024 U.S. election is approaching. Which party do you think will dominate in swing states? Please discuss Trump and the Democrats.
            Pennsylvania: Blue-collar workers in our state tend to support the Democrats because of their economic policies.
            Michigan: Union voters also strongly support the Democrats, especially on labor rights.

            [Relevant Memory]
            Q: What issues are most important to elderly voters in Florida?
            A: Elderly voters are most concerned about healthcare and social security policies, and the Republican stance on these issues is more popular.

            Please respond to Michigan and add your perspective...
        """
        return (
            f"{self.system_prompt}\n"
            f"[Dialogue History]\n{recent_dialogue}"
            f"{memory_section}\n"
            f"Please respond to {conversation_history[-1]['speaker']} and add your perspective..."
        )

    def respond(self, conversation_history):
        prompt = self.build_prompt(conversation_history)
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=150
        ).choices[0].message.content
        
        # Update memory
        self.update_memory(conversation_history[-1]["content"], response)
        return response

    def update_memory(self, last_question, response):
        # Maintain short-term memory
        self.short_mem.append((last_question, response))
        if len(self.short_mem) > self.short_mem_size:
            self.short_mem.pop(0)
            
        # Store long-term memory
        memory_text = f"Q: {last_question}\nA: {response}"
        self.long_mem.append({
            "text": memory_text,
            "vector": self.encoder.encode(memory_text)
        })

def multi_agent_chat(agents, rounds=5, initial_prompt="Hello!"):
    history = [{"speaker": "Moderator", "content": initial_prompt}]
    
    for i in range(rounds):
        current_agent = agents[i % len(agents)]
        print(f"\nCurrent Speaker: {current_agent.name}")
        response = current_agent.respond(history)
        
        history.append({
            "speaker": current_agent.name,
            "content": response
        })
        
        print(f"{current_agent.name}: {response}")
        print("-"*50)

if __name__ == "__main__":
    # Initialize three swing state agents and set short-term memory size
    florida = DeepSeekAgent("Florida", 
        "You represent Florida, a key swing state with a large elderly and Latino population. You need to respond to the views of other states and analyze the U.S. election from Florida's perspective.",
        short_mem_size=2)  # Florida remembers the last 2 dialogues
    
    pennsylvania = DeepSeekAgent("Pennsylvania", 
        "You represent Pennsylvania, an industrial state where blue-collar workers and suburban voters are crucial to the election outcome. You need to respond to the views of other states and analyze the U.S. election from Pennsylvania's perspective.",
        short_mem_size=3)  # Pennsylvania remembers the last 3 dialogues
    
    michigan = DeepSeekAgent("Michigan", 
        "You represent Michigan, a state known for its automotive industry. Union and minority voters have a significant impact on the election. You need to respond to the views of other states and analyze the U.S. election from Michigan's perspective.",
        short_mem_size=4)  # Michigan remembers the last 4 dialogues
    
    # Conduct 5 rounds of dialogue
    multi_agent_chat(
        agents=[florida, pennsylvania, michigan],
        rounds=5,
        initial_prompt="The 2024 U.S. election is approaching. Which party do you think will dominate in swing states? Please discuss Trump and the Democrats. Simulate the election."
    )