import json
from typing import Type

from crewai import LLM, Agent, Crew, Task
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from game_mcp.game_client import Defuser as DefuserClient, Expert as ExpertClient
os.environ.setdefault("BOMB_SERVER_URL", "http://localhost:8080")


try:
    with open("gemini_API_key.json", "r") as f:
        data = json.load(f)
        gemini_key = data.get("api_key")
except FileNotFoundError:
    gemini_key = os.getenv("GEMINI_API_KEY")

# --- 1. Define Tools ---
class DefuserToolInput(BaseModel):
    prompt: str = Field(..., description="Command or query to the bomb server")
    class Config:
        extra = 'allow'

class DefuserTool(BaseTool):
    name: str = "DefuserTool"
    description: str  = "Send commands (e.g., state, cut wire) to the bomb server and receive responses"
    args_schema: Type[BaseModel] = DefuserToolInput

    async def _run(self, prompt: str) -> str:
        client = DefuserClient()
        await client.connect_to_server(os.getenv("BOMB_SERVER_URL"))
        result = await client.run(prompt)
        await client.cleanup()
        return result

class ExpertToolInput(BaseModel):
    query: str = Field(..., description="Request manual instructions")
    class Config:
        extra = 'allow'

class ExpertTool(BaseTool):
    name: str = "ExpertTool"
    description: str = "Fetch bomb manual text for a given query"
    args_schema: Type[BaseModel] = ExpertToolInput

    async def _run(self, query: str) -> str:
        client = ExpertClient()
        await client.connect_to_server(os.getenv("BOMB_SERVER_URL"))
        result = await client.run()
        await client.cleanup()
        return result

# --- 2. Initialize Gemini LLM ---

llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=gemini_key,
    temperature=0.4,
    top_p=0.9,
    top_k=50
)

# --- 3. Define Agents ---
defuser_agent = Agent(
    role="Bomb Defuser",
    goal="Observe bomb state and execute disarm commands.",
    backstory="You see the bomb but do not have the manual; you must ask the Expert for instructions.",
    llm=llm,
    tools=[DefuserTool()],
    verbose=True
)

expert_agent = Agent(
    role="Bomb Expert",
    goal="Provide clear, step-by-step instructions from the manual.",
    backstory="You have the bomb manual and guide the Defuser.",
    llm=llm,
    tools=[ExpertTool()],
    verbose=True
)

# --- 4. Crew and Tasks ---
# Task: Defuser observes and asks Expert
def defuser_to_expert(context: dict) -> str:
    state = context['last_defuser_observation']
    prompt = f"### Observation\n{state}\n### Request\nWhat should I do next?"
    return prompt

# Task: Expert advises
def expert_to_defuser(context: dict) -> str:
    obs = context['last_defuser_observation']
    manual = context['manual_text']
    prompt = f"### Manual Excerpt\n{manual}\n### Observation\n{obs}\n### Instruction"
    return prompt

crew = Crew(
    agents=[defuser_agent, expert_agent]
)

# --- 5. Game Loop ---

_defuser_tool = DefuserTool()
_expert_tool = ExpertTool()

async def main_loop():
    context = {}
    while True:
        # 1) Defuser: get raw state from the server via DefuserTool
        raw_state = await _defuser_tool._run("state")
        print("[Raw State]:", raw_state)
        if any(k in raw_state.lower() for k in ["disarmed", "exploded"]):
            print("Game Over:", raw_state)
            break

        context['last_defuser_observation'] = raw_state

        # 2) Expert: fetch manual via ExpertTool
        manual_text = await _expert_tool._run(raw_state)
        context['manual_text'] = manual_text

        # 3) Expert: generate advice using its LLM
        advice_prompt = expert_to_defuser(context)
        advice = await expert_agent.chat(advice_prompt)
        print("[Expert Advice]:", advice)
        context['last_expert_advice'] = advice

        # 4) Defuser: decide action using its LLM
        action_prompt = f"""### Expert Advice
            {advice}
            ### Action"""
        action = await defuser_agent.chat(action_prompt)
        print("[Defuser Action]:", action)

        # 5) Defuser: send action to server
        result = await _defuser_tool._run(action)
        print("[Server Response]:", result)
        if any(k in result.lower() for k in ["disarmed", "exploded"]):
            print("Game Over:", result)
            break

if __name__ == "__main__":
    import asyncio, dotenv
    dotenv.load_dotenv()
    asyncio.run(main_loop())
