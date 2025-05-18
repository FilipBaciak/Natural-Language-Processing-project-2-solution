import json
import os
from typing import Dict, Any

from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your custom tools
from crewai_bomb.tools import DefuserTool, ExpertTool

# Import game clients
from game_mcp.game_client import Defuser as DefuserClient, Expert as ExpertClient

# Default model name - can be overridden by environment variable or specific calls
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"  # Standard model name for Gemini Flash

async def create_bomb_defusal_crew(server_url: str, gemini_api_key: str) -> Dict[str, Any]:
    """
    Creates and configures the Bomb Defusal Crew with Defuser and Expert agents.

    Args:
        server_url: The URL of the bomb defusal game server.
        gemini_api_key: The API key for Google Gemini.

    Returns:
        A dictionary containing the configured "crew", "defuser_client", and "expert_client".
    """

    # 1. Initialize the LLM
    llm = LLM(
        model="gemini/gemini-2.0-flash",  # specifies Gemini 2.0 Flash model
        api_key=gemini_api_key,
        provider="gemini/"
    )
    print(f"Using LLM: {GEMINI_MODEL_NAME}")

    # 2. Create and connect game clients for the tools
    defuser_game_client = DefuserClient()
    expert_game_client = ExpertClient()

    print(f"Connecting DefuserClient to {server_url}...")
    await defuser_game_client.connect_to_server(server_url)
    print(f"Connecting ExpertClient to {server_url}...")
    await expert_game_client.connect_to_server(server_url)
    print("Game clients connected.")

    # 3. Instantiate tools with the connected clients
    defuser_action_tool = DefuserTool(defuser_game_client=defuser_game_client)
    expert_manual_tool = ExpertTool(expert_game_client=expert_game_client)
    print("Crew tools instantiated.")

    # 4. Define Agents
    defuser_agent = Agent(
        role="Defuser",
        goal=(
            "Interface with a live bomb. Accurately observe its state using available tools, "
            "clearly describe the situation to the Expert, and precisely execute the Expert's "
            "instructions to defuse the bomb. Your actions directly determine success or failure."
        ),
        backstory=(
            "You are a highly trained field operative, calm under immense pressure. "
            "You are at the bomb site with direct access to its components. "
            "You rely on a remote Expert for defusal knowledge. Clarity in your descriptions "
            "and precision in your actions are paramount. Failure is not an option."
        ),
        llm=llm,
        tools=[defuser_action_tool],
        verbose=True,
        allow_delegation=False,
    )

    expert_agent = Agent(
        role="Expert",
        goal=(
            "Receive and interpret bomb descriptions from the Defuser. "
            "Utilize the bomb manual to identify the correct procedures. "
            "Provide clear, concise, and actionable single-step instructions to the Defuser. "
            "Ensure instructions are unambiguous and lead towards successful defusal."
        ),
        backstory=(
            "You are a master of explosive ordnance disposal (EOD) procedures, possessing "
            "encyclopedic knowledge of various bomb mechanisms via manuals. "
            "You operate remotely, relying solely on the Defuser's reports. "
            "Your analytical skills and ability to provide precise guidance are critical "
            "to saving the day."
        ),
        llm=llm,
        tools=[expert_manual_tool],
        verbose=True,
        allow_delegation=False,
    )
    print("Crew agents defined.")

    # 5. Define Tasks
    task_observe_and_describe = Task(
        description=(
            "It's your turn, Defuser. Your primary goal is to assess the bomb and inform the Expert.\n"
            "1. Use your 'DefuserActionTool' with the command 'state' to get the current raw status of the bomb.\n"
            "2. Analyze this raw status. If it indicates the bomb is already 'disarmed' or 'exploded', "
            "report the game over condition and include the raw status.\n DO NOT REPORT THE GAME OVER CONDITION ELSEWHERE."
            "3. Otherwise, if you already have the current bomb state, describe the bomb's current state based on the raw status, focusing on details "
            "pertinent to defusal (e.g., wire colors, button labels, module types).\n"
            # "Use 'initial_query' for any starting context; ignore 'previous_round_outcome' if not provided."
        ),
        agent=defuser_agent,
        expected_output=(
            "Bomb state description string "
            "or a clear game over statement."
        ),
    )

    task_expert_instruct = Task(
        description=(
            "Expert, the Defuser has provided an update on the bomb's status.\n"
            "1. Analyze the Defuser's report. If it indicates disarmed/exploded, simply acknowledge.\n"
            "2. Otherwise, consult the defusal manual via 'ExpertManualTool'. Use the report as your query.\n"
            "3. Provide ONE clear, actionable instruction for the Defuser (e.g., 'cut wire 3')."
        ),
        agent=expert_agent,
        context=[task_observe_and_describe],
        expected_output=(
            "A single, precise command or game over acknowledgement."
        ),
    )

    task_defuser_act = Task(
        description=(
            "Defuser, you have an instruction from the Expert.\n"
            "1. If it's a game over acknowledgement, repeat it.\n"
            "2. Otherwise, execute the exact command using 'DefuserActionTool'.\n"
            "3. Output the raw server response after performing the action."
        ),
        agent=defuser_agent,
        context=[task_expert_instruct],
        expected_output=(
            "The raw string response from the game server or the game over acknowledgement."
        ),
    )
    print("Crew tasks defined.")

    # 6. Assemble the Crew
    bomb_defusal_crew = Crew(
        agents=[defuser_agent, expert_agent],
        tasks=[task_observe_and_describe, task_expert_instruct, task_defuser_act],
        process=Process.sequential,
        verbose=True
    )
    print("Bomb Defusal Crew assembled.")

    return {
        "crew": bomb_defusal_crew,
        "defuser_client": defuser_game_client,
        "expert_client": expert_game_client
    }

# Example test
async def _test_crew_creation():
    try:
        with open("gemini_API_key.json", "r") as f:
            data = json.load(f)
            gemini_key = data.get("api_key")
    except FileNotFoundError:
        gemini_key = os.getenv("GEMINI_API_KEY")
    server = "http://localhost:8080"
    if not gemini_key:
        print("GEMINI_API_KEY not set.")
        return

    clients_and_crew = await create_bomb_defusal_crew(server, gemini_key)
    crew = clients_and_crew["crew"]
    initial_inputs = {'initial_query': "The bomb is active. Get initial state."}
    try:
        result = crew.kickoff(inputs=initial_inputs)
        print(f"Kickoff result: {result}")
    except Exception as e:
        print(f"Error during crew kickoff: {e}")
    finally:
        await clients_and_crew["defuser_client"].cleanup()
        await clients_and_crew["expert_client"].cleanup()

if __name__ == '__main__':
    import asyncio
    asyncio.run(_test_crew_creation())
