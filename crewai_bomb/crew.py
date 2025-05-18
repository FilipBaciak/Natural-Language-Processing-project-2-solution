# crewai_bomb/crew.py
import os
from typing import Dict, Any

from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your custom tools
from crewai_bomb.tools_old import DefuserTool, ExpertTool

# Import game clients
from game_mcp.game_client import Defuser as DefuserClient, Expert as ExpertClient

# Default model name - can be overridden by environment variable or specific calls
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Standard model name for Gemini Flash

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
    # You can adjust temperature, top_p, top_k as needed, similar to Task 2.2 experiments
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=gemini_api_key,
        temperature=0.5, # Moderate temperature for a balance of creativity and determinism
        top_p=0.9,
        # top_k=40, # top_k can also be configured
        # max_output_tokens=1024 # Optional: set a default max token limit
    )
    print(f"Using LLM: {GEMINI_MODEL_NAME}")

    # 2. Create and connect game clients for the tools
    # These clients will be managed (connected here, cleaned up in main.py)
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
    # Defuser Agent
    defuser_agent = Agent(
        role="Bomb Defuser Specialist",
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
        # memory=True # Optional: enable memory for the agent if complex multi-turn context is needed beyond task outputs
    )

    # Expert Agent
    expert_agent = Agent(
        role="Bomb Defusal Manual Expert and Strategist",
        goal=(
            "Receive and interpret bomb descriptions from the Defuser. "
            "Utilize the bomb defusal manual to identify the correct procedures. "
            "Provide clear, concise, and actionable single-step instructions to the Defuser. "
            "Ensure instructions are unambiguous and lead towards successful defusal."
        ),
        backstory=(
            "You are a master of explosive ordnance disposal (EOD) procedures, possessing "
            "encyclopedic knowledge of various bomb mechanisms via defusal manuals. "
            "You operate remotely, relying solely on the Defuser's reports. "
            "Your analytical skills and ability to provide precise guidance are critical "
            "to saving the day."
        ),
        llm=llm,
        tools=[expert_manual_tool],
        verbose=True,
        allow_delegation=False,
        # memory=True
    )
    print("Crew agents defined.")

    # 5. Define Tasks
    # Task for the Defuser to observe and describe the bomb state
    task_observe_and_describe = Task(
        description=(
            "It's your turn, Defuser. Your primary goal is to assess the bomb and inform the Expert.\n"
            "1. Use your 'DefuserActionTool' with the command 'state' to get the current raw status of the bomb.\n"
            "2. Analyze this raw status. If the output indicates the bomb is already 'disarmed' or 'exploded', "
            "your report should clearly state this game over condition and include the raw status.\n"
            "3. Otherwise, formulate a clear, concise natural language description of the bomb's current state "
            "based on the raw status. Focus on details pertinent to defusal (e.g., wire colors, button labels, module types).\n"
            "The input for this task might be {{initial_query}} or {{previous_round_outcome}} if provided by the system; use it for context.\n"
            "Your final output for this task MUST be a single string containing your observation and description for the Expert, or the game over status."
        ),
        agent=defuser_agent,
        expected_output=(
            "A string containing the raw bomb state (from the 'state' command) followed by a "
            "natural language description of the bomb, or a clear statement of game over status (e.g., 'Bomb exploded!', 'Bomb disarmed!')."
        ),
        # async_execution=False # Ensures sequential execution if not using context explicitly for all steps
    )

    # Task for the Expert to analyze and provide instructions
    task_expert_instruct = Task(
        description=(
            "Expert, the Defuser has provided an update on the bomb's status (this is the context from the previous task).\n"
            "1. Carefully analyze the Defuser's report. If it indicates the game is already over (disarmed/exploded), "
            "simply acknowledge this fact in your output.\n"
            "2. If the game is ongoing, use your 'ExpertManualTool' to consult the defusal manual. "
            "The Defuser's description of the bomb situation should be your query to the manual tool.\n"
            "3. Based on the manual and the Defuser's report, formulate ONE SINGLE, CLEAR, ACTIONABLE instruction for the Defuser to perform next. "
            "Be very specific (e.g., 'cut wire 3', 'press the red button', 'release on digit 5').\n"
            "DO NOT ask questions back. Provide a direct command or confirm game over.\n"
            "Your final output for this task MUST be a single string: either the precise instruction or the game over acknowledgement."
        ),
        agent=expert_agent,
        context=[task_observe_and_describe], # Depends on the output of the observation task
        expected_output=(
            "A single string containing one specific command for the Defuser (e.g., 'cut red wire', 'press button') "
            "or an acknowledgement if the game is already over."
        ),
        # async_execution=False
    )

    # Task for the Defuser to execute the action and report the result
    task_defuser_act = Task(
        description=(
            "Defuser, you have received an instruction from the Expert (this is the context from the previous task).\n"
            "1. If the Expert's message is a game over acknowledgement, your task is to simply repeat that acknowledgement as your output.\n"
            "2. Otherwise, the Expert has given you a specific command (e.g., 'cut wire 1', 'press button'). "
            "Use your 'DefuserActionTool' to execute this EXACT command on the bomb.\n"
            "3. Your final output for this task MUST be the full, raw response you receive from the game server after performing the action. "
            "This response will indicate the new bomb state, or if it exploded/was disarmed."
        ),
        agent=defuser_agent,
        context=[task_expert_instruct], # Depends on the output of the expert's instruction
        expected_output=(
            "The raw string response from the game server after executing the command "
            "(e.g., 'Remaining time: 60. Module: Wires. State: 3 wires present...'), "
            "or the game over acknowledgement from the expert."
        ),
        # async_execution=False
    )
    print("Crew tasks defined.")

    # 6. Assemble the Crew
    bomb_defusal_crew = Crew(
        agents=[defuser_agent, expert_agent],
        tasks=[task_observe_and_describe, task_expert_instruct, task_defuser_act],
        process=Process.sequential,  # Tasks will run in the order they are listed
        verbose=2  # 0 for no output, 1 for agent actions, 2 for agent actions and thoughts
        # memory=True # Optional: enable memory for the entire crew
    )
    print("Bomb Defusal Crew assembled.")

    return {
        "crew": bomb_defusal_crew,
        "defuser_client": defuser_game_client,
        "expert_client": expert_game_client
    }

# Example of how this might be called (for testing, actual call is in main.py)
async def _test_crew_creation():
    api_key = os.getenv("GEMINI_API_KEY")
    server = "http://localhost:8080"
    if not api_key:
        print("GEMINI_API_KEY not set.")
        return

    clients_and_crew = None
    try:
        clients_and_crew = await create_bomb_defusal_crew(server, api_key)
        crew = clients_and_crew["crew"]
        print("Crew created successfully.")
        # In a real scenario, you'd call crew.kickoff() here with some inputs
        # initial_inputs = {'initial_query': "The bomb is active. Get initial state."}
        # result = crew.kickoff(inputs=initial_inputs)
        # print(f"Kickoff result: {result}")
    except Exception as e:
        print(f"Error during crew creation test: {e}")
    finally:
        if clients_and_crew:
            if clients_and_crew["defuser_client"]:
                await clients_and_crew["defuser_client"].cleanup()
            if clients_and_crew["expert_client"]:
                await clients_and_crew["expert_client"].cleanup()

if __name__ == '__main__':
    # This requires an event loop to run _test_crew_creation
    # import asyncio
    # asyncio.run(_test_crew_creation())
    pass