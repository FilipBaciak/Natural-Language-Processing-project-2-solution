# crewai_bomb/crew.py
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_bomb.tools import GameInteractionTool, GetManualTool
from game_mcp.game_client import BombClient # For type hinting

def create_bomb_defusal_crew(game_client: BombClient, gemini_api_key: str):
    # Initialize the Gemini LLM
    # Ensure GEMINI_API_KEY is set if not passing directly
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or passed as argument.")

    # Using "gemini-1.5-flash-latest" as it's a common identifier for the flash model.
    # You can try "gemini-2.0-flash" if that's the specific version you have access to
    # and it's supported by the ChatGoogleGenerativeAI wrapper.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=gemini_api_key,
        temperature=0.6, # Adjust for creativity vs. precision
        # top_p=0.9, # Optional: Adjust nucleus sampling
        # convert_system_message_to_human=True # May be useful depending on model and prompts
    )

    # Instantiate tools with the single, connected game_client
    interaction_tool = GameInteractionTool(game_client=game_client)
    manual_tool = GetManualTool(game_client=game_client)

    # Define Agents
    defuser_agent = Agent(
        role="Bomb Defuser",
        goal=(
            "Your mission is to defuse the bomb by meticulously following the Expert's guidance. "
            "Use the 'Game Interaction Tool' to interact with the bomb. "
            "1. Start by using the command 'state' to observe the bomb. "
            "2. Clearly describe the bomb's state (time, strikes, module details) to the Expert. "
            "3. Precisely execute the actions instructed by the Expert (e.g., 'cut wire 1') using the tool. "
            "4. After each action, use the 'state' command again to get the updated status. "
            "5. Report the full outcome (action taken, server response, new state) back to the Expert. "
            "Continue this cycle until the bomb is defused ('BOMB SUCCESSFULLY DISARMED!') or explodes ('BOOM!')."
        ),
        backstory=(
            "You are an elite operative, hands-on with a complex bomb. You can see and interact with it "
            "but lack the defusal manual. Success hinges on clear communication with the Expert and "
            "flawless execution of their instructions. Your life is on the line."
        ),
        tools=[interaction_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15 # Max iterations for this agent within a single task execution
    )

    expert_agent = Agent(
        role="Bomb Manual Expert",
        goal=(
            "Your objective is to guide the Defuser to successfully defuse the bomb. "
            "1. Listen carefully to the Defuser's description of the bomb module. "
            "2. Use the 'Get Manual Tool' to consult the bomb defusal manual for the current module. "
            "3. Based on the manual and the Defuser's observations, provide clear, unambiguous, step-by-step instructions. "
            "4. If the Defuser's information is insufficient, ask specific clarifying questions before giving an action command. "
            "5. Adapt your instructions based on the outcomes reported by the Defuser. "
            "Your guidance must lead to 'BOMB SUCCESSFULLY DISARMED!' and prevent 'BOOM!' at all costs."
        ),
        backstory=(
            "You are a veteran bomb disposal expert with access to all defusal manuals. You are remote "
            "and cannot see the bomb, relying entirely on the Defuser's reports. "
            "Precision, patience, and clarity in your instructions are paramount."
        ),
        tools=[manual_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15
    )

    # Define Tasks for one round of interaction
    task_describe_bomb = Task(
        description=(
            "You are the Defuser. Your first critical step is to understand the bomb. "
            "Use your 'Game Interaction Tool' with the command 'state'. This will give you the current bomb status, "
            "time remaining, strikes, and a description of the active module. "
            "Then, formulate a clear and concise message to the Expert, detailing everything you observed. "
            "Example: 'Bomb has 60s, 0 strikes. Module: Regular Wires. I see 4 wires: red, blue, yellow, black.' "
            "This initial observation is vital for the Expert."
        ),
        expected_output=(
            "A detailed textual description of the bomb's initial state and the visible module, "
            "as observed using the 'state' command. This description will be passed to the Expert."
        ),
        agent=defuser_agent,
    )

    task_provide_instructions = Task(
        description=(
            "You are the Expert. The Defuser has just provided you with their observation of the bomb module. "
            "Carefully review their message. "
            "Now, use your 'Get Manual Tool' to fetch the instructions for the currently active bomb module. "
            "Correlate the Defuser's observation with the manual's rules. "
            "Formulate precise, step-by-step instructions for the Defuser on the exact action to take. "
            "Example: 'Based on your description and the manual, you should cut the blue wire.' "
            "If the Defuser's information is insufficient for a clear decision, ask specific clarifying questions "
            "before providing an action command."
        ),
        expected_output=(
            "Clear, actionable instructions for the Defuser (e.g., 'Cut the second wire from the top.') "
            "OR specific clarifying questions if the Defuser's initial description was ambiguous or incomplete."
        ),
        agent=expert_agent,
        context=[task_describe_bomb], # This task uses the output of task_describe_bomb
    )

    task_execute_and_report = Task(
        description=(
            "You are the Defuser. The Expert has provided you with instructions. "
            "Carefully read and understand the Expert's command. "
            "Use your 'Game Interaction Tool' to execute the specified action (e.g., 'cut blue wire', 'press red button'). "
            "IMMEDIATELY AFTER EXECUTING THE ACTION, use the 'Game Interaction Tool' again with the 'state' command "
            "to get the updated status of the bomb (time, strikes, current module, etc.). "
            "Report back to the Expert with: "
            "1. The action you took. "
            "2. The server's direct response to your action. "
            "3. The new full bomb state from your follow-up 'state' command. "
            "Crucially, explicitly state if the response was 'BOOM!' or 'BOMB SUCCESSFULLY DISARMED!'. "
            "If the game is not over, this comprehensive report will help the Expert decide the next step."
        ),
        expected_output=(
            "A comprehensive report including: the action taken, the server's immediate response to that action, "
            "the new full bomb state (from a subsequent 'state' call), "
            "and a clear indication if the game is over (defused/exploded). Example: "
            "'Action: cut blue wire. Server response: Wire cut successfully. New state: Bomb has 45s, 0 strikes. Next module: The Button...' "
            "Or: 'Action: cut red wire. Server response: BOOM! New state: Bomb has exploded... Game over.'"
        ),
        agent=defuser_agent,
        context=[task_provide_instructions], # This task uses the output of task_provide_instructions
    )

    # Create Crew with sequential process for one round of describe -> instruct -> execute
    bomb_defusal_crew = Crew(
        agents=[defuser_agent, expert_agent],
        tasks=[task_describe_bomb, task_provide_instructions, task_execute_and_report],
        process=Process.sequential,
        verbose=2, # 0 for no output, 1 for task-level, 2 for agent-level thoughts and actions
        # memory=True, # Enable memory for the crew to remember past interactions in the session
        # full_output=True # If you want the output of each task in the final result
    )
    # Note: This setup runs one full cycle. For a complete game with multiple modules/steps,
    # you might need to:
    # 1. Loop `crew.kickoff()` in your main.py, checking the game state.
    # 2. Design more complex prompts and potentially a manager agent if the LLMs
    #    don't naturally continue the conversation across multiple cycles within this structure.
    # The `max_iter` on agents allows them multiple thought-action steps *within their current task*.

    return bomb_defusal_crew