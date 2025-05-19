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
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # Standard model name for Gemini Flash

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
        model=f"gemini/{GEMINI_MODEL_NAME}",  # specifies Gemini 2.0 Flash model
        api_key=gemini_api_key,
        provider="gemini/",
        config={'temperature': 0.5, 'top_p': 0.8, 'top_k': 20}

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
    defuser_action_tool = DefuserTool(server_url = server_url)
    expert_manual_tool = ExpertTool(server_url = server_url)
    print("Crew tools instantiated.")

    # 4. Define Agents
    defuser_agent = Agent(
        role="Defuser",
        goal=(
            "Interface with a live bomb. YOU MUST use tools to get the bomb's current state, "
            "accurately describe this OBSERVED state to the Expert, and then precisely execute the Expert's "
            "instructions using tools. Your actions directly determine success or failure."
        ),
        backstory=(
            "You are a highly trained field operative. You are at the bomb site. "
            "You DO NOT have prior knowledge of the bomb's state before each observation. "
            "YOU MUST WAIT FOR THE RESULT OF THE TOOL CALL BEFORE PROCEEDING. "
            "THE BOMB STATE CHANGES FROM OBSERVATION TO OBSERVATION. "
            "You rely on your tools for observation and action, and a remote Expert for defusal knowledge. "
            "Clarity in your descriptions (based ONLY on tool output) and precision in your actions are paramount. Failure is not an option."
        ),
        llm=llm,
        tools=[defuser_action_tool],
        verbose=True,
        allow_delegation=False,
        # memory = False,
        cache = False,
    )

    expert_agent = Agent(
        # ... (expert_agent definition remains the same) ...
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
        cache = False,
    )
    print("Crew agents defined.")

    # 5. Define Tasks
    task_observe_and_describe = Task(
        description=(
            "DEFUSER, YOUR IMMEDIATE AND CRITICAL TASK IS TO ASSESS THE BOMB AND REPORT ITS STATE TO THE EXPERT. "
            "YOU CURRENTLY HAVE NO INFORMATION ABOUT THE BOMB'S STATE. YOU MUST NOT GUESS OR RECALL ANY PREVIOUS STATE. "
            "FOLLOW THESE STEPS IMPERATIVELY AND IN THE SPECIFIED ORDER:\n"
            "1. INVOKE TOOL TO GET RAW DATA: YOU MUST IMMEDIATELY use your 'DefuserActionTool' with the exact command 'state'. "
            "THIS IS YOUR FIRST AND ABSOLUTELY MANDATORY ACTION. This tool call will return the 'raw_bomb_status'. "
            "DO NOT PROCEED WITHOUT SUCCESSFULLY EXECUTING THIS TOOL CALL AND OBTAINING THE 'raw_bomb_status'.\n"
            "DO NOT USE ANY OTHER COMMAND THAN 'state'."
            "2. ANALYZE 'raw_bomb_status' FOR GAME END: Once the 'DefuserActionTool' returns the 'raw_bomb_status', YOU MUST EXAMINE IT. "
            "If, AND ONLY IF, the 'raw_bomb_status' explicitly states the bomb is 'disarmed' or 'exploded' (or similar definitive game-ending phrases like 'BOOM!'), "
            "YOU MUST report this game over condition. Your report MUST include the FULL, UNMODIFIED 'raw_bomb_status' as evidence. "
            "DO NOT declare game over under any other circumstances.\n"
            "3. GENERATE DESCRIPTION FOR EXPERT (IF GAME IS ONGOING): If the 'raw_bomb_status' does NOT indicate the game is over, "
            "YOU MUST then generate a detailed, factual description of the bomb's current appearance and components. "
            "Focus on details that would be relevant for identifying the module and its components "
            "based on a manual. For example, mention the number of wires and their colors, serial number, "
            "any symbols, numbers, button labels, or the sequence of flashing lights. "
            "Be factual and descriptive."
            "Describe everything you see and know about the bomb -- all the details."
            "In particular, you should describe all the numbers you can see such as stage number."
            "Do not say anything else -- just the information in the bomb state. "
            "This description MUST BE BASED SOLELY AND ENTIRELY ON THE 'raw_bomb_status' YOU JUST OBTAINED FROM THE 'DefuserActionTool' IN STEP 1. "
            "Focus on observable details critical for defusal (e.g., wire colors, numbers on wires, button labels, module types, displayed symbols or numbers, active lights). "
            "BE PRECISE, THOROUGH, AND OBJECTIVE. DO NOT ADD ANY INTERPRETATIONS, ASSUMPTIONS, OR INFORMATION NOT DIRECTLY PRESENT IN THE 'raw_bomb_status'."
            "YOU SHOULD ALSO DESCRIBE ALL THE PROVIDED COMMANDS."),
        agent=defuser_agent,
        expected_output=(
            "EITHER: A highly detailed and factual description of the bomb's current state, explicitly stated as being derived *directly* from the 'raw_bomb_status' obtained via the 'DefuserActionTool'. The description must only contain information present in the tool's output.\n"
            "OR: A definitive game over statement (e.g., 'GAME OVER: Bomb Disarmed as per server response.' or 'GAME OVER: Bomb Exploded as per server response.') which MUST include the complete, verbatim 'raw_bomb_status' received from the server."
        ),
    )

    task_expert_instruct = Task(
        # ... (task_expert_instruct definition can remain largely the same, but ensure it emphasizes using the Defuser's *latest* report)
        description=(
            "EXPERT, THE DEFUSER HAS PROVIDED AN UPDATE WHICH IS SUPPOSEDLY BASED ON A FRESH OBSERVATION. YOUR TASK IS TO PROVIDE A SINGLE, ACTIONABLE INSTRUCTION. EXECUTE THESE STEPS:\n"
            "1. SCRUTINIZE the Defuser's report. If the report indicates the game is over (disarmed/exploded), YOU MUST simply acknowledge this fact. NO FURTHER ACTION IS NEEDED.\n"
            "2. IF THE GAME IS ONGOING: YOU MUST use your 'ExpertManualTool' IMMEDIATELY. Use the Defuser's LATEST provided description as the SOLE basis for your query to the manual. DO NOT rely on any previous information.\n"
            "FIRST USE THE MANUAL AND THEN REASON ABOUT YOUR OUTPUT"
            "3. BASED ON THE MANUAL AND THE DEFUSER'S LATEST DESCRIPTION, YOU MUST FORMULATE EXACTLY ONE, CLEAR, UNAMBIGUOUS, AND ACTIONABLE INSTRUCTION for the Defuser. "
            "This instruction MUST be a command the Defuser can directly execute with their tool (e.g., 'cut wire 3', 'press red'). "
            "DO NOT provide explanations, dialogues, or multiple options. ONE. SINGLE. COMMAND."
            "YOUR COMMAND MUST BE ONE OF THE COMMAND PROVIDED BY THE DEFUSER."

            f"""
                **Reasoning:**  Reason about your action.
                **Single Action:** Provide only one action. The Defuser will report back, and you will then instruct the next step. 

                **SPECIAL INSTRUCTIONS -- APPLY THESE *ONLY IF* THE MANUAL EXCERPT IS FOR THE 'SIMON SAYS' MODULE:**
                The Simon Says module requires careful interpretation of the manual.

                0.  IGNORE THE ROUND NUMBER PROVIDED BY THE DEFUSER ENTIRELY 
                1.  **Vowels:** For any checks involving vowels in the bomb's serial number, the vowels are A, E, I, O, U. 
                Y is NOT a vowel.
                The vowels are: A, E, I, O, U -- serial number contains an vowel only if it contains an A, E, I, O, or U.
                Think whether the serial number contains an A, E, I, O, or U to choose the appropriate manual part.
                Pay special attention to whether the serial number contains an A, E, I, O, or U - it is of utmost importance.
                First repeat the serial number to yourself and then think whether it contains an A, E, I, O, or U.
                Repeat each letter of the serial number to yourself and then think whether it is one of the vowels A, E, I, O or U.
                When you find a vowel or reach the end of the serial number, tell yourself whether you found a vowel or not.
                ---

                2.  **Manual Error Correction / Interpretation:**
                    *   The manual contains an error or can be misleading regarding its "Round N".
                    *   You MUST interpret the manual's columns as corresponding to the *position of the light in the flashing sequence*.
                    *   Specifically:
                        *   For the **1st** flashing light, use the manual's information that corresponds to "Round 1".
                        *   For the **2nd** flashing light, use the manual's information that corresponds to "Round 2".
                        *   For the **3rd** flashing light (and any subsequent, if applicable), use the manual's information that corresponds to "Round 3".
                    *   This positional mapping (1st flash -> Round 1, 2nd flash -> Round 2 column, etc.) is CRITICAL.
                3.  **Determining Button Presses:**
                    *   The Defuser will report a sequence of flashing light colors and the inputs.
                    *   If none inputs were pressed yet, use the first light in the flashing sequence.
                    *   If the Defuser provided no inputs, use the first light in the flashing sequence.
                    *   If the button says "Press a colored button to start sequence" or haven't mentioned the previous inputs, use the first light in the flashing sequence.
                    *   If N inputs were pressed already, use the light in the flashing sequence which is at the position corresponding to the N+1.
                    *   Repeat each color in the input and count them accordingly. Deduce the adequate light in the flashing sequence.
                    *   Provide the reasoning for your choice and make sure that it is correct, as it is a crucial part.  
                    *   Reflect upon your choice and double check that it is correct. 
                    *   YOU SHOULD ALWAYS DOUBLE CHECK YOUR CHOICE IN YOUR REASONING.
                    *   For the chosen color in the flashing sequence, in order:
                        a.  Note its color (e.g., Red).
                        b.  Note its position in the flashing sequence (e.g., 1st, 2nd, 3rd).
                        c.  Consult the manual: Find the row for the *flashing color* (e.g., Red).
                        d.  In that row, find the column corresponding to its *position in the sequence* (as per rule 1 above, e.g., for 1st flash, use Round 1 column).
                        e.  The entry in that cell is the color of the button the Defuser should press for *that specific flash*.
                4.  **Instruction Format for Simon Says:**
                    *   Example: If the flashing sequence is "Red (1st), Blue (2nd)" and the manual lookup (applying rule 1 & 2) determines:
                        *   Red (1st flash) -> Press Yellow
                        *   Blue (2nd flash) -> Press Green
                    *   Then your instruction should be: "Press Yellow" if no buttons were pressed yet, or "Press Green" if the Yellow button was pressed already.

                5  .  **Serial Number Vowel Check (Simon Says Specific):** The manual section for Simon Says will have different sub-tables based on whether the bomb's serial number contains a vowel. Use the Vowel definition from "GENERAL INSTRUCTIONS".
    """

        ),

        agent=expert_agent,
        context=[task_observe_and_describe],
        expected_output=(
            "A single, precise, directly executable command for the Defuser, OR a simple game over acknowledgement."
        ),
    )

    task_defuser_act = Task(
        # ... (task_defuser_act definition remains largely the same) ...
        description=(
            "DEFUSER, YOU HAVE RECEIVED AN INSTRUCTION FROM THE EXPERT. YOUR TASK IS TO EXECUTE IT OR ACKNOWLEDGE GAME OVER. FOLLOW THESE STEPS:\n"
            "1. EXAMINE the Expert's instruction. If it is a game over acknowledgement, YOU MUST repeat that acknowledgement verbatim. NO OTHER ACTION IS REQUIRED.\n"
            "2. IF IT IS AN ACTION COMMAND: YOU MUST execute that command using your 'DefuserActionTool' IMMEDIATELY. DO NOT DEVIATE. DO NOT INTERPRET. EXECUTE THE COMMAND AS GIVEN.\n"
            "DO NOT SAY THAT THE GAME IS OVER IF YOU ARE NOT TOLD SO BY THE EXPERT."
            "3. YOU MUST output the ENTIRE, UNMODIFIED, raw server response you receive after performing the action. This is CRITICAL for the next cycle."
            "DO NOT EXECUTE MORE THAN ONE COMMAND."
        ),
        agent=defuser_agent,
        context=[task_expert_instruct],
        expected_output=(
            "The complete, raw string response from the game server after executing the command, OR the verbatim game over acknowledgement from the Expert."
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
