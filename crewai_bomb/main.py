# crewai_bomb/main.py
import argparse
import asyncio
import json
import os
import nest_asyncio # For handling nested asyncio event loops
from game_mcp import game_client
from crewai_bomb.crew import create_bomb_defusal_crew

async def run_crew_defusal(server_url: str, gemini_api_key: str):
    # Apply nest_asyncio once at the beginning.
    # This is crucial for allowing asyncio.run() calls from tools
    # if the main script or CrewAI itself runs within an event loop.
    nest_asyncio.apply()

    try:
        # Create the Bomb Defusal Crew
        bomb_defusal_crew_dict = await create_bomb_defusal_crew(server_url, gemini_api_key)
        bomb_defusal_crew = bomb_defusal_crew_dict["crew"]

        print("\nStarting the Bomb Defusal Crew in a loop...")

        game_continues = True
        iteration_count = 0
        max_iterations = 150  # Safety break to prevent infinite loops
        final_status_message = "Game ended due to maximum iterations."

        while game_continues and iteration_count < max_iterations:
            iteration_count += 1
            print(f"\n--- Starting Crew Iteration {iteration_count} ---")

            # The `inputs` for kickoff. For our current task setup in crew.py,
            # the first task (task_observe_and_describe) always starts by fetching the current bomb state.
            # Therefore, no specific inputs are strictly needed to carry over state between iterations here.
            # If your tasks were designed to need explicit context from the *previous full crew run*,
            # this is where you'd pass it, e.g., inputs={'previous_outcome': last_result}
            # The agents have their memory and can use it to remember the previous inputs.
            kickoff_inputs = {}

            result = await bomb_defusal_crew.kickoff_async(inputs=kickoff_inputs)

            print("\n----------------------------------------------------")
            print(f"Output of Crew Iteration {iteration_count}:")
            print(result)  # This 'result' is the output of the last task in the crew sequence
            print("----------------------------------------------------")

            # Check for game-ending conditions in the result string
            result = str(result)
            result_lower = result.lower()

            if "bomb successfully disarmed" in result_lower or \
                    "bomb disarmed!" in result_lower:
                final_status_message = "BOMB SUCCESSFULLY DISARMED!"
                print(f"Game Concluded: {final_status_message}")
                game_continues = False
            elif "bomb has exploded" in result_lower or \
                    "bomb exploded!" in result_lower or \
                    "boom!" in result_lower:  # "BOOM!" is a common server response for explosion
                final_status_message = "BOMB EXPLODED!"
                print(f"Game Concluded: {final_status_message}")
                game_continues = False
            elif "game over" in result_lower:  # Check if an agent explicitly states "game over"
                # This could happen if task_observe_and_describe sees a pre-existing ended state
                final_status_message = f"Game Over: {result}"
                print(f"Game Concluded: {final_status_message}")
                game_continues = False

            if not game_continues and iteration_count == max_iterations:  # Overwrite if max_iterations hit
                final_status_message = f"Game ended: Reached maximum iterations ({max_iterations}). Last result: {result}"

        # You might want to check the game_client's last known state or parse 'result'
        # to determine if the bomb was defused or exploded for a final status message.
        # For example, if the last 'result' contains "BOMB SUCCESSFULLY DISARMED!" or "BOOM!".

        await bomb_defusal_crew_dict["defuser_client"].cleanup()
        await bomb_defusal_crew_dict["expert_client"].cleanup()
    except ConnectionRefusedError:
        print(f"Main Error: Could not connect to the server at {server_url}. Ensure the server is running.")
    except Exception as e:
        print(f"Main Error: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="CrewAI Bomb Defusal Client")
    try:
        with open("gemini_API_key.json", "r") as f:
            data = json.load(f)
            gemini_key = data.get("api_key")
    except FileNotFoundError:
        gemini_key = os.getenv("GEMINI_API_KEY")
    parser.add_argument(
        "--url",
        default=os.getenv("MCP_SERVER_URL", "http://localhost:8080"),
        help="MCP Game Server URL (default: http://localhost:8080 or MCP_SERVER_URL env var)"
    )
    parser.add_argument(
        "--gemini_api_key",
        default=gemini_key,
        help="Gemini API Key (can also be set via GEMINI_API_KEY env var or in a json file)"
    )
    args = parser.parse_args()

    if not args.gemini_api_key:
        print("Error: Gemini API Key is required. "
              "Set --gemini_api_key argument or GEMINI_API_KEY environment variable.")
        return

    # Python's asyncio.run is a good way to run the top-level async function
    try:
        asyncio.run(run_crew_defusal(args.url, args.gemini_api_key))
    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting...")
        import sys
        sys.exit(0)
    except Exception as e:
        print(f"Main Error: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()