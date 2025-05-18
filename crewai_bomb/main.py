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
    # nest_asyncio.apply()


    try:

        # Create the Bomb Defusal Crew, passing the connected client
        bomb_defusal_crew_dict = await create_bomb_defusal_crew(server_url, gemini_api_key)

        print("\nKicking off the Bomb Defusal Crew...")
        # The `inputs` for kickoff would be used if your first task in the crew
        # explicitly requires input from the kickoff call.
        # In our current setup, the first task (task_describe_bomb) initiates its own action.
        # So, inputs={} or no inputs argument is fine.
        bomb_defusal_crew = bomb_defusal_crew_dict["crew"]
        result = bomb_defusal_crew.kickoff()

        print("\n----------------------------------------------------")
        print("Crew execution finished.")
        print("Final Result/Output of the last task:")
        print(result)
        print("----------------------------------------------------")

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
    except Exception as e:
        print(f"Main Error: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()