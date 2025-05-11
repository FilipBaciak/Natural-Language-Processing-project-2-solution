# crewai_bomb/main.py
import argparse
import asyncio
import os
import nest_asyncio # For handling nested asyncio event loops

from game_mcp.game_client import BombClient # Use the base BombClient
from crewai_bomb.crew import create_bomb_defusal_crew

async def run_crew_defusal(server_url: str, gemini_api_key: str):
    # Apply nest_asyncio once at the beginning.
    # This is crucial for allowing asyncio.run() calls from tools
    # if the main script or CrewAI itself runs within an event loop.
    nest_asyncio.apply()

    game_client = BombClient() # Single client instance for the shared game session

    try:
        print(f"Attempting to connect to game server at {server_url}...")
        await game_client.connect_to_server(server_url)
        print(f"Successfully connected to game server at {server_url}.")

        # Create the Bomb Defusal Crew, passing the connected client
        bomb_defusal_crew = create_bomb_defusal_crew(game_client, gemini_api_key)

        print("\nKicking off the Bomb Defusal Crew...")
        # The `inputs` for kickoff would be used if your first task in the crew
        # explicitly requires input from the kickoff call.
        # In our current setup, the first task (task_describe_bomb) initiates its own action.
        # So, inputs={} or no inputs argument is fine.
        result = bomb_defusal_crew.kickoff()

        print("\n----------------------------------------------------")
        print("Crew execution finished.")
        print("Final Result/Output of the last task:")
        print(result)
        print("----------------------------------------------------")

        # You might want to check the game_client's last known state or parse 'result'
        # to determine if the bomb was defused or exploded for a final status message.
        # For example, if the last 'result' contains "BOMB SUCCESSFULLY DISARMED!" or "BOOM!".

    except ConnectionRefusedError:
        print(f"Main Error: Could not connect to the server at {server_url}. Ensure the server is running.")
    except Exception as e:
        print(f"Main Error: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if game_client and game_client.session:
            print("\nShutting down client connection...")
            await game_client.cleanup()
        else:
            print("\nClient was not fully initialized or already cleaned up.")
        print("Bomb Defusal Crew application finished.")

def main():
    parser = argparse.ArgumentParser(description="CrewAI Bomb Defusal Client")
    parser.add_argument(
        "--url",
        default=os.getenv("MCP_SERVER_URL", "http://localhost:8080"),
        help="MCP Game Server URL (default: http://localhost:8080 or MCP_SERVER_URL env var)"
    )
    parser.add_argument(
        "--gemini_api_key",
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API Key (can also be set via GEMINI_API_KEY env var)"
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

if __name__ == "__main__":
    main()