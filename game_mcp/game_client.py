import argparse
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client  # SSE transport from SDK


class BombClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._sse_ctx: Optional[Any] = None  # Context manager returned by sse_client
        self._read = None
        self.write = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.server_url: Optional[str] = None

    async def connect_to_server(self, server_url: str):
        """
        Open an SSE connection to the MCP server and initialize an MCP ClientSession.
        """
        if self.session:
            print("Already connected. Disconnecting first to establish a new connection.")
            await self.cleanup()

        self.server_url = server_url
        print(f"Attempting to connect to MCP server at {server_url}...")
        try:
            self._sse_ctx = sse_client(server_url)
            self._read, self.write = await self.exit_stack.enter_async_context(self._sse_ctx)

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self._read, self.write)
            )
            await self.session.initialize()
            print(f"Successfully connected to MCP server at {server_url}")
            # For debugging connection:
            # print("Available tools:", (await self.session.list_tools()))
        except ConnectionRefusedError:
            print(f"Connection refused at {server_url}. Ensure the server is running.")
            self.server_url = None
            raise
        except Exception as e:
            print(f"An error occurred during connection: {e}")
            self.server_url = None
            await self.cleanup()  # Attempt to clean up any partial setup
            raise


    async def process_query(self, tool_name: str, tool_args: dict[str, str]) -> str:
        """
        Call a tool on the MCP server using the active session.
        """
        if not self.session:
            raise RuntimeError("Not connected to server. Call connect_to_server() first.")
        try:
            print(f"Calling tool '{tool_name}' with args '{tool_args}'...")
            response = await self.session.call_tool(tool_name, tool_args)
            print(f"Received response: {response}")
            return response.content[0].text
        except Exception as e:
            print(f"Error calling tool '{tool_name}' with args '{tool_args}': {e}")
            raise

    async def cleanup(self):
        """
        Close the MCP session and SSE connection cleanly.
        """
        print("Cleaning up client connection...")
        await self.exit_stack.aclose()
        self.session = None
        self._sse_ctx = None
        self._read = None
        self.write = None
        self.server_url = None
        print("Client connection cleaned up.\n\n\n")



class Defuser(BombClient):
    async def run(self, action: str) -> str:
        # Uses 'game_interaction' tool
        return await self.process_query('game_interaction', {'command': action})


class Expert(BombClient):
    async def run(self) -> str:
        # Uses 'get_manual' tool
        return await self.process_query('get_manual', {})


async def ainput(prompt: str = "") -> str:
    """Asynchronously get input from the console."""
    return await asyncio.get_event_loop().run_in_executor(
        None,  # Uses the default ThreadPoolExecutor
        input,
        prompt
    )

async def main():
    """ Main function to connect to the server and run the clients based on CLI args. """
    parser = argparse.ArgumentParser(description="MCP Bomb Defusal Client")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--role",
        choices=["Defuser", "Expert"],
        required=True,
        help="Client role: Defuser or Expert"
    )
    args = parser.parse_args()

    client: Optional[BombClient] = None  # Initialize client to None

    if args.role == "Defuser":
        client = Defuser()
    elif args.role == "Expert":
        client = Expert()
    else:
        # This case should not be reached due to argparse choices
        print(f"Invalid role: {args.role}")
        return

    try:
        await client.connect_to_server(args.url)
        print(f"\nConnected as {args.role}. Type 'quit' or 'exit' to stop.")
        print("----------------------------------------------------")

        if isinstance(client, Defuser):
            # Display initial state for the Defuser
            try:
                initial_state = await client.run("state")
                print(f"\nServer response (Initial State):\n{initial_state}")
                if "BOOM!" in initial_state or "BOMB SUCCESSFULLY DISARMED!" in initial_state:
                    print("Game is already over.")
                    return
            except Exception as e:
                print(f"Error getting initial state: {e}")
                return # Exit if initial state fails

            while True:
                action = await ainput("\nEnter command for Defuser (e.g., 'state', 'cut wire 1'): ")
                if action.lower() in ["quit", "exit"]:
                    break
                if not action.strip():
                    print("Please enter a command.")
                    continue

                try:
                    response = await client.run(action)
                    print(f"\nServer response:\n{response}")
                    if "BOOM!" in response or "BOMB SUCCESSFULLY DISARMED!" in response:
                        print("Game over.")
                        break
                except Exception as e:
                    print(f"Error during Defuser action: {e}")
                    # Decide if we should break or continue
                    # For now, let's allow continuing
                    # break

        elif isinstance(client, Expert):
            while True:
                user_input = await ainput("\nPress Enter to get manual, or type 'quit'/'exit' to stop: ")
                if user_input.lower() in ["quit", "exit"]:
                    break

                try:
                    manual = await client.run()  # Expert's run method calls get_manual
                    print(f"\nServer response (Manual):\n{manual}")
                    if "BOOM!" in manual or "BOMB SUCCESSFULLY DISARMED!" in manual:
                        print("Game over.")
                        break
                except Exception as e:
                    print(f"Error during Expert action: {e}")
                    # break

    except ConnectionRefusedError:
        # Already handled and printed in connect_to_server, but good to have a top-level catch
        print(f"Main: Could not connect to the server at {args.url}. Please ensure the server is running.")
    except RuntimeError as e:
        print(f"Main: Runtime error: {e}")
    except Exception as e:
        print(f"Main: An unexpected error occurred: {e}")
    finally:
        if client and client.session: # Check if client was initialized and connected
            print("\nShutting down client...")
            await client.cleanup()
        else:
            print("\nClient was not fully initialized or already cleaned up.")
        print("Client application finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting...")

async def expert_test(expert_client: Expert):
    """Test the Expert class"""
    result = await expert_client.run()
    possible_outputs = ["BOOM!", "BOMB SUCCESSFULLY DISARMED!", "Regular Wires Module", "The Button Module",
                        "Memory Module", "Simon Says Module"]

    assert any(result.find(output) != -1 for output in possible_outputs), f"Expert test failed"


async def defuser_test(defuser_client: Defuser):
    """Test the Defuser class"""
    result = await defuser_client.run("state")

    possible_outputs = ["BOMB STATE"]

    assert any(result.find(output) != -1 for output in possible_outputs), f"Defuser test failed"

