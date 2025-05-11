# crewai_bomb/tools.py
import asyncio
from crewai_tools import BaseTool
from pydantic import ConfigDict # For Pydantic v2 model_config
from game_mcp.game_client import BombClient # Import the base BombClient

# IMPORTANT: If you run into issues with asyncio event loops,
# especially "RuntimeError: This event loop is already running",
# you'll need to use nest_asyncio.
# Add `import nest_asyncio` and `nest_asyncio.apply()`
# at the beginning of your `crewai_bomb/main.py` script.

class GameInteractionTool(BaseTool):
    name: str = "Game Interaction Tool"
    description: str = (
        "Use this tool to interact with the bomb game. "
        "Input should be a command string (e.g., 'state', 'cut wire 1', 'press button'). "
        "Always use 'state' first to understand the bomb or after an action to see the result."
    )
    game_client: BombClient
    model_config = ConfigDict(extra='allow') # Pydantic v2 style

    def __init__(self, game_client: BombClient, **kwargs):
        super().__init__(**kwargs)
        self.game_client = game_client
        if not self.game_client.session:
            # This check is important. The client should be connected.
            print("Warning: GameClient passed to GameInteractionTool may not be connected.")

    def _run(self, command: str) -> str:
        if not command:
            return "Error: No command provided to Game Interaction Tool."
        if not self.game_client.session:
            return "Error: GameClient is not connected to the server. Cannot perform action."
        try:
            # asyncio.run can be problematic if an event loop is already running.
            # nest_asyncio.apply() in the main script helps manage this.
            return asyncio.run(self.game_client.process_query('game_interaction', {'command': command}))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e) or \
               "Task attached to a different loop" in str(e):
                # This indicates nest_asyncio might not be applied or there's a complex loop issue.
                return (f"Asyncio loop error during game interaction: {e}. "
                        "Ensure nest_asyncio.apply() is called at the start of your main script.")
            return f"Error executing game interaction command '{command}': {e}"
        except Exception as e:
            return f"Error executing game interaction command '{command}': {e}"

class GetManualTool(BaseTool):
    name: str = "Get Manual Tool"
    description: str = (
        "Use this tool to retrieve the relevant section of the bomb defusal manual "
        "for the current module. Input to this tool is ignored."
    )
    game_client: BombClient
    model_config = ConfigDict(extra='allow') # Pydantic v2 style

    def __init__(self, game_client: BombClient, **kwargs):
        super().__init__(**kwargs)
        self.game_client = game_client
        if not self.game_client.session:
            print("Warning: GameClient passed to GetManualTool may not be connected.")

    def _run(self, argument: str = None) -> str: # Argument is ignored
        if not self.game_client.session:
            return "Error: GameClient is not connected to the server. Cannot get manual."
        try:
            return asyncio.run(self.game_client.process_query('get_manual', {}))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e) or \
               "Task attached to a different loop" in str(e):
                return (f"Asyncio loop error during get manual: {e}. "
                        "Ensure nest_asyncio.apply() is called at the start of your main script.")
            return f"Error retrieving manual: {e}"
        except Exception as e:
            return f"Error retrieving manual: {e}"
