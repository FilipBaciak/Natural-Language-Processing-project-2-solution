# crewai_bomb/tools_old.py
import asyncio
import traceback # Optional: for more detailed error logging during development

from crewai.tools import BaseTool


from game_mcp.game_client import Defuser as DefuserClient, Expert as ExpertClient


class DefuserTool(BaseTool):
    name: str = "DefuserActionTool"
    description: str = (
        "Use this tool to perform an action on the bomb or query its state. "
        "Input must be a specific command string recognized by the game, "
        "e.g., 'state', 'cut wire 1', 'press button', 'release on 3'. "
        "The tool executes the command and returns the bomb's response."
    )
    # This client is an instance of the DefuserClient class from game_mcp.game_client.py
    defuser_game_client: DefuserClient

    def __init__(self, defuser_game_client: DefuserClient, **kwargs):
        super().__init__(**kwargs)
        # Ensure class-defined name and description are used if not overridden by kwargs to super()
        self.name = DefuserTool.name
        self.description = DefuserTool.description
        self.defuser_game_client = defuser_game_client

    def _run(self, command: str) -> str:
        """
        Sends a command to the bomb via the DefuserClient and returns the game's response.
        The 'command' argument is the string command formulated by the Defuser LLM agent.
        """
        print(f"[{self.name}] Received command from agent: '{command}'")
        try:
            # Ensure your DefuserClient class (from game_mcp.game_client.py)
            # has an async method like 'perform_action' that takes the command string
            # and interacts with the game server (e.g., using BombClient.send_command).
            if not hasattr(self.defuser_game_client, "perform_action"):
                err_msg = f"DefuserClient is missing the 'perform_action' method."
                print(f"[{self.name}] Error: {err_msg}")
                return f"Tool Error: {err_msg}"

            # Call the async method from the synchronous _run using asyncio.run
            # nest_asyncio.apply() in main.py makes this possible.
            response = asyncio.run(self.defuser_game_client.run(action=command))
            print(f"[{self.name}] Response from game: '{response}'")
            return response
        except Exception as e:
            print(f"[{self.name}] Error executing command '{command}': {e}")
            # traceback.print_exc() # Uncomment for detailed stack trace during debugging
            return f"Error: Could not execute command '{command}'. Detail: {str(e)}"

class ExpertTool(BaseTool):
    name: str = "ExpertManualTool"
    description: str = (
        "Use this tool to retrieve the bomb defusal manual for the current (or a specified) bomb module. "
        "Input should be a query or description related to the module (e.g., 'wires module', 'button rules'). "
        "The tool returns the relevant manual content as a string."
    )
    # This client is an instance of the ExpertClient class from game_mcp.game_client.py
    expert_game_client: ExpertClient

    def __init__(self, expert_game_client: ExpertClient, **kwargs):
        super().__init__(**kwargs)
        # Ensure class-defined name and description are used
        self.name = ExpertTool.name
        self.description = ExpertTool.description
        self.expert_game_client = expert_game_client

    def _run(self, module_query: str) -> str:
        """
        Retrieves manual information using the ExpertClient based on the module query.
        The 'module_query' is the string input from the Expert LLM agent.
        """
        print(f"[{self.name}] Received manual query from agent: '{module_query}'")
        try:
            # Ensure your ExpertClient class (from game_mcp.game_client.py)
            # has an async method like 'get_manual_instructions'.
            # This method would typically use BombClient.send_command("get_manual")
            # and might use 'module_query' if the backend supports more specific queries.
            if not hasattr(self.expert_game_client, "get_manual_instructions"):
                err_msg = "ExpertClient is missing the 'get_manual_instructions' method."
                print(f"[{self.name}] Error: {err_msg}")
                return f"Tool Error: {err_msg}"

            manual_content = asyncio.run(self.expert_game_client.get_manual_instructions(module_description=module_query))
            print(f"[{self.name}] Manual content retrieved.")
            # To avoid overwhelming the LLM, you might want to summarize or indicate if content is too long.
            # For now, returning the full content.
            return manual_content
        except Exception as e:
            print(f"[{self.name}] Error retrieving manual for query '{module_query}': {e}")
            # traceback.print_exc() # Uncomment for detailed stack trace
            return f"Error: Could not retrieve manual for '{module_query}'. Detail: {str(e)}"