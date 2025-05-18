# crewai_bomb/tools.py
import asyncio
import traceback # Optional: for more detailed error logging during development
from typing import Type

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
    defuser_game_client: Type[DefuserClient] = None

    def __init__(self, defuser_game_client: DefuserClient, **kwargs):
        super().__init__(**kwargs)
        # Ensure class-defined name and description are used if not overridden by kwargs to super()
        # self.name = DefuserTool.name?
        # self.description = DefuserTool.description
        self.defuser_game_client = defuser_game_client

    def _run(self, command: str) -> str:
        """
        Sends a command to the bomb via the DefuserClient and returns the game's response.
        The 'command' argument is the string command formulated by the Defuser LLM agent.
        """
        print(f"[{self.name}] Received command from agent: '{command}'")
        try:
            # Call the async method from the synchronous _run using asyncio.run
            # nest_asyncio.apply() in main.py makes this possible.
            # response = asyncio.run(self.defuser_game_client.run(action=command))
            response = self.defuser_game_client.run(action=command)
            print(f"[{self.name}] Response from game: '{response}'")
            return response
        except Exception as e:
            print(f"[{self.name}] Error executing command '{command}': {e}")
            # traceback.print_exc() # Uncomment for detailed stack trace during debugging
            return f"Error: Could not execute command '{command}'. Detail: {str(e)}"

    def clean_up(self):
        self.defuser_game_client.cleanup()

    # def __del__(self):
    #     self.defuser_game_client.cleanup()




class ExpertTool(BaseTool):
    name: str = "ExpertManualTool"
    description: str = (
        "Use this tool to retrieve the bomb defusal manual for the current bomb module. "
        "The tool returns the relevant manual content as a string."
    )
    # This client is an instance of the ExpertClient class from game_mcp.game_client.py
    expert_game_client: Type[ExpertClient] = None

    def __init__(self, expert_game_client: ExpertClient, **kwargs):
        super().__init__(**kwargs)
        # Ensure class-defined name and description are used
        # self.name = ExpertTool.name
        # self.description = ExpertTool.description
        self.expert_game_client = expert_game_client

    def _run(self) -> str:
        """
        Retrieves manual information using the ExpertClient based on the module query.
        The 'module_query' is the string input from the Expert LLM agent.
        """
        print(f"[{self.name}] Received manual query from agent.'")
        try:

            manual_content = self.expert_game_client.run()
            print(f"[{self.name}] Manual content retrieved.")
            # To avoid overwhelming the LLM, you might want to summarize or indicate if content is too long.
            # For now, returning the full content.
            return manual_content
        except Exception as e:
            print(f"[{self.name}] Error retrieving manual: {e}")
            # traceback.print_exc() # Uncomment for detailed stack trace
            return f"Error: Could not retrieve manual content'. Detail: {str(e)}"

    def clean_up(self):
        self.expert_game_client.cleanup()

    # def __del__(self):
    #     self.expert_game_client.cleanup()