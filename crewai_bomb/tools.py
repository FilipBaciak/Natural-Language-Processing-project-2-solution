from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from game_mcp.game_client import Defuser, Expert

class DefuserToolInput(BaseModel):
    prompt: str = Field(..., description="Command or question to the Defuser client.")
    class Config:
        extra = 'allow'

class DefuserTool(BaseTool):
    name: str = "Bomb Defuser Client"
    description: str = "Sends commands to the bomb defusal server and returns observations or action results."
    args_schema = DefuserToolInput

    async def _run(self, prompt: str) -> str:
        # Assume game_mcp.game_client.Defuser exists and connects to the server
        client = Defuser()
        client.connect_to_server()
        response = client.game_interaction(prompt)

        return response


class ExpertToolInput(BaseModel):
    query: str = Field(..., description="Expert query about bomb component.")
    class Config:
        extra = 'allow'

class ExpertTool(BaseTool):
    name: str = "Bomb Expert Manual"
    description: str = "Provides instructions from the bomb defusal manual."
    args_schema = ExpertToolInput

    async def _run(self, query: str) -> str:
        expert = Expert()
        result = expert.get_manual(query)
        return result
