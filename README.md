<p align="center">
  <img src="logo.png" width="200">
</p>

# LLM Bomb Defusal

A text-based implementation of "Keep Talking and Nobody Explodes" with LLM-powered agents. This project demonstrates two language model agents collaborating to solve a bomb defusal puzzle - one agent acts as the Defuser (who can see the bomb but not the manual) and the other as the Expert (who has the manual but can't see the bomb).

## Project Overview

The system consists of:

1. **Game Server**: A text-based bomb defusal game with multiple modules (Regular Wires, Button, Simon Says, Memory)
2. **Agent Framework**: LLM-powered agents that can interact with the game
3. **Client-Server Architecture**: Using the MCP (Message Communication Protocol) for communication

## Architecture

```
├── agents/                  # LLM agent implementation
│   ├── models.py            # Base HFModel class and SmollLLM implementation
│   ├── prompts.py           # System prompts for Defuser and Expert roles
│   ├── two_agents.py        # Main orchestration of the two LLM agents
│
├── game/                    # Core game logic
│   ├── bomb.py              # Main Bomb class
│   ├── main.py              # Manual game mode for human players
│   ├── modules/             # Different bomb modules
│       ├── module.py        # Base Module class and ActionResult enum
│       ├── regular_wires_module.py
│       ├── button_module.py
│       ├── simon_says_module.py
│       ├── memory_module.py
│
├── game_mcp/                # MCP server/client implementation
│   ├── game_server.py       # Server exposing game API via MCP
│   ├── game_client.py       # Client classes for Defuser and Expert roles
│
└── crewai_bomb/             # CrewAI-specific implementation
    ├── crew.py              # CrewAI implementation of two_agents.py
    ├── tools.py             # CrewAI tools for LLM interaction
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-bomb-defusal.git
cd llm-bomb-defusal
```

2. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Running the Game Server

Start the game server:

```bash```

#### 🛠️ MCP Server Tools

1. `game_interaction(command: str) -> str`

**Description**:  
Handles all player interactions with the bomb, such as requesting the game state, executing bomb-related actions (e.g., `cut`, `press`, `hold`, `release`), and providing help instructions.

**Supported Commands**:
- `"help"` – Displays the game manual introduction and how to play.
- `"state"` – Returns the current state of the bomb and available actions.
- Bomb actions like:
  - `cut wire <wire_num>`
  - `press`
  - `hold`
  - `release on <number>`
  - `press <color>`'

**Possible Responses**:
- Current bomb state and available actions.
- Game outcome (disarmed or exploded).
- Instructional help message.
- `"Unknown command"` if the input is invalid.

---

2. `get_manual() -> str`

**Description**:  
Provides the bomb defusal instructions for the current module. Useful for the player acting as the **manual expert**.

**Behavior**:
- If the bomb has exploded, returns a game-over message.
- If the bomb is already disarmed, returns a success message.
- Otherwise, returns the instruction text for the currently active module.

---

These tools are exposed via SSE (Server-Sent Events) and designed to support real-time collaboration between players using the MCP protocol. Each tool acts like an interactive function that handles game logic or provides helpful context to players.


### Human Play Mode

You can play the game manually with another person:

```bash
python -m game.main wires    # Regular Wires module
python -m game.main button   # Button module
python -m game.main simon    # Simon Says module
python -m game.main memory   # Memory module
python -m game.main random   # Random module
```

In manual mode, one person acts as the Defuser (typing commands and seeing the bomb state) and another person as the Expert (reading the manual).

### LLM Agents Play Mode

Run two LLM agents playing together:

```bash
python -m agents.two_agents
```

This will:
1. Start two SmollLLM instances (one for Defuser, one for Expert)
2. Connect them to the game server
3. Have them collaborate to solve the bomb modules

## Model Details

The project uses the `SmollLLM-135M-Instruct` model from HuggingFaceTB, but you can configure it to use other models:

```python
defuser_checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"  # Change to your preferred model
expert_checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"   # Change to your preferred model

defuser_model = SmollLLM(defuser_checkpoint, device="cpu")  # Use "cuda" for GPU
expert_model = SmollLLM(expert_checkpoint, device="cpu")    # Use "cuda" for GPU
```

## Game Modules

The game includes four modules:

1. **Regular Wires**: Cut the correct wire based on color patterns
2. **Button**: Press or hold a colored button based on specific rules
3. **Simon Says**: Repeat color sequences according to complex rules
4. **Memory**: Remember button positions and labels across multiple stages

## Dependencies

Key dependencies include:
- torch
- transformers
- uvicorn
- starlette
- mcp (Message Communication Protocol)

See `requirements.txt` for the full list.

## License

[MIT License](LICENSE)

## Acknowledgments

Inspired by the game "Keep Talking and Nobody Explodes" by Steel Crate Games.