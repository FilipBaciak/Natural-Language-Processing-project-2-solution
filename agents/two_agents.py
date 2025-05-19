import asyncio
import json # For potential JSON parsing in future experiments
import os
import re
from typing import Any

# Import from agents.prompts (where you've defined your experimental prompts)
from agents.prompts import expert_prompt, defuser_prompt, defuser_observation_prompt
from game_mcp.game_client import Defuser, Expert
from agents.models import GeminiAPIModel


async def run_two_agents(
        defuser_model: Any, # Or Union[HFModel, GeminiAPIModel] if you define it
        expert_model: Any,  # Or Union[HFModel, GeminiAPIModel]
        server_url: str = "http://0.0.0.0:8080",
        max_new_tokens_action_advice: int = 2000, # Max tokens for expert advice and defuser action
        max_new_tokens_description: int = 1000,  # Max tokens for defuser's description
        max_new_tokens_defuser_action: int = 100,
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 50
) -> None:
    """
    Main coroutine that orchestrates two LLM agents (Defuser and Expert)
    interacting with the bomb-defusal server, with a more realistic
    Defuser observation-description-Expert advice flow.

    :param defuser_model: The HFModel for the Defuser's role.
    :param expert_model: The HFModel for the Expert's role.
    :param server_url: The URL where the bomb-defusal server is running.
    :param max_new_tokens_action_advice: Max tokens for advice/action generation.
    :param max_new_tokens_description: Max tokens for bomb state description generation.
    :param max_new_tokens_defuser_action: Max tokens for defuser action generation.
    :param temperature: LLM temperature setting.
    :param top_p: LLM top_p (nucleus sampling) setting.
    :param top_k: LLM top_k setting.
    """
    defuser_client = Defuser()
    expert_client = Expert()
    exchange_count = 0
    history = []

    print("--- Starting New Agent Run ---")
    print(f"LLM Parameters: Temperature={temperature}, Top-p={top_p}, Top-k={top_k}")
    print(f"Max Tokens (Action/Advice): {max_new_tokens_action_advice}, Max Tokens (Description): {max_new_tokens_description}")
    print("-" * 60)

    try:
        # 1) Connect both clients to the same server
        await defuser_client.connect_to_server(server_url)
        await expert_client.connect_to_server(server_url)

        while True:
            exchange_count += 1
            print(f"\n--- Exchange #{exchange_count} ---")

            # 2) Defuser checks the bomb's current state (RAW data from server)
            bomb_state_raw = await defuser_client.run("state")
            print("\n[DEFUSER sees BOMB STATE (RAW)]:")
            print(bomb_state_raw)

            if "Bomb disarmed!" in bomb_state_raw or "Bomb exploded!" in bomb_state_raw:
                print(f"\n--- Game Over (Exchange #{exchange_count}) ---")
                print(f"Final Bomb State: {bomb_state_raw.strip()}")
                break

            # 2.5) Defuser LLM describes the bomb state for the Expert
            print("\n[DEFUSER LLM is generating description of bomb state...]")
            obs_messages = defuser_observation_prompt(bomb_state_raw) # From agents.prompts
            defuser_description_for_expert = defuser_model.generate_response(
                obs_messages,
                max_new_tokens=max_new_tokens_description,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
            print("\n[DEFUSER'S GENERATED DESCRIPTION FOR EXPERT]:")
            print(defuser_description_for_expert)

            # 3) Expert retrieves the relevant manual text
            manual_text = await expert_client.run()
            print("\n[EXPERT sees MANUAL]:")
            print(manual_text) # Uncomment for verbose logging if needed

            # 4) Expert LLM uses the manual text + Defuser's GENERATED DESCRIPTION
            #    to generate instructions
            print("\n[EXPERT LLM is generating advice...]")
            exp_messages = expert_prompt(manual_text, defuser_description_for_expert, history) # Pass the description
            expert_advice_raw = expert_model.generate_response(
                exp_messages,
                max_new_tokens=max_new_tokens_action_advice,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )

            expert_advice_for_defuser = expert_advice_raw

            print("\n[EXPERT FINAL ADVICE to DEFUSER]:") # This will show what's actually passed
            print(expert_advice_for_defuser)

            # 5) Defuser LLM uses the RAW bomb state + expert advice to pick a single action
            #    The Defuser LLM needs the raw state to know what it's acting upon.
            print("\n[DEFUSER LLM is deciding action...]")
            def_messages = defuser_prompt(bomb_state_raw, expert_advice_for_defuser) # From agents.prompts
            def_action_raw = defuser_model.generate_response(
                def_messages,
                max_new_tokens=max_new_tokens_defuser_action,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
            history.append(def_action_raw)

            print("\n[DEFUSER RAW ACTION OUTPUT]:")
            print(def_action_raw)


            # 6) Attempt to extract a known command from def_action_raw
            #    If no recognized command is found, default to "help"
            action = "help"
            for line in def_action_raw.splitlines():
                line = line.strip().lower()
                if line.startswith(("cut", "press", "hold", "release", "help", "state")):
                    action = line.strip()
                    break

            print("\n[DEFUSER ACTION DECIDED]:", action)


            # 7) Send that action to the server
            result = await defuser_client.run(action)
            print("\n[SERVER RESPONSE]:")
            print(result)
            print("-" * 60) # End of exchange visual separator

            if "BOMB SUCCESSFULLY DISARMED" in result or "BOMB HAS EXPLODED" in result:
                print(f"\n--- Game Over (Exchange #{exchange_count}) ---")
                print(f"Final Result: {result.strip()}")
                break
    except ConnectionRefusedError:
        print(f"ERROR: Connection refused at {server_url}. Ensure the game server is running.")
    except Exception as e:
        print(f"An unexpected error occurred in run_two_agents: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nTotal exchanges in this run: {exchange_count}")
        await defuser_client.cleanup()
        await expert_client.cleanup()
        print("--- Agent Run Finished ---")


if __name__ == "__main__":

    # # --- Configuration for Experiments ---
    gemini_model_name = "gemini-2.0-flash"
    # LLM Inference Parameters to Experiment With.
    current_temperature = 0.1
    current_top_p = 1
    current_top_k = None
    current_max_new_tokens_action_advice = 2000 # Allow more tokens for reasoning if needed
    current_max_new_tokens_description = 1000
    param_defuser_action_max_tokens = 100  # Low to enforce consistent commands
    # Server URL
    game_server_url = "http://localhost:8080" # Ensure this matches your server
    # --- End of Configuration ---

    print("Initializing LLM models...")

    try:
        with open("gemini_API_key.json", "r") as f:
            data = json.load(f)
            gemini_key = data.get("api_key")
    except FileNotFoundError:
        gemini_key = os.getenv("GEMINI_API_KEY")

    try:  # Ensure your GEMINI_API_KEY environment variable is set
        defuser_model_instance = GeminiAPIModel(model_name=gemini_model_name, api_key=gemini_key)
        expert_model_instance = GeminiAPIModel(model_name=gemini_model_name, api_key=gemini_key)
    except ValueError as e:
        print(f"Error initializing GeminiAPIModel: {e}")
        print("Please ensure your GEMINI_API_KEY is set as an environment variable.")
        exit(1)
    print(f"Using Defuser & Expert Model: {gemini_model_name} (via API)")

    print(f"Starting game with server: {game_server_url}")
    asyncio.run(
        run_two_agents(
            defuser_model=defuser_model_instance,
            expert_model=expert_model_instance,
            server_url=game_server_url,
            max_new_tokens_action_advice=current_max_new_tokens_action_advice,
            max_new_tokens_defuser_action=param_defuser_action_max_tokens,  # Pass the very low value
            max_new_tokens_description=current_max_new_tokens_description,
            temperature=current_temperature,
            top_p=current_top_p,
            top_k=current_top_k
        )
    )