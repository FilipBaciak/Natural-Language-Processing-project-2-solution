
from typing import List, Dict
import json

# --- Configuration 1: Standard Text-Based Prompts ---

def defuser_observation_prompt_standard(bomb_state: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are the Defuser. Describe the bomb module to your Expert partner concisely, "
        "focusing on details relevant for identification based on a manual (e.g., "
        "wire colors/count, symbols, numbers, labels, lights)."
    )
    user_content = (
        f"Observe the bomb module:\n{bomb_state}\n\n"
        f"Provide a clear description for your Expert."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]


def expert_prompt_standard(manual_text: str, defuser_description: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are an experienced Bomb Defusal Expert. The Defuser will describe their observations. "
        "Consult the manual excerpt and provide clear, step-by-step instructions for the immediate next action. "
        "Be precise and focus on one action at a time."
    )
    user_content = (
        f"Defuser's Observation:\n{defuser_description}\n\n"
        f"Manual Excerpt:\n{manual_text}\n\n"
        f"Give the single, most direct instruction for the Defuser's next action."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]


def defuser_prompt_standard(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a command generation bot. Output ONLY the EXACT game command based on the observed bomb state "
        "and Expert's advice. NO other text. NO explanations. "
        "VALID COMMANDS: cut wire <number>, press <color_or_label>, hold, release on <number>, state, help. "
        "Examples:\n"
        "- Advice: Cut the red wire, State: wire 1 is red -> Output: cut wire 1\n"
        "- Advice: Press the blue button -> Output: press blue\n"
        "- Advice unclear/ambiguous -> Output: help"
    )
    user_content = (
        f"BOMB_STATE:\n{bomb_state}\n\n"
        f"EXPERT_ADVICE:\n{expert_advice}\n\n"
        f"COMMAND_ONLY:"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]


# --- Configuration 2: Structured Markdown Prompts ---

def defuser_observation_prompt(bomb_state: str) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM to describe the bomb state
    it observes.

    :param bomb_state: Current bomb state text from the server.
    :return: A list of dicts for the Defuser LLM to generate a description.
    """
    system_msg = (
        "You are the Defuser. You are looking at a bomb module. "
        "Your primary task right now is to clearly and concisely describe what you see to your Expert partner. "
        "Your Expert partner has the manual but cannot see the bomb. "
        "Focus on details that would be relevant for identifying the module and its components "
        "based on a manual. For example, mention the number of wires and their colors, "
        "any symbols, numbers, button labels, or the sequence of flashing lights. "
        "Be factual and descriptive."
        "Describe everything you see and know about the bomb."
    )
    user_content = (
        f"--- Bomb State Information Start ---\n{bomb_state}\n--- Bomb State Information End ---\n\n"
        f"Please formulate a clear description of this module that I can send to my Expert partner. "
        f"What should I tell them about what I see?"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


# Rewritten defuser_prompt
def defuser_prompt(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM to decide on an executable game command.
    This version uses a very direct, short, and example-driven system prompt.

    :param bomb_state: Current bomb state text from the server.
    :param expert_advice: Instructions from the Expert.
    :return: A list of dicts for the Defuser LLM to generate a precise game command.
    """
    system_msg = (
        "You are a command generation bot. "
        "Your SOLE function is to take observed bomb state and expert advice, then output a SINGLE, EXACT game command. "
        "Your response MUST BE ONLY the command. NO other words. NO explanations. NO conversation. "
        "VALID COMMANDS: 'cut wire <number>', 'press <color_or_label>', 'hold', 'release on <number>', 'state', 'help'.\n"
        # "EXAMPLES:\n"
        # "If advice is 'Cut the first red wire' and state shows 'wire 1 is red', output: cut wire 1\n"
        # "If advice is 'Press the blue button', output: press blue\n"
        # "If advice is 'Hold the button', output: hold\n"
        # "If advice is 'Release when the timer shows a 4', output: release on 4\n"
        # "If advice is unclear or you cannot determine a valid command, output: help\n"
        # "Any deviation from outputting ONLY the command is a FAILURE."
        "YOU SHOULD ONLY USE COMMANDS SPECIFIED AN AVAILABLE COMMANDS LISTED BELOW, DO NOT USE COMMAND WHICH IS NOT EXACTLY WRITTEN AS AVAILABLE."
        "YOUR RESPONSE MUST ALWAYS BE A COMMAND WHICH IS LISTED IN THE BOMB STATE AS AVAILABLE"
    )
    user_content = (
        f"BOMB_STATE:\n{bomb_state}\n\n"
        f"EXPERT_ADVICE:\n{expert_advice}\n\n"
        f"COMMAND_ONLY:"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages

def expert_prompt(manual_text: str, defuser_description: str) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM to provide advice.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_description: A natural language description of what the Defuser sees.
    :return: A list of dicts for the Expert LLM to generate clear instructions.
    """
    system_msg = (
        "You are a helpful and very experienced Bomb Defusal Expert. Your partner, the Defuser, "
        "is at the bomb and will describe what they see. You have access to the bomb defusal manual. "
        "Listen carefully to the Defuser's description, consult the manual excerpt provided, "
        "and then give clear, step-by-step instructions on what the Defuser should do next. "
        "Be precise and focus only on the immediate next action required. "
        "When formulating your instruction, aim for clarity that helps the Defuser easily determine an "
        "exact game command (e.g., 'tell them to cut wire 3', or 'advise them to press the blue button')."
    )
    user_content = (
        f"Okay Expert, here's the situation. The Defuser is looking at the bomb and reports this:\n"
        f"--- Defuser's Report Start ---\n{defuser_description}\n--- Defuser's Report End ---\n\n"
        f"I've pulled up the relevant section of the manual for you:\n"
        f"--- Manual Excerpt Start ---\n{manual_text}\n--- Manual Excerpt End ---\n\n"
        f"Based on all that, what is the single, most direct instruction you can give the Defuser for their next action?"
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


# --- Configuration 3: JSON-Formatted Prompts ---

def defuser_observation_prompt_json(bomb_state: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are the Defuser. Describe the bomb module to your Expert partner concisely, "
        "providing a JSON object with the key 'description' containing relevant details "
        "for identification based on a manual (e.g., wire colors/count, symbols, numbers, labels, lights)."
    )
    user_content = json.dumps({"bomb_state": bomb_state})
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]


def expert_prompt_json(manual_text: str, defuser_description: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are an experienced Bomb Defusal Expert. The Defuser will send a JSON object "
        "with the key 'description'. Consult the manual excerpt and provide a JSON object "
        "with the key 'instruction' containing clear, step-by-step instructions for the immediate next action. "
        "Be precise and focus on one action at a time."
    )
    user_content = json.dumps({
        "defuser_description": defuser_description,
        "manual_excerpt": manual_text
    })
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}]


def defuser_prompt_json(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a command generation bot. You will receive a JSON object with 'bomb_state' and 'expert_advice'. "
        "Output ONLY a JSON object with the key 'command' containing the EXACT game command. "
        "VALID COMMANDS: cut wire <number>, press <color_or_label>, hold, release on <number>, state, help. "
        "Examples:\n"
        "- Advice: Cut the first red wire, State: wire 1 is red -> {\"command\": \"cut wire 1\"}\n"
        "- Advice: Press the blue button -> {\"command\": \"press blue\"}\n"
        "- Advice unclear/ambiguous -> {\"command\": \"help\"}"
        "YOU SHOULD ONLY USE COMMANDS SPECIFIED AN AVAILABLE COMMANDS LISTED BELOW, DO NOT USE COMMAND WHICH IS NOT EXACTLY WRITTEN AS AVAILABLE."
        "YOUR RESPONSE MUST ALWAYS BE A COMMAND WHICH IS LISTED IN THE BOMB STATE AS AVAILABLE"
    )
    user_content = json.dumps({
        "bomb_state": bomb_state,
        "expert_advice": expert_advice
    })
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages