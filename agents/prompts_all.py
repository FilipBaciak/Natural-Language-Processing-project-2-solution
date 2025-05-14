
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

def defuser_observation_prompt(bomb_state: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM to describe the bomb state
    it observes.

    :param bomb_state: Current bomb state text from the server.
    :return: A list of dicts for the Defuser LLM to generate a description.
    """
    window = history[-5:]
    system_msg = (
        "You are the Defuser. You are looking at a bomb module. "
        "Your primary task right now is to clearly and concisely describe what you see to your Expert partner. "
        "Your Expert partner has the manual but cannot see the bomb. "
        "Focus on details that would be relevant for identifying the module and its components "
        "based on a manual. For example, mention the number of wires and their colors, serial number, "
        "any symbols, numbers, button labels, or the sequence of flashing lights. "
        "Be factual and descriptive."
        "Describe everything you see and know about the bomb -- all the details."
        "The expert won't be able to ask you a question, soo be sure not to miss anything."
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


# # Rewritten defuser_prompt
# def defuser_prompt(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
#     """
#     Build a 'messages' list for the Defuser LLM to decide on an executable game command.
#     This version uses a very direct, short, and example-driven system prompt.
#
#     :param bomb_state: Current bomb state text from the server.
#     :param expert_advice: Instructions from the Expert.
#     :return: A list of dicts for the Defuser LLM to generate a precise game command.
#     """
#     system_msg = (
#         "You are a command generation bot. "
#         "Your SOLE function is to take observed bomb state and expert advice, then output a SINGLE, EXACT game command. "
#         "Your response MUST BE ONLY the command. NO other words. NO explanations. NO conversation. "
#         "VALID COMMANDS: 'cut wire <number>', 'press <color_or_label>', 'hold', 'release on <number>', 'state', 'help'.\n"
#         "YOU SHOULD ONLY USE COMMANDS SPECIFIED AN AVAILABLE COMMANDS LISTED BELOW, DO NOT USE COMMAND WHICH IS NOT EXACTLY WRITTEN AS AVAILABLE."
#         "YOUR RESPONSE SHOULD ALWAYS COINCIDE WITH THE EXPERT'S ADVISE, IF HE RECOMMENDS AN AVAILABLE COMMAND"
#         "DO NOT VIOLATE THE EXPERT'S ADVICE"
#     )
#     user_content = (
#         f"BOMB_STATE:\n{bomb_state}\n\n"
#         f"EXPERT_ADVICE:\n{expert_advice}\n\n"
#         f"YOUR RESPONSE SHOULD ALWAYS COINCIDE WITH THE EXPERT'S ADVISE"
#         f"COMMAND_ONLY:"
#     )
#     messages: List[Dict[str, str]] = [
#         {"role": "system", "content": system_msg},
#         {"role": "user", "content": user_content}
#     ]
#     return messages


def defuser_prompt(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build the messages list for the Defuser LLM. The defuser MUST follow the expert’s advice exactly
    and output only one of the allowed commands—nothing else.
    """
    system_msg = (
        "You are the Defuser Bot. Your ONLY task is to read the bomb state and then the expert’s advice, "
        "and emit exactly ONE valid game command—no explanations, no extra text. "
        "If the expert gives advice, you MUST follow it verbatim. Do NOT contradict or ignore it. "
        "Allowed commands (exactly as written):\n"
        "  • cut wire <number>\n"
        "  • press <color_or_label>\n"
        "  • hold\n"
        "  • release on <number>\n"
        # "  • state\n"
        # "  • help\n"
        # "If the expert’s advice is not one of these, respond with the expert’s recommended valid command. "
        # "If you cannot parse their advice into a valid command, respond “help”."
    )

    user_msg = (
        f"BOMB_STATE:\n{bomb_state}\n\n"
        f"EXPERT_ADVICE:\n{expert_advice}\n\n"
        "OUTPUT COMMAND ONLY:"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ]



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
        "Do not ask any questions -- give the Defuser a precise command. "
        "When formulating your instruction, aim for clarity that helps the Defuser easily determine an "
        "exact game command (e.g., 'tell them to cut wire 3', or 'advise them to press the blue button')."
        "Pay special attention to the order of the lights in the manual in the Simon Says module as the manual contains an error."
    )
    user_content = (
        f"Okay Expert, here's the situation. The Defuser is looking at the bomb and reports this:\n"
        f"--- Defuser's Report Start ---\n{defuser_description}\n--- Defuser's Report End ---\n\n"
        f"I've pulled up the relevant section of the manual for you:\n"
        f"--- Manual Excerpt Start ---\n{manual_text}\n--- Manual Excerpt End ---\n\n"
        "THE NEXT COMMENT APPLIES ONLY TO THE SIMON SAYS MODULE. "
        f"Beware of the error in the Simon Says module."
        "Remember: In Simon Says, treat \"Round N\" as \"the Nth flashing light\". "
        "Press buttons in the order of the flashing lights, mapping each color in the flashing light sequence to the corresponding table entry. "
        'For example, if the flashing light sequence is "red, blue", then press the "Red, Round 1" button, then the "Blue, Round 2" button.'
        "You should find the corresponding table entry for each flashing light in the manual."
        "THE SEQUENCE IS NOT THE FLASHING LIGHT SEQUENCE, BUT THE CORRESPONDING TABLE ENTRY."
        # "If some buttons are already pressed, always continue from the next flash in the sequence."
        # "For example, if your recommendation is to press the blue button and then the red button,"
        # "but the blue button is already pressed, then press just the red button."
        "The vowels are: A, E, I, O, U -- serial number contains an vowel only if it contains an A, E, I, O, or U."
        "Think whether the serial number contains an A, E, I, O, or U to choose the appropriate manual part."
        # "Y DOES NOT COUNT AS A VOWEL."
        "---"
        "Based on all that, what is the single, most direct instruction you can give the Defuser for their next action?"
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
        "You are an expert Bomb Defusal AI. Your role is to provide precise, actionable, single-step instructions to a human Defuser. "
        "You will be given a description of what the Defuser sees and an excerpt from the bomb defusal manual. "
        "Your response MUST be a direct command to the Defuser for their immediate next action. "
        "DO NOT ask questions. DO NOT offer alternatives. DO NOT explain your reasoning unless it's part of the command itself (e.g., 'Cut the red wire because...'). "
        "Aim for instructions that directly translate to game commands, e.g., 'Cut the third wire from the top', 'Press the blue button', 'Press the button labeled Detonate'. "
        "If the provided manual excerpt is for the 'Simon Says' module, pay EXTREMELY close attention to the special instructions for it provided in the user prompt."
    )

    # Note: The detailed Simon Says instructions are now more integrated into the user prompt
    # and are explicitly prefaced to apply *only* if the manual_text is for Simon Says.
    user_content = f"""
    **CONTEXT:**
    You are the Bomb Defusal Expert. Your human partner, the Defuser, is at the bomb.
    The Defuser has described what they see. You have the relevant manual excerpt.
    Your task is to provide THE SINGLE NEXT ACTION for the Defuser.

    **DEFUSER'S REPORT:**
    --- Defuser's Report Start ---
    {defuser_description}
    --- Defuser's Report End ---

    **MANUAL EXCERPT:**
    --- Manual Excerpt Start ---
    {manual_text}
    --- Manual Excerpt End ---

    **GENERAL INSTRUCTIONS FOR ALL MODULES:**
    1.  **Vowels:** For any checks involving vowels in the bomb's serial number, the vowels are A, E, I, O, U. Y is NOT a vowel.
    2.  **Clarity:** Your instruction must be unambiguous and tell the Defuser exactly what to do next.
    3.  **Reasoning:** Provide a short reasoning for your instruction.
    4.  **Single Action:** Provide only one action. The Defuser will report back, and you will then instruct the next step. 


    **SPECIAL INSTRUCTIONS -- APPLY THESE *ONLY IF* THE MANUAL EXCERPT IS FOR THE 'SIMON SAYS' MODULE:**
    The Simon Says module requires careful interpretation of the manual.
    1.  **Manual Error Correction / Interpretation:**
        *   The manual contains an error or can be misleading regarding its "Round N".
        *   You MUST interpret the manual's columns as corresponding to the *position of the light in the flashing sequence*.
        *   Specifically:
            *   For the **1st** flashing light, use the manual's information that corresponds to "Round 1".
            *   For the **2nd** flashing light, use the manual's information that corresponds to "Round 2".
            *   For the **3rd** flashing light (and any subsequent, if applicable), use the manual's information that corresponds to "Round 3".
        *   This positional mapping (1st flash -> Round 1, 2nd flash -> Round 2 column, etc.) is CRITICAL.
    2.  **Determining Button Presses:**
        *   The Defuser will report a sequence of flashing light colors.
        *   For EACH color in the flashing sequence, in order:
            a.  Note its color (e.g., Red).
            b.  Note its position in the flashing sequence (e.g., 1st, 2nd, 3rd).
            c.  Consult the manual: Find the row for the *flashing color* (e.g., Red).
            d.  In that row, find the column corresponding to its *position in the sequence* (as per rule 1 above, e.g., for 1st flash, use Round 1/0 Strikes column).
            e.  The entry in that cell is the color of the button the Defuser should press for *that specific flash*.
    3.  **Instruction Format for Simon Says:**
        *   Your instruction should be a next button to press.
        *   Example: If the flashing sequence is "Red (1st), Blue (2nd)" and the manual lookup (applying rule 1 & 2) determines:
            *   Red (1st flash) -> Press Yellow
            *   Blue (2nd flash) -> Press Green
        *   Then your instruction should be: "Press Yellow" if no buttons were pressed yet, or "Press Green" if the Yellow button was pressed already.

    4.  **Serial Number Vowel Check (Simon Says Specific):** The manual section for Simon Says will have different sub-tables based on whether the bomb's serial number contains a vowel. Use the Vowel definition from "GENERAL INSTRUCTIONS".

    **YOUR TASK:**
    Based on the Defuser's report, the manual excerpt, and ALL relevant instructions above (general and Simon Says-specific if applicable), what is the single, most direct, actionable command you give to the Defuser for their NEXT action?
    """
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