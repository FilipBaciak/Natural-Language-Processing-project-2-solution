from typing import List, Dict
import json


# This file contains many prompt configurations used in the experiments
# It is not an file that should be a part of an executed script




# --- Configuration 1 ---
# Natural language, without reasoning

from typing import List, Dict


def defuser_observation_prompt1(bomb_state: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM to describe the bomb state
    it observes.

    :param bomb_state: Current bomb state text from the server.
    :param history: A list of past interactions (kept in signature, not used in this version).
    :return: A list of dicts for the Defuser LLM to generate a description.
    """
    user_content = (
        "You are the Defuser. Your Expert partner cannot see the bomb or ask clarifying questions, "
        "so your description must be complete. "
        "Factually describe everything visible on the module: number of wires and their colors, "
        "serial number, any symbols, numbers (like stage number), button labels, or light sequences. "
        "Be thorough."
        f"Here's the information about the bomb: \n{bomb_state}\n\n"
        f"Describe this module for my Expert partner."
    )

    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_content}
    ]
    return messages


def defuser_prompt1(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build the messages list for the Defuser LLM. The defuser MUST follow the expert’s advice exactly
    and output only the specified command(s).
    """
    user_msg = (
        "You are the Defuser Bot. Your task is to execute the expert's advice precisely. "
        "Output only the game command(s) as specified below. No extra text or explanations.\n"
        "Default: Output EXACTLY ONE command: `cut wire <number>`, `press <color_or_label>`, `hold`, `release on <number>`.\n"
        "EXCEPTION for Memory Module: If the current module is Memory, you MUST output two lines: \n"
        "1. The label of the button to press and its position (e.g., 'Button Label: 4, Position: 1').\n"
        "2. The command (e.g., 'press 4').\n"
        "Follow the expert's advice verbatim."
        f"Here's the description of the bomb state by the defuser: \n{bomb_state}\n\n"
        f"Here's the advice provided by the expert: \n{expert_advice}\n\n"
        "Your command output:"
    )

    return [
        {"role": "user", "content": user_msg}
    ]


def expert_prompt1(manual_text: str, defuser_description: str, history: List[str]) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM to provide advice.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_description: A natural language description of what the Defuser sees.
    :param history: A list of previous Defuser actions.
    :return: A list of dicts for the Expert LLM to generate clear instructions.
    """

    user_content = (
        "You are a Bomb Defusal Expert. Provide a single, precise, actionable command "
        "for the Defuser based on their report and the manual. "
        "If the module is Simon Says, adhere strictly to the Simon Says specific instructions provided in the user message."
        f"""
        Defuser's Report:
        {defuser_description}
        \n
        Manual Excerpt:
        {manual_text}
        \n
        
        History of Defuser's actions: {(" ".join(history) if history else "No actions yet.")}
        
        **Simon Says Module Instructions (Apply *only if* this is the Simon Says module):**
        1.  **Serial Vowel Rule:** Check if the bomb's serial number contains a vowel (A, E, I, O, U only). This determines which part of the Simon Says manual to use.
        2.  **Flash-to-Manual Mapping:**
            *   The manual's "Round 1" column applies to the 1st light in the flashing sequence.
            *   "Round 2" column applies to the 2nd light.
            *   "Round 3" column applies to the 3rd (and any subsequent) lights.
            *   Ignore any round number reported by the Defuser.
        3.  **Determining Press:**
            *   Identify the current light in the flashing sequence to act upon (e.g., if N buttons were already pressed for this Simon Says sequence, focus on the (N+1)th light).
            *   Using this light's color (row in manual) and its sequence position mapped to the manual's "Round" (column in manual, per rule 2), find the button color to press.
        
        What is your advice for the defuser?
    """)
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_content}
    ]
    return messages






# --- Configuration 2 ---
#Structured markdowns without reasoning and explicit instructions
from typing import List, Dict

def defuser_observation_prompt2(bomb_state: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM to describe the bomb state
    it observes.

    :param bomb_state: Current bomb state text from the server.
    :param history: A list of past interactions (kept in signature, not used in this version).
    :return: A list of dicts for the Defuser LLM to generate a description.
    """
    user_content = (
        "You are the Defuser. Your Expert partner cannot see the bomb or ask clarifying questions, "
        "so your description must be complete. "
        "Factually describe everything visible on the module: number of wires and their colors, "
        "serial number, any symbols, numbers (like stage number), button labels, or light sequences. "
        "Be thorough."
        f"--- Bomb State Information Start ---\n{bomb_state}\n--- Bomb State Information End ---\n\n"
        f"Describe this module for my Expert partner."
    )
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_content}
    ]
    return messages


def defuser_prompt2(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build the messages list for the Defuser LLM. The defuser MUST follow the expert’s advice exactly
    and output only the specified command(s).
    """
    user_msg = (
        "You are the Defuser Bot. Your task is to execute the expert's advice precisely. "
        "Output only the game command(s) as specified below. No extra text or explanations.\n"
        "Default: Output EXACTLY ONE command: `cut wire <number>`, `press <color_or_label>`, `hold`, `release on <number>`.\n"
        "EXCEPTION for Memory Module: If the current module is Memory, you MUST output two lines: \n"
        "1. The label of the button to press and its position (e.g., 'Button Label: 4, Position: 1').\n"
        "2. The command (e.g., 'press 4').\n"
        "Follow the expert's advice verbatim."
        f"BOMB_STATE:"
        "--- Bomb State Start ---\n"
        f"\n{bomb_state}\n"
        "--- Bomb State Start ---\n\n"
        
        "--- Expert Advice Start ---\n"
        f"EXPERT_ADVICE:\n{expert_advice}\n"
        "--- Expert Advice End ---\n\n"
        
        "Your command output:"
    )

    return [
        {"role": "user", "content": user_msg}
    ]


def expert_prompt2(manual_text: str, defuser_description: str, history: List[str]) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM to provide advice.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_description: A natural language description of what the Defuser sees.
    :param history: A list of previous Defuser actions.
    :return: A list of dicts for the Expert LLM to generate clear instructions.
    """

    user_content = (
        "You are a Bomb Defusal Expert. Provide a single, precise, actionable command "
        "for the Defuser based on their report and the manual. "
        "If the module is Simon Says, adhere strictly to the Simon Says specific instructions provided in the user message."
        f"""
        Defuser's Report:
        --- Defuser's Report Start ---
        {defuser_description}
        --- Defuser's Report End ---
        
        Manual Excerpt:
        --- Manual Excerpt Start ---
        {manual_text}
        --- Manual Excerpt End ---
        
        History of Defuser's actions: 
        --- Defuser's Actions Start ---
        {(" ".join(history) if history else "No actions yet.")}
        --- Defuser's Actions End ---
        
        **Simon Says Module Instructions (Apply *only if* this is the Simon Says module):**
        1.  **Serial Vowel Rule:** Check if the bomb's serial number contains a vowel (A, E, I, O, U only; Y is not a vowel). This determines which part of the Simon Says manual to use.
        2.  **Flash-to-Manual Mapping:**
            *   The manual's "Round 1" column applies to the 1st light in the flashing sequence.
            *   "Round 2" column applies to the 2nd light.
            *   "Round 3" column applies to the 3rd (and any subsequent) lights.
            *   Ignore any round number reported by the Defuser.
        3.  **Determining Press:**
            *   Identify the current light in the flashing sequence to act upon (e.g., if N buttons were already pressed for this Simon Says sequence, focus on the (N+1)th light).
            *   Using this light's color (row in manual) and its sequence position mapped to the manual's "Round" (column in manual, per rule 2), find the button color to press.
        
        What is the single, direct, actionable command for the Defuser's next action?
    """)
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_content}
    ]
    return messages






# --- Configuration 3 ---
#Jsons without reasoning and explicit instructions


from typing import List, Dict


def defuser_observation_prompt3(bomb_state: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM to describe the bomb state
    it observes.

    :param bomb_state: Current bomb state text from the server.
    :param history: A list of past interactions (kept in signature, not used in this version).
    :return: A list of dicts for the Defuser LLM to generate a description.
    """
    user_content = (
        "You are the Defuser. Your Expert partner cannot see the bomb or ask clarifying questions, "
        "so your description must be complete. "
        "Factually describe everything visible on the module: number of wires and their colors, "
        "serial number, any symbols, numbers (like stage number), button labels, or light sequences. "
        "Be thorough."
    )
    user_content = (
        f"--- Bomb State Information Start ---\n{bomb_state}\n--- Bomb State Information End ---\n\n"
        f"Describe this module for my Expert partner."
    )
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


def defuser_prompt3(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build the messages list for the Defuser LLM. The defuser MUST follow the expert’s advice exactly
    and output only the specified command(s).
    """
    system_msg = (
        "You are the Defuser Bot. Your task is to execute the expert's advice precisely. "
        "Output only the game command(s) as specified below. No extra text or explanations.\n"
        "Default: Output EXACTLY ONE command: `cut wire <number>`, `press <color_or_label>`, `hold`, `release on <number>`.\n"
        "EXCEPTION for Memory Module: If the current module is Memory, you MUST output two lines: \n"
        "1. The label of the button to press and its position (e.g., 'Button Label: 4, Position: 1').\n"
        "2. The command (e.g., 'press 4').\n"
        "Follow the expert's advice verbatim."
    )

    user_msg = (
        f"BOMB_STATE:"
        "--- Bomb State Start ---\n"
        f"\n{bomb_state}\n"
        "--- Bomb State Start ---\n\n"

        "--- Expert Advice Start ---\n"
        f"EXPERT_ADVICE:\n{expert_advice}\n"
        "--- Expert Advice End ---\n\n"

        "Your command output:"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def expert_prompt3(manual_text: str, defuser_description: str, history: List[str]) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM to provide advice.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_description: A natural language description of what the Defuser sees.
    :param history: A list of previous Defuser actions.
    :return: A list of dicts for the Expert LLM to generate clear instructions.
    """

    system_msg = (
        "You are a Bomb Defusal Expert. Provide a single, precise, actionable command "
        "for the Defuser based on their report and the manual. "
        "If the module is Simon Says, adhere strictly to the Simon Says specific instructions provided in the user message."
    )

    user_content = f"""
        Defuser's Report:
        --- Defuser's Report Start ---
        {defuser_description}
        --- Defuser's Report End ---

        Manual Excerpt:
        --- Manual Excerpt Start ---
        {manual_text}
        --- Manual Excerpt End ---

        History of Defuser's actions: 
        --- Defuser's Actions Start ---
        {(" ".join(history) if history else "No actions yet.")}
        --- Defuser's Actions End ---

        **Simon Says Module Instructions (Apply *only if* this is the Simon Says module):**
        1.  **Serial Vowel Rule:** Check if the bomb's serial number contains a vowel (A, E, I, O, U only; Y is not a vowel). This determines which part of the Simon Says manual to use.
        2.  **Flash-to-Manual Mapping:**
            *   The manual's "Round 1" column applies to the 1st light in the flashing sequence.
            *   "Round 2" column applies to the 2nd light.
            *   "Round 3" column applies to the 3rd (and any subsequent) lights.
            *   Ignore any round number reported by the Defuser.
        3.  **Determining Press:**
            *   Identify the current light in the flashing sequence to act upon (e.g., if N buttons were already pressed for this Simon Says sequence, focus on the (N+1)th light).
            *   Using this light's color (row in manual) and its sequence position mapped to the manual's "Round" (column in manual, per rule 2), find the button color to press.

        What is the single, direct, actionable command for the Defuser's next action?
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


# --- Configuration 4:  ---
# Json + Structured Markdown Prompts + explicit reasoning enforcement

def defuser_observation_prompt4(bomb_state: str, history: List[Dict[str, str]] = []) -> List[Dict[str, str]]:
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
        "In particular, you should describe all the numbers you can see such as stage number."
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


def defuser_prompt4(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build the messages list for the Defuser LLM. The defuser MUST follow the expert’s advice exactly
    and output only one of the allowed commands—nothing else.
    """
    system_msg = (
        "You are the Defuser Bot. Your ONLY task is to read the bomb state and then the expert’s advice, "
        "and emit exactly ONE valid game command—no explanations, no extra text. "
        "There is only ONE exception to this rule:"
        "If you are in the memory module, print the label and position of the button you should press and then the command in the next line."
        "If the expert gives advice, you MUST follow it verbatim. Do NOT contradict or ignore it. "
        "Allowed commands (exactly as written):\n"
        "  • cut wire <number>\n"
        "  • press <color_or_label>\n"
        "  • hold\n"
        "  • release on <number>\n"
    )

    user_msg = (
        f"BOMB_STATE:\n{bomb_state}\n\n"
        f"EXPERT_ADVICE:\n{expert_advice}\n\n"
        "IF YOU ARE IN THE MEMORY MODULE, PRINT THE LABEL AND POSITION (with appropriate names) OF THE BUTTON YOU SHOULD PRESS AND THEN THE COMMAND IN THE NEXT LINE."
        "OUTPUT COMMAND ONLY:"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def expert_prompt4(manual_text: str, defuser_description: str, history: List[str]) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM to provide advice.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_description: A natural language description of what the Defuser sees.
    :return: A list of dicts for the Expert LLM to generate clear instructions.
    """

    system_msg = (
        "You are an expert Bomb Defusal AI. Your role is to provide precise, actionable, single-step instructions to a human Defuser. "
        "You will be given a description of what the Defuser sees and an excerpt from the bomb defusal manual. "
        "Repeat to yourself the reasing for choosing the next action. "
        "Aim for instructions that directly translate to game commands, e.g., 'Cut the third wire from the top', 'Press the blue button', 'Press the button labeled Detonate'. "
        "If the provided manual excerpt is for the 'Simon Says' module, pay EXTREMELY close attention to the special instructions for it provided in the user prompt."
    )

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

    2.  **Clarity:** Your instruction must be unambiguous and tell the Defuser exactly what to do next.
    3.  **Reasoning:** 
    4.  **Single Action:** Provide only one action. The Defuser will report back, and you will then instruct the next step. 

    **You are also provided with the following history of the Defuser's actions:**
    {" ".join(history)}

    **SPECIAL INSTRUCTIONS -- APPLY THESE *ONLY IF* THE MANUAL EXCERPT IS FOR THE 'SIMON SAYS' MODULE:**
    The Simon Says module requires careful interpretation of the manual.

    0.  IGNORE THE ROUND NUMBER PROVIDED BY THE DEFUSER ENTIRELY 
    1.  **Vowels:** For any checks involving vowels in the bomb's serial number, the vowels are A, E, I, O, U. 
    Y is NOT a vowel.
    The vowels are: A, E, I, O, U -- serial number contains an vowel only if it contains an A, E, I, O, or U.
    Think whether the serial number contains an A, E, I, O, or U to choose the appropriate manual part.
    Pay special attention to whether the serial number contains an A, E, I, O, or U - it is of utmost importance.
    First repeat the serial number to yourself and then think whether it contains an A, E, I, O, or U.
    Repeat each letter of the serial number to yourself and then think whether it is one of the vowels A, E, I, O or U.
    When you find a vowel or reach the end of the serial number, tell yourself whether you found a vowel or not.
    ---

    2.  **Manual Error Correction / Interpretation:**
        *   The manual contains an error or can be misleading regarding its "Round N".
        *   You MUST interpret the manual's columns as corresponding to the *position of the light in the flashing sequence*.
        *   Specifically:
            *   For the **1st** flashing light, use the manual's information that corresponds to "Round 1".
            *   For the **2nd** flashing light, use the manual's information that corresponds to "Round 2".
            *   For the **3rd** flashing light (and any subsequent, if applicable), use the manual's information that corresponds to "Round 3".
        *   This positional mapping (1st flash -> Round 1, 2nd flash -> Round 2 column, etc.) is CRITICAL.
    3.  **Determining Button Presses:**
        *   The Defuser will report a sequence of flashing light colors and the inputs.
        *   If none inputs were pressed yet, use the first light in the flashing sequence.
        *   If the button says "Press a colored button to start sequence", use the first light in the flashing sequence.
        *   If N inputs were pressed already, use the light in the flashing sequence which is at the position corresponding to the N+1.
        *   Analyse the number of inputs provided -- use the adequate light in the flashing sequence.
        *   Provide the reasoning for your choice and make sure that it is correct, as it is a crucial part.  
        *   Reflect upon your choice and double check that it is correct.     
        *   For the chosen color in the flashing sequence, in order:
            a.  Note its color (e.g., Red).
            b.  Note its position in the flashing sequence (e.g., 1st, 2nd, 3rd).
            c.  Consult the manual: Find the row for the *flashing color* (e.g., Red).
            d.  In that row, find the column corresponding to its *position in the sequence* (as per rule 1 above, e.g., for 1st flash, use Round 1 column).
            e.  The entry in that cell is the color of the button the Defuser should press for *that specific flash*.
    4.  **Instruction Format for Simon Says:**
        *   Example: If the flashing sequence is "Red (1st), Blue (2nd)" and the manual lookup (applying rule 1 & 2) determines:
            *   Red (1st flash) -> Press Yellow
            *   Blue (2nd flash) -> Press Green
        *   Then your instruction should be: "Press Yellow" if no buttons were pressed yet, or "Press Green" if the Yellow button was pressed already.

    5  .  **Serial Number Vowel Check (Simon Says Specific):** The manual section for Simon Says will have different sub-tables based on whether the bomb's serial number contains a vowel. Use the Vowel definition from "GENERAL INSTRUCTIONS".

    **YOUR TASK:**
    Based on the Defuser's report, the manual excerpt, and ALL relevant instructions above (general and Simon Says-specific if applicable), what is the single, most direct, actionable command you give to the Defuser for their NEXT action?
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages