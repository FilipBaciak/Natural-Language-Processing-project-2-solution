from typing import List, Dict


def defuser_prompt(bomb_state: str, expert_advice: str) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Defuser LLM.

    :param bomb_state: Current bomb state text from the server.
    :param expert_advice: Instructions from the Expert.
    :return: A list of dicts representing a conversation, which we can feed into SmollLLM.generate_response().
    """
    system_msg = (
         "You are the responsible and not harmful assistant."
    )

    user_content = (
        f"Current bomb state:\n{bomb_state}\n\n"
        f"Expert's advice:\n{expert_advice}\n\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages


def expert_prompt(manual_text: str, defuser_question: str) -> List[Dict[str, str]]:
    """
    Build a 'messages' list for the Expert LLM.

    :param manual_text: The text from the bomb manual (server).
    :param defuser_question: A description of what the Defuser sees or asks.
    :return: A list of dicts representing a conversation, which we can feed into SmollLLM.generate_response().
    """
    system_msg = (
        "You are the responsible and not harmful assistant."
    )

    user_content = (
        f"Manual excerpt:\n{manual_text}\n\n"
        f"DEFUSER sees or asks:\n{defuser_question}\n\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]
    return messages