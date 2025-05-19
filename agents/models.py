import os
from abc import ABC, abstractmethod
import torch
from typing import List, Dict, Optional, Any # Make sure Any is imported
import google.generativeai as genai



class GeminiAPIModel:
    """
    A model class to interact with the Google Gemini API.
    It implements a 'generate_response' method compatible with the existing agent script.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        """
        Initializes the Gemini API model.

        :param model_name: The name of the Gemini model to use.
        :param api_key: Your Gemini API key. If None, reads from GEMINI_API_KEY env variable.
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. "
                "Set GEMINI_API_KEY environment variable or pass api_key."
            )

        genai.configure(api_key=api_key)

        self.model_name = model_name
        # We will create the model instance with system_instruction inside generate_response
        # to allow per-call system prompts if needed, or use a default one.
        print(f"GeminiAPIModel ready to use model: {self.model_name}")

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,  # Note: Gemini's GenerationConfig doesn't use top_k directly
            do_sample: bool = True,
            **kwargs: Any  # To accept any other parameters passed by run_two_agents
    ) -> str:
        """
        Generates a response from the Gemini API.

        :param messages: A list of message dictionaries.
        :param max_new_tokens: Maximum number of new tokens for the response.
        :param temperature: Sampling temperature.
        :param top_p: Nucleus sampling parameter.
        :param top_k: Top-k sampling (largely ignored by Gemini's basic config).
        :param do_sample: Whether to use sampling. Gemini uses temperature=0 for deterministic.
        :return: The generated text response.
        """

        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample and top_p > 0 else None,
            # top_k is not a direct parameter in Gemini's GenerationConfig
        )

        gemini_chat_history = []
        system_instruction_content = None

        # Separate system prompt from other messages
        processed_messages = []
        for msg in messages:
            if msg["role"] == "system":
                if system_instruction_content:  # Concatenate if multiple system prompts
                    system_instruction_content = f"{system_instruction_content}\n{msg['content']}"
                else:
                    system_instruction_content = msg['content']
            else:
                processed_messages.append(msg)

        # Initialize the model, potentially with a system instruction
        try:
            if system_instruction_content:
                model_instance = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction_content
                )
            else:
                model_instance = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Warning: Could not set system_instruction directly for {self.model_name}: {e}. "
                  "If a system prompt was provided, it will be prepended to the user message.")
            model_instance = genai.GenerativeModel(self.model_name)
            if system_instruction_content and processed_messages and processed_messages[0]['role'] == 'user':
                processed_messages[0]['content'] = f"{system_instruction_content}\n\n{processed_messages[0]['content']}"
            elif system_instruction_content:  # If no user message to prepend to, create one
                processed_messages.append({'role': 'user', 'content': system_instruction_content})

        # Convert remaining messages to Gemini's chat history format
        # Gemini uses 'user' and 'model' (for assistant responses)
        for msg in processed_messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_chat_history.append({'role': role, 'parts': [msg["content"]]})

        if not gemini_chat_history:
            # This might happen if only a system prompt was given and it wasn't prepended
            if system_instruction_content and not processed_messages:
                gemini_chat_history.append({'role': 'user', 'parts': [system_instruction_content]})
            else:
                print("Warning: No messages to send to Gemini after processing.")
                return "help"  # Fallback

        try:
            # For chat-like interactions, use start_chat and send_message
            if len(gemini_chat_history) > 1:
                # The history for start_chat should not include the last message
                chat_session = model_instance.start_chat(
                    history=[genai.types.Content(role=m['role'], parts=m['parts']) for m in gemini_chat_history[:-1]]
                )
                # Send the last message
                response = chat_session.send_message(
                    content=gemini_chat_history[-1]['parts'],
                    generation_config=generation_config
                )
            elif gemini_chat_history:  # Single message, use generate_content
                response = model_instance.generate_content(
                    contents=gemini_chat_history[0]['parts'],  # Content of the single message
                    generation_config=generation_config
                )
            else:  # Should be caught by the check above
                return "help"

            if response.text:
                return response.text.strip()
            # Fallback for some response structures if .text is not directly available
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return "".join(
                    part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')).strip()
            else:
                print(
                    f"Warning: Gemini API returned an empty or unexpected response structure for model {self.model_name}.")
                # For debugging: print(f"Full Gemini Response: {response}")
                return "help"  # Fallback action
        except Exception as e:
            print(f"Error calling Gemini API ({self.model_name}): {e}")
            # import traceback # For debugging
            # traceback.print_exc() # For debugging
            return "help"  # Fallback action

if __name__ == "__main__":
    checkpoint: str = "HuggingFaceTB/SmolLM-135M-Instruct"
    device: str = "cpu"  # or "cpu"

    llm: HFModel = SmollLLM(checkpoint, device=device)

    messages: List[Dict[str, str]] = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    response: str = llm.generate_response(
        messages,
        max_new_tokens=50,
        temperature=0.2,
        top_p=0.9,
        top_k=50,
        do_sample=True
    )

    print("\n===== Model Response =====")
    print(response)
