import random
import string
from game.modules.module import Module, ActionResult


class SimonSaysModule(Module):
    def __init__(self):
        super().__init__()
        self.colors = ["red", "blue", "green", "yellow"]
        self.sequence = []
        self.current_round = 0
        self.max_rounds = 5
        self.serial_number = self._generate_serial_number()
        self.has_vowel = any(c in "aeiou" for c in self.serial_number.lower())
        self.user_sequence = []
        self.generate_sequence()

    def _generate_serial_number(self) -> str:
        """Generate a random serial number with letters and at least one digit."""
        # Generate 5 random letters
        letters = ''.join(random.choices(string.ascii_uppercase, k=5))
        # Generate at least one digit
        digit = str(random.randint(0, 9))
        # Insert the digit at a random position
        position = random.randint(0, len(letters))
        return letters[:position] + digit + letters[position:]

    def generate_sequence(self):
        """Generate a random sequence of colors."""
        self.sequence = [random.choice(self.colors) for _ in range(self.max_rounds)]

    def get_color_mapping(self, color: str, index: int) -> str:
        """Get the mapped color based on the serial number and current round."""
        # Mappings for serial numbers with vowels
        vowel_mappings = [
            # Round 1
            {
                "red": "blue",
                "blue": "red",
                "green": "yellow",
                "yellow": "green"
            },
            # Round 2
            {
                "red": "yellow",
                "blue": "green",
                "green": "blue",
                "yellow": "red"
            },
            # Round 3
            {
                "red": "green",
                "blue": "red",
                "green": "yellow",
                "yellow": "blue"
            },
            # Round 4
            {
                "red": "red",
                "blue": "blue",
                "green": "green",
                "yellow": "yellow"
            },
            # Round 5
            {
                "red": "yellow",
                "blue": "green",
                "green": "red",
                "yellow": "blue"
            }
        ]

        # Mappings for serial numbers without vowels
        no_vowel_mappings = [
            # Round 1
            {
                "red": "blue",
                "blue": "yellow",
                "green": "green",
                "yellow": "red"
            },
            # Round 2
            {
                "red": "red",
                "blue": "blue",
                "green": "yellow",
                "yellow": "green"
            },
            # Round 3
            {
                "red": "yellow",
                "blue": "green",
                "green": "blue",
                "yellow": "red"
            },
            # Round 4
            {
                "red": "green",
                "blue": "red",
                "green": "red",
                "yellow": "blue"
            },
            # Round 5
            {
                "red": "blue",
                "blue": "green",
                "green": "yellow",
                "yellow": "green"
            }
        ]

        if self.has_vowel:
            mapping = vowel_mappings[index]
        else:
            mapping = no_vowel_mappings[index]

        return mapping[color]

    def instruction(self) -> str:
        """Return the instruction manual for this module."""
        manual = """## Simon Says Module

This module presents a sequence of flashing colored lights that the defuser must repeat in a specific order.
The sequence gets longer with each successful round. The correct buttons to press depend on the colors shown,
the serial number, and the current round number.

Serial Number Rules:
- If the serial number contains a vowel (A, E, I, O, U):

  | Color Flashed | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
  |---------------|---------|---------|---------|---------|---------|
  | Red           | Blue    | Yellow  | Green   | Red     | Yellow  |
  | Blue          | Red     | Green   | Red     | Blue    | Green   |
  | Green         | Yellow  | Blue    | Yellow  | Green   | Red     |
  | Yellow        | Green   | Red     | Blue    | Yellow  | Blue    |

- If the serial number does NOT contain a vowel:

  | Color Flashed | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |
  |---------------|---------|---------|---------|---------|---------|
  | Red           | Blue    | Red     | Yellow  | Green   | Blue    |
  | Blue          | Yellow  | Blue    | Green   | Red     | Green   |
  | Green         | Green   | Yellow  | Blue    | Red     | Yellow  |
  | Yellow        | Red     | Green   | Red     | Blue    | Green   |

For each color that flashes, tell the defuser which color button to press according to the tables above.
The sequence will get longer with each successful round. If the defuser makes a mistake, the module will
explode immediately."""
        return manual

    def _get_state(self) -> tuple[str, list[str]]:
        """Return the current state and available actions."""
        if self.current_round >= self.max_rounds:
            return "Module disarmed!", []

        # If we're waiting for the user to start the sequence
        if len(self.user_sequence) == 0:
            state_desc = "Simon show you a sequence of colors. Repeat the sequence by pressing the buttons.\n"
            state_desc += f"Serial number: {self.serial_number}\n"
            state_desc += f"Round: {self.current_round + 1}/{self.max_rounds}\n"
            state_desc += "Flashing sequence: " + ", ".join(self.sequence[:self.current_round + 1]) + "\n"
            state_desc += "Press a colored button to start sequence."

            actions = [f"press {color}" for color in self.colors]
            return state_desc, actions
        else:
            # If we're waiting for user input next in the sequence
            current_sequence = self.sequence[:self.current_round + 1]
            state_desc = "Continue the sequence by pressing the next colored button.\n"
            state_desc += f"Serial number: {self.serial_number}\n"
            state_desc += f"Round: {self.current_round + 1}/{self.max_rounds}\n"
            state_desc += "Flashing sequence: " + ", ".join(current_sequence) + "\n"
            state_desc += f"Your inputs so far: {', '.join(self.user_sequence)}\n"
            state_desc += "Press a colored button to continue."

            actions = [f"press {color}" for color in self.colors]
            return state_desc, actions

    def _do_action(self, action: str) -> ActionResult:
        """Perform the specified action."""
        try:
            color = action.lower().replace("press ", "").strip()
            if color not in self.colors:
                return ActionResult.INCORRECT

            self.user_sequence.append(color)

            # Check if the user has completed the current round's sequence
            for i, color in enumerate(self.sequence[:len(self.user_sequence)]):
                if self.get_color_mapping(color, i) != self.user_sequence[i]:
                    print("Correct mapping is: ", self.get_color_mapping(color, i))
                    return ActionResult.EXPLODED

            if len(self.user_sequence) == self.current_round + 1:
                # Correct sequence
                self.current_round += 1
                self.user_sequence = []

            if self.current_round + 1 >= self.max_rounds:
                self.set_disarmed()
                return ActionResult.DISARMED

            return ActionResult.CHANGED

        except (ValueError, IndexError):
            return ActionResult.INCORRECT
