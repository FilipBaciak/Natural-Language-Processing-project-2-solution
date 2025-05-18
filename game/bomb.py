from game.modules.regular_wires_module import RegularWiresModule
from game.modules.button_module import ButtonModule
from game.modules.memory_module import MemoryModule
from game.modules.simon_says_module import SimonSaysModule
from game.modules.module import ActionResult


class Bomb:
    def __init__(self):
        self.modules = [RegularWiresModule(), ButtonModule(), SimonSaysModule(), MemoryModule()]
        self.current_module = 0
        self.exploded = False
        self.disarmed = False

    def explode(self):
        self.exploded = True

    def disarm(self):
        self.disarmed = True

    def do_action(self, action: str) -> ActionResult:
        if self.exploded:
            return ActionResult.EXPLODED
        if self.disarmed:
            return ActionResult.DISARMED

        result = self.modules[self.current_module].do_action(action)

        if result == ActionResult.DISARMED:
            self.current_module += 1
            if self.current_module >= len(self.modules):
                self.disarm()
                return ActionResult.DISARMED

            return ActionResult.CHANGED

        if result == ActionResult.EXPLODED:
            self.explode()
            return ActionResult.EXPLODED

        return result

    def state(self) -> tuple[str, list[str]]:
        if self.exploded:
            return "Bomb exploded!", []
        if self.disarmed:
            return "Bomb disarmed!", []

        return self.modules[self.current_module].state()