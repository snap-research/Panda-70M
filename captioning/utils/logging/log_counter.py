class LogCounter:
    """
    Utility class for logging at regular intervals
    """
    def __init__(self, frequency: int):
        self.frequency = frequency
        self.current_step = 0

    def is_logging_step(self) -> bool:
        return self.current_step % self.frequency == 0

    def step(self) -> bool:
        if self.frequency < 0:
            return False
        result = self.is_logging_step()
        if result:
            self.current_step = 0
        self.current_step += 1
        return result
