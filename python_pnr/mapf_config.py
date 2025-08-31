class MAPFConfig:
    def __init__(self, max_steps=1000, timeout=10000, heuristic_weight=1.0):
        self.max_steps = max_steps
        self.timeout = timeout
        self.heuristic_weight = heuristic_weight 