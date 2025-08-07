class MAPFSearchResult:
    def __init__(self):
        self.paths = {}  # actor_id -> list of Node/Point
        self.success = False
        self.runtime = 0.0
        self.steps = 0
        self.stats = {}

    def add_path(self, actor_id, path):
        self.paths[actor_id] = path 