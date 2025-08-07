from .actor import Actor

class ActorSet:
    def __init__(self):
        self.actors = []

    def add_actor(self, actor: Actor):
        self.actors.append(actor)

    def get_actor_by_id(self, actor_id):
        """根据actor的id查找actor"""
        for actor in self.actors:
            if actor.id == actor_id:
                return actor
        return None

    def __iter__(self):
        return iter(self.actors)

    def __len__(self):
        return len(self.actors)

    def __getitem__(self, idx):
        return self.actors[idx]

    def clear(self):
        self.actors.clear() 