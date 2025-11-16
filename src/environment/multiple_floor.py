import single_floor as sf
class MultipleFloorEnv:
    def __init__(self, num_floors, floor_size, num_agents):
        self.floors = [sf.SingleFloorEnv(floor_size, num_agents) for _ in range(num_floors)]
        self.num_floors = num_floors

    def reset(self):
        for floor in self.floors:
            floor.reset()

    def step(self, actions):
        rewards = []
        for i, floor in enumerate(self.floors):
            reward = floor.step(actions[i])
            rewards.append(reward)
        return rewards

    def render(self):
        for i, floor in enumerate(self.floors):
            print(f"Floor {i+1}:")
            floor.render()