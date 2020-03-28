
class ReplayBuffer:
    # TODO implement replay buffer

    def __init__(self):
        self.buffer = []

    def push(self, trainsition):
        #TODO implement push buffer
        pass

class QNetowrk:
    # TODO implement q network
    def __init__(self, replay_buffer):
        self.buffer = replay_buffer

    def train(self):
        # TODO implement train
        pass

class DQN:
    def __init__(self):
        self.buffer = ReplayBuffer()
        self.update_network = QNetowrk(self.buffer)
        self.policy_network = QNetowrk(None)

    def get_state(self, grid_map):
        # TODO convert map to state vector and return
        pass

    def get_action(self, state):
        # TODO get action vector from current state using QNetwork
        pass

    def set_action(self, action, grid_map):
        # TODO input action vector and set pair result to map
        # TODO return reward and next_state
        pass

    def step(self, grid_map):
        # TODO implement a step
        curr_state = self.get_state(grid_map)
        curr_action = self.get_action(curr_state)
        reward, next_state = self.set_action(curr_action, grid_map)

        # TODO update replay buffer
        self.buffer.push((curr_state, curr_state, reward, next_state))

        # TODO update/train q network
        self.update_network.train()


