import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
from gridmap import GridMap


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    # TODO implement replay buffer

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trainsition):
        #Saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    # TODO implement q network
    def __init__(self, inputs, outputs, hidden):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden,outputs)
        #self.bn2 = nn.BatchNorm1d(output)

    def forward(self, x):
        # TODO implement train
        x2 = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(x2)
        return out


class Full_DQN:
    def __init__(self):
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.replay_capacity = 10000
        # Sizes to be determined
        self.state_size = 1
        self.action_size = 1
        self.hidden_size = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.memory = ReplayMemory(self.replay_capacity)
        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0

    def select_action(self,state,num_passengers, num_cars):
        # Possible Action size = Num_passengers 0
        # make list of free passengers
        # pick randomly from this list

        # if car does not have passenger
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:

            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def get_state(self, grid_map, car_index):
        # We are going to stack everything as a vector
        # Indicators, cars (px, py, 1=matched), passengers(pickup_x, pickup_y, dest_x, dest_y, 1=matched)
        # Vector Size = (C + 3*C + 5*P, 1) = (4*C + 5*P, 1)
        cars = grid_map.cars
        passengers = grid_map.passengers

        # Indicator for Which car is now free
        indicator = np.zeros(len(cars))
        indicator[car_index] = 1

        # Encode information about cars
        cars_vector = np.zeros(3*len(cars))
        for i in range(len(cars)):
            cars_vector[3*i]   = cars[i].position[0]
            cars_vector[3*i+1] = cars[i].position[1]
            cars_vector[3*i+2] = 1 if cars[i].passenger is not None else 0

        # Encode information about passengers
        passengers_vector = np.zeros(5*len(passengers))
        for i in range(len(passengers)):
            passengers_vector[5*i]   = passengers[i].pick_up_point[0]
            passengers_vector[5*i+1] = passengers[i].pick_up_point[1]
            passengers_vector[5*i+2] = passengers[i].drop_off_point[0]
            passengers_vector[5*i+3] = passengers[i].drop_off_point[1]
            passengers_vector[5*i+4] = 1 if passengers[i].status != 'wait_pair' else 0

        return np.concatenate((indicator, cars_vector, passengers_vector))


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


if __name__ == '__main__':
    m = GridMap(0, (10,10), 3, 3)
    print(m)
    #print('path from (0,0) to (5,5):')
    #path = m.plan_path((0,0),(5,5))
    #sprint(path)
    #m.visualize()

    Model = Full_DQN()
    s = Model.get_state(m, 1)
    print(s)