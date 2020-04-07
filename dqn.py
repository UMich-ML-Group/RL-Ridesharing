import numpy as np
import random
import math
#from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gridmap import GridMap


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    # TODO implement replay buffer

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
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


class Full_DQN():
    def __init__(self, env):
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.replay_capacity = 10000
        # Sizes to be determined
        self.state_size = 4*env.grid_map.num_cars + 5*(env.grid_map.num_passengers)
        self.action_size = env.grid_map.num_passengers+1
        self.hidden_size = 200

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.memory = ReplayMemory(self.replay_capacity)
        self.policy_net = DQN(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

        self.steps_done = 0

    def select_action(self,state,num_passengers, num_cars):
        # make list of free passengers
        # pick randomly from this list    

        # Free passenger List
        # [0...P-1] are passengers, P is do nothing
        free_passenger_idx = [num_passengers] # -1 is for do nothing
        for i in range(num_passengers):
            if not state[4*num_cars + 5*i+4]: # free if not matched
                free_passenger_idx.append(i)

        # Possible Action size = Num_passengers + 1 (do nothing)

        # if car does not have passenger
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1 # might not want to change this here or maybe += 1/C
        if sample > eps_threshold:
            with torch.no_grad():
                # We should check that the passenger is free for the car we are matching
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward. 

                # out.shape = (P+1,C)
                #out = self.policy_net(torch.tensor(state, device=self.device)) #, dtype=torch.long))
                self.policy_net.eval()
                out = self.policy_net(torch.tensor(state.T, device=self.device, dtype=torch.float))

                # Choose maximum value out of the free passengers or do nothing (idx P)
                max_val = -9999999999
                for i in free_passenger_idx:
                    val = out[0,i]
                    if val > max_val:
                        max_index = i
                        max_val = val
    
                action = np.zeros((num_passengers+1,1))
                action[max_index] = 1

                return torch.tensor(action, device=self.device, dtype=torch.float)
        else:
            # Now we have a list of free passengers
            idx = random.choice(free_passenger_idx)

            # Form action output as OHE
            # Last OHE index == 1 means do nothing
            action = np.zeros((num_passengers+1,1))
            action[idx] = 1

            return torch.tensor(action, device=self.device, dtype=torch.float)

            #return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

    def get_state(self, grid_map, car_index):
        # We are going to stack everything as a vector
        # Indicators, cars (px, py, 1=matched), passengers(pickup_x, pickup_y, dest_x, dest_y, 1=matched)
        # Vector Size = (C + 3*C + 5*P, 1) = (4*C + 5*P, 1)
        cars = grid_map.cars
        passengers = grid_map.passengers

        # Indicator for Which car is now free
        indicator = np.zeros((len(cars),1))
        indicator[car_index] = 1

        # Encode information about cars
        cars_vector = np.zeros((3*len(cars),1))
        for i in range(len(cars)):
            cars_vector[3*i]   = cars[i].position[0]
            cars_vector[3*i+1] = cars[i].position[1]
            cars_vector[3*i+2] = 1 if cars[i].passenger is not None else 0

        # Encode information about passengers
        passengers_vector = np.zeros((5*len(passengers),1))
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

    '''
    def step(self, grid_map):
        cars = grid_map.cars
        for c in cars:


        # TODO implement a step
        curr_state = self.get_state(grid_map)
        curr_action = self.get_action(curr_state)
        reward, next_state = self.set_action(curr_action, grid_map)

        # TODO update replay buffer
        self.buffer.push((curr_state, curr_state, reward, next_state))

        # TODO update/train q network
        self.update_network.train()
    '''


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state).float()
        action_batch = torch.cat(batch.action).long()
        reward_batch = torch.cat(batch.reward).float()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        #print(loss)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


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

    num_episodes = 50

