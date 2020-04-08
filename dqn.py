import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.optimize
from collections import namedtuple
from itertools import count
from environment import *
from gridmap import GridMap
import matplotlib.pyplot as plt
import copy 



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
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


class DQN_Agent:
    def __init__(self, env, input_size, output_size, hidden_size, batch_size = 128, lr = 0.001, gamma = .999, eps_start = 0.9, 
                 eps_end = 0.05, eps_decay = 200, target_update = 300, replay_capacity = 10000, num_episodes = 2000):
        self.env = env
        self.orig_env = copy.deepcopy(env)
        self.grid_map = env.grid_map
        self.cars = env.grid_map.cars
        self.num_cars = len(self.cars)
        self.passengers = env.grid_map.passengers
        self.num_passengers = len(self.passengers)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.replay_capacity = replay_capacity
        self.num_episodes = num_episodes
        self.steps_done = 0
        self.lr = lr
        self.episode_durations = []
        self.loss_history = []
        
        self.memory = ReplayMemory(self.replay_capacity)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.input_size, self.output_size , self.hidden_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size, self.hidden_size).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        #self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr = self.lr)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.steps_done = 0
        
    def get_assignment(self, state, q_values = None, rand = False):
        # Given a set of q_values for every car and every posible invidual action, use the hungarian method 
        # to find the best set of actionsor or choose random actions given the constraints of the enviroment 
        
        action = np.zeros(self.num_cars)

        cars_status = list(state[0][2:3*self.num_cars:3])
        pass_status = list(state[0][3*self.num_cars+4::5])

        pass_to_remove = [i for i,passenger in enumerate(pass_status) if passenger != 1]
        pass_to_match = [i for i in range(self.num_passengers) if i not in pass_to_remove]

          
        if rand:
            cars_to_remove = [i for i,car in enumerate(cars_status) if car != 1] 
            cars_to_match = [i for i in range(self.num_cars) if i not in cars_to_remove]
            
            assignments = random.sample(range(len(pass_to_match)+1),len(cars_to_match))
            
            for i, assig in enumerate(assignments):
                if assig == len(pass_to_match):
                    action[cars_to_match[i]] = self.num_passengers
                else:
                    action[cars_to_match[i]] = pass_to_match[assig]
        else:
            cars_to_remove = [i for i,car in enumerate(cars_status) if (car != 1 or np.amax(q_values[i]) == q_values[i,-1])]  
            cars_to_match = [i for i in range(self.num_cars) if i not in cars_to_remove]
            
            q_filt = q_values[:, :-1]
            q_filt= np.delete(q_filt, cars_to_remove, 0)
            q_filt = np.delete(q_filt, pass_to_remove, 1)
            
            row_idx, col_idx = scipy.optimize.linear_sum_assignment(-q_filt)
            
            
            for i, row in enumerate(row_idx):
                action[cars_to_match[row]] = pass_to_match[col_idx[i]]
                
            for i in range(len(cars_to_match)):
                if i not in row_idx:
                    action[cars_to_match[i]] = self.num_passengers
                    
        action[cars_to_remove] = self.num_passengers #do nothing
        return action.astype(int)
        
        

    def select_action(self,state):
        #Select action with epsilon greedy
              
        sample = random.random()
        
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        state = torch.tensor(state, device = self.device, dtype=torch.float).unsqueeze(0)
        # ERASE
        # with torch.no_grad():
        #     self.policy_net.eval() 
        #     state_high = [1., 2., 0., 1., 4., 0., 2., 0.]
        #     state_high = torch.tensor(state_high, device = self.device, dtype=torch.float).unsqueeze(0)
        #     q_values_high = self.policy_net(state_high).cpu().numpy().reshape(self.num_cars, self.num_passengers + 1) #ERASE
        #     print("Q_values_high: ",q_values_high) #ERASE
        
        if sample > eps_threshold:
            # Choose best action
            with torch.no_grad():
                
                self.policy_net.eval() # not sure if right
                q_values = self.policy_net(state).cpu().numpy().reshape(self.num_cars, self.num_passengers + 1) 

                return self.get_assignment(state, q_values)
        else:
            #Choose random action
            return self.get_assignment(state, rand = True)
            


    def random_action(self, state):
        return self.get_assignment(state, rand = True)
    
    def get_state(self):
        # Cars (px, py, 1=matched), Passengers(pickup_x, pickup_y, dest_x, dest_y, 1=matched)
        # Vector Size = 3*C + 5*P 
        cars = self.cars
        passengers = self.passengers

        # Encode information about cars
        cars_vec = np.zeros(3*len(cars))
        
        for i, car in enumerate(cars):    
            cars_vec[3*i: 3*i + 3]  = [car.position[0], car.position[1], 1 if car.status == "idle"  else 0]

        # Encode information about passengers
        passengers_vec = np.zeros(5*len(passengers))
        for i, passenger in enumerate(passengers):
            passengers_vec[5*i: 5*i + 5]  = [passenger.pick_up_point[0], passenger.pick_up_point[1],
                                             passenger.drop_off_point[0],passenger.drop_off_point[1],  
                                             1 if passenger.status == 'wait_pair' else 0]

        return np.concatenate((cars_vec, passengers_vec))
    
    
    def train(self):
        
        
        for episode in range(self.num_episodes):
            #self.reset() 
            self.reset_orig_env()
            state = self.get_state()

            for t in count():
                
                    
                action = self.select_action(state)
                #action = self.random_action([state])
                reward, done = self.env.dqn_step(action)
                # if (state[2] == 1):
                #     print("Action: ", action)
                #     print("Step: ", t)
                #     input("Press enter to step")
                
                    
                    
                #print("Action: ", action)
                #print("Reward: ", reward)
                #print("Done: ", done)
                # #print(self.grid_map)
                # print("Step: ", t)
                #print("State :", state)
                #input("Press enter to step")
                #self.grid_map.visualize()
                

                next_state = self.get_state()
                    
                 
                               
                self.memory.push(torch.tensor(state, device = self.device, dtype=torch.float).unsqueeze(0), 
                                 torch.tensor(action, device = self.device, dtype=torch.long).unsqueeze(0), 
                                 torch.tensor(next_state, device = self.device, dtype=torch.float).unsqueeze(0) if not done else None, 
                                 torch.tensor(reward, device = self.device, dtype=torch.float).unsqueeze(0) )  
                
                # Move to the next state 
                
                state = next_state  
                
                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    self.plot_loss_history()
                    break         
            
            # Update the target network, copying all weights and biases in DQN
            print("Episode: ", episode)
            if episode % self.target_update == 0:
                print("Target_net_updated")
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
        print("Finished")  
            
    def reset(self):
        self.env.reset()
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        
    def reset_orig_env(self):

        self.env = copy.deepcopy(self.orig_env)
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        self.grid_map.init_zero_map_cost()

        
        
           

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
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

       
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        #self.policy_net.train()
        #self.target_net.train()
        #state_action_values = self.policy_net(state_batch).view(self.batch_size, self.num_cars, self.num_passengers + 1).gather(2, action_batch.unsqueeze(2)).squeeze()
        state_action_values = self.policy_net(state_batch).gather(1,action_batch)

        #state_action_values = state_action_values# .unsqueeze(1)
        #print(self.policy_net(state_batch).view(self.batch_size, self.num_cars, self.num_passengers + 1))
        #print(action_batch.unsqueeze(2))
        #print(state_action_values)
        
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #self.target_net.train()
        
        #next_state_action_values = torch.zeros((self.batch_size, self.num_cars), device=self.device)
        next_state_action_values = torch.zeros((self.batch_size, self.num_cars), device=self.device)
        next_q_values = self.target_net(non_final_next_states).view(non_final_next_states.size()[0], self.num_cars, self.num_passengers + 1) 
        #print(self.target_net(non_final_next_states).size())
        #print(self.target_net(non_final_next_states).max(1)[0].detach().size())
        #print(self.target_net(non_final_next_states).max(1)[0])
        
        non_final_mask = non_final_mask.unsqueeze(1)

        
        #next_state_action_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() # erase


        
        next_action_batch = self.get_best_batch_actions(non_final_next_states, next_q_values).squeeze()
        #next_action_batch = next_action_batch.unsqueeze(1) # CHANGE LATER
        # non_final_mask = non_final_mask.unsqueeze(1) # CHANGE LATER

        next_state_action_values[non_final_mask] = next_q_values.gather(2, next_action_batch.view((-1,1,1))).squeeze().detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_action_values * self.gamma) + reward_batch

        # Compute Huber loss
        #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        #print(state_batch)
        #print(reward_batch)
        

        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        #for name, param in self.policy_net.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        #print(loss)

    def get_best_batch_actions(self, states, q_values):
        actions = np.zeros((q_values.size()[0], self.num_cars))
        for i,(state,q) in enumerate(zip(states,q_values)):
            q_value = q.detach().cpu().numpy().reshape(self.num_cars, self.num_passengers + 1) 
            actions[i] = self.get_assignment(state.unsqueeze(0), q_value)      
        return torch.tensor(actions, device = self.device, dtype=torch.long)

    def plot_durations(self):
        print("Saving durations plot ...")
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Episode Duration history')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        plt.savefig("Durations_history")
        
    def plot_loss_history(self):
        print("Saving loss history ...")
        plt.figure(2)
        plt.clf()
        loss = torch.tensor(self.loss_history, dtype=torch.float)
        plt.title('Loss history')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.plot(self.loss_history)
        plt.savefig("Loss_history")

if __name__ == '__main__':
    num_cars = 1
    num_passengers = 3
    
    grid_map = GridMap(1, (5,5), num_cars, num_passengers)
    cars = grid_map.cars
    passengers = grid_map.passengers
    env = Environment(grid_map)
    
    #print('path from (0,0) to (5,5):')
    #path = m.plan_path((0,0),(5,5))
    #sprint(path)
    #m.visualize()
    input_size = 3*num_cars + 5*num_passengers # cars (px, py, 1=matched), passengers(pickup_x, pickup_y, dest_x, dest_y, 1=matched)
    output_size = num_cars * (num_passengers + 1) # num_cars * (num_passengers + 1)
    hidden_size = 100
    agent = DQN_Agent(env, input_size, output_size, hidden_size)
    agent.train()