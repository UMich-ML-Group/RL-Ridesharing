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
from algorithm import *
from dqn import ReplayMemory, DQN
from q_mixer import QMixer
import matplotlib.pyplot as plt
import copy 


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))

class Agent:
    def __init__(self, env, input_size, output_size, hidden_size, max_cars = 10, max_passengers = 10, 
                 mix_hidden = 32, batch_size = 128, lr = 0.001, gamma = .999, eps_start = 0.9, 
                 eps_end = 0.05, eps_decay = 750,  replay_capacity = 10000, num_save = 200, num_episodes = 10000, 
                 mode="random", training = False, load_file = None):
        self.env = env
        self.orig_env = copy.deepcopy(env)
        self.grid_map = env.grid_map
        self.cars = env.grid_map.cars
        self.num_cars = len(self.cars)
        self.passengers = env.grid_map.passengers
        self.num_passengers = len(self.passengers)
        self.max_cars = max_cars
        self.max_passengers = max_passengers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.replay_capacity = replay_capacity
        self.num_episodes = num_episodes
        self.steps_done = 0
        self.lr = lr
        self.mode = mode
        self.num_save = num_save
        self.training = training
        self.algorithm = PairAlgorithm()
        self.episode_durations = []
        self.duration_matrix = np.zeros((self.max_passengers, self.max_cars))
        self.count_matrix = np.zeros((self.max_passengers, self.max_cars))
        self.loss_history = []
        
        self.memory = ReplayMemory(self.replay_capacity)
        
        self.device = torch.device("cpu")
        print("Device being used:", self.device)
        self.policy_net = DQN(self.input_size, self.output_size , self.hidden_size).to(self.device)
        
        self.params = list(self.policy_net.parameters())

        
        if self.mode == "qmix":
            self.mixer = QMixer(self.input_size, self.max_passengers, mix_hidden).to(self.device)
            self.params += list(self.mixer.parameters())
            
        
        if load_file:
            self.policy_net.load_state_dict(torch.load(load_file))
            if self.mode == "qmix": 
                self.mixer.load_state_dict(torch.load("mixer_" + load_file))
                self.mixer.eval()
            self.policy_net.eval()
            self.load_file = "Pretrained_" + load_file
            print("Checkpoint loaded")
        else:         
            self.load_file = self.mode + "_model_num_cars_" + str(self.num_cars) + "_num_passengers_" + str(self.num_passengers) + \
                    "_num_episodes_" + str(self.num_episodes) + "_hidden_size_" + str(self.hidden_size) + ".pth"
            
        self.optimizer = optim.RMSprop(self.params, lr = self.lr)
        #self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1500, gamma=0.1)

        

    def select_action(self,state):
        #Select action with epsilon greedy
        sample = random.random()
        
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
            
        print(eps_threshold)

        self.steps_done += 1
        
        if not self.training:
            eps_threshold = 0.0

        if sample > eps_threshold:
            # Choose best action
            with torch.no_grad():
                
                self.policy_net.eval() 
                action = self.policy_net(state).view(self.max_passengers, self.max_cars)[:, :self.num_cars].max(1)[1].view(1,self.max_passengers)
                action[0, self.num_passengers:] = self.max_cars
                return action

        else:
            #Choose random action
            action = torch.tensor([[random.randrange(self.num_cars) for car in range(self.max_passengers)]], device=self.device, dtype=torch.long)
            action[0, self.num_passengers:] = self.max_cars
            return action

            

    def random_action(self, state):
        return torch.tensor([[random.randrange(self.num_cars) for car in range(self.num_passengers)]], device=self.device, dtype=torch.long)
    
    
    def get_state(self):
        # Cars (px, py, 1=matched), Passengers(pickup_x, pickup_y, dest_x, dest_y, 1=matched)
        # Vector Size = 3*C + 5*P 
        cars = self.cars
        passengers = self.passengers
        indicator_cars_vec = np.zeros(self.max_cars)
        indicator_passengers_vec = np.zeros(self.max_passengers)

        # Encode information about cars
        cars_vec = np.array([0]*(2*self.max_cars))
        
        for i, car in enumerate(cars):    
            cars_vec[2*i: 2*i + 2]  = [car.position[0], car.position[1]]
            indicator_cars_vec[i] = 1
        

        # Encode information about passengers
        passengers_vec = np.array([0]*(4*self.max_passengers))
        for i, passenger in enumerate(passengers):
            passengers_vec[4*i: 4*i + 4]  = [passenger.pick_up_point[0], passenger.pick_up_point[1],
                                             passenger.drop_off_point[0],passenger.drop_off_point[1]]
            indicator_passengers_vec[i] = 1


        return torch.tensor(np.concatenate((cars_vec,indicator_cars_vec, passengers_vec, indicator_passengers_vec)), device= self.device, dtype=torch.float).unsqueeze(0)
    
    
    def train(self):
        
        duration_sum = 0.0
        
        for episode in range(self.num_episodes):
            
            self.reset_different_num()
            #self.reset() 
            #self.reset_orig_env()
            
            state = self.get_state() 
            
            if self.mode == "dqn" or self.mode == "qmix":
                action = self.select_action(state)

            elif self.mode == "random":
                action = self.random_action([state])
                
            elif self.mode == "greedy":
                action = [self.algorithm.greedy_fcfs(self.grid_map)]
                action = torch.tensor(action, device=self.device, dtype=torch.long)
                #print(action.size())
                #print(action[:,:self.num_passengers])

            
            reward, duration = self.env.step(action[:,:self.num_passengers], self.mode)

            if self.mode == "dqn":
                reward.extend([0]*(self.max_passengers-self.num_passengers))
                
            self.episode_durations.append(duration)
            count = self.count_matrix[self.num_passengers-1, self.num_cars-1] 
            self.duration_matrix[self.num_passengers-1, self.num_cars-1] = self.duration_matrix[self.num_passengers-1, self.num_cars-1] * (count / (count + 1)) + duration / (count + 1)
            self.count_matrix[self.num_passengers-1, self.num_cars-1] += 1
            duration_sum += duration
            
            if self.training:
                self.memory.push(state, action, torch.tensor(reward, device = self.device, dtype=torch.float).unsqueeze(0))  
                self.optimize_model()
                
                self.plot_durations(self.mode)
                self.plot_loss_history(self.mode)
             
                
            if self.training and episode % self.num_save == 0:
                torch.save(self.policy_net.state_dict(), "episode_" + str(episode) + "_" +self.load_file )
                if self.mode == "qmix":
                    torch.save(self.mixer.state_dict(), "mixer_episode_" + str(episode) + "_" +self.load_file)
                print("Checkpoint saved")
                
                    
            print("Episode: ", episode)

           
        if self.training:
            torch.save(self.policy_net.state_dict(), self.load_file )
            if self.mode == "qmix":
                torch.save(self.mixer.state_dict(), "mixer_" + self.load_file)
            print("Checkpoint saved")
            
        print("Average duration was ", duration_sum/self.num_episodes)
        print("Finished")  
        np.save("Duration_matrix", self.duration_matrix)
        np.save("Count_matrix" , self.count_matrix)
        print(self.duration_matrix)
        print(self.count_matrix)
            
    def reset(self):
        
        self.env.reset()
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        
    def reset_different_num(self):
        
        self.env.grid_map.cars = []
        self.env.grid_map.passengers = []
        self.env.grid_map.num_passengers = random.randint(1,self.max_passengers)
        self.env.grid_map.num_cars = random.randint(1,self.max_cars)
        self.env.grid_map.add_passenger(self.env.grid_map.num_passengers)
        self.env.grid_map.add_cars(self.env.grid_map.num_cars) 
        
        self.grid_map = self.env.grid_map
        self.num_passengers = self.env.grid_map.num_passengers
        self.num_cars = self.env.grid_map.num_cars
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
        batch = Transition(*zip(*transitions))
    
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        self.policy_net.train()
        
        q_values = self.policy_net(state_batch).view(self.batch_size, self.max_passengers, self.max_cars)
        q_values = torch.cat((q_values, torch.zeros((self.batch_size, self.max_passengers, 1), device = self.device)), 2)
        state_action_values = q_values.gather(2,action_batch.unsqueeze(2)).squeeze()

        
        # Compute the expected Q values
        expected_state_action_values = reward_batch
        

        # Compute Huber loss
        if self.mode == "dqn":
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        elif self.mode == "qmix":
            self.mixer.train()
            chosen_action_qvals = self.mixer(state_action_values, state_batch)
            loss = F.smooth_l1_loss(chosen_action_qvals, reward_batch.view(-1, 1, 1))
            #loss = F.mse_loss(chosen_action_qvals, reward_batch.view(-1, 1, 1))


        self.loss_history.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def plot_durations(self, filename):
        print("Saving durations plot ...")
        plt.figure(2)
        plt.clf()

        total_steps = np.array(self.episode_durations)

        N = len(total_steps)
        window_size = 200
        if N < window_size:
            total_steps_smoothed = total_steps
        else:
            total_steps_smoothed = np.zeros(N-window_size)

            for i in range(N-window_size):
                window_steps = total_steps[i:i+window_size]
                total_steps_smoothed[i] = np.average(window_steps)

        plt.title('Episode Duration history')
        plt.xlabel('Episode')
        plt.ylabel('Duration')

        plt.plot(total_steps_smoothed)
        np.save("Duration_"+filename, total_steps_smoothed)
        #plt.savefig("Durations_history_" + filename)
        
    def plot_loss_history(self, filename):
        print("Saving loss history ...")
        plt.figure(2)
        plt.clf()
        #loss = torch.tensor(self.loss_history, dtype=torch.float)

        total_loss = np.array(self.loss_history)

        N = len(total_loss)
        window_size = 50
        if N < window_size:
            total_loss_smoothed = total_loss
        else:
            total_loss_smoothed = np.zeros(N-window_size)

            for i in range(N-window_size):
                window_steps = total_loss[i:i+window_size]
                total_loss_smoothed[i] = np.average(window_steps)


        plt.title('Loss history')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.plot(self.loss_history)
        np.save("Loss_"+filename, total_loss_smoothed)
        #plt.savefig("Loss_history_" + filename)

if __name__ == '__main__':
    init_cars = 2
    init_passengers = 7
    max_cars = 20
    max_passengers = 20
    
    
    
    grid_map = GridMap(1, (500,500), init_cars, init_passengers)
    cars = grid_map.cars
    passengers = grid_map.passengers
    env = Environment(grid_map)


    input_size = 3*max_cars + 5*max_passengers # cars (px, py), passengers(pickup_x, pickup_y, dest_x, dest_y)
    output_size = max_cars * max_passengers  # num_cars * (num_passengers + 1)
    hidden_size = 512
    #load_file = "episode_50000_dqn_model_num_cars_2_num_passengers_7_num_episodes_100000_hidden_size_512.pth"
    load_file = None
    #greedy, random, dqn, qmix
    agent = Agent(env, input_size, output_size, hidden_size, max_cars=max_cars, max_passengers = max_passengers, 
                  load_file = load_file, lr=0.001, mix_hidden = 64, batch_size=128, eps_decay = 20000, num_episodes=1000, 
                  mode = "dqn", training = True)
    agent.train()
    
    # qmix= np.load("Duration_matrix_qmix.npy")
    # greedy = np.load("Duration_matrix_greedy.npy")
    # random = np.load("Duration_matrix_random.npy")
    
    
    # # qmix_count = np.load("Count_matrix_qmix.npy")
    # greedy_count = np.load("Count_matrix_greedy.npy")
    # qmix_count = np.load("Count_matrix_qmix.npy")
    # random_count = np.load("Count_matrix_random.npy")
    # print(np.sum(greedy*greedy_count)/10000)
    # print(np.sum(qmix*qmix_count)/10000)
    # print(np.sum(random*random_count)/10000)
    
