#!/usr/bin/python3

import time
from IPython.display import clear_output
from gridmap import GridMap
from environment import Environment
from dqn import Full_DQN
from itertools import count
import torch
import numpy as np


'''
# Need to update this to use joint_action 
def main():
    grid_map = GridMap(1, (7,7), 3, 3)
    env = Environment(grid_map)
    step_count = 0
    while True:
        finished = True
        for p in grid_map.passengers:
            if p.status != 'dropped':
              finished = False
        clear_output()
        grid_map.visualize()
        print('-'*10)
        env.step()
        # env.step(action)
        step_count += 1
        time.sleep(1)
        if finished:
            print('step cost:', step_count)
            break
'''

def main():
    # Initialize env
    grid_map = GridMap(1, (7,7), 3, 3)
    env = Environment(grid_map)
    num_cars = env.grid_map.num_cars
    num_passengers = env.grid_map.num_passengers

    Model = Full_DQN(env)

    episode_durations = []

    reward_sum = 0

    num_episodes= 50 # maybe increase this to about 300 later
    for i_episode in range(num_episodes):

        # Initialize the environment
        env.reset()

        for t in count():
            print('t: ' + str(t) )
            # Select an action for each car and accumulate into joint_action
            joint_action = []

            # Make this a function in Model
            #joint_action = Model.select_joint_action(env)

            for c in range(num_cars):
                state = Model.get_state(env.grid_map, c)
                action = Model.select_action(state,num_passengers, num_cars)
                joint_action.append(action)

            # State before you step, get rid of indicator
            state = Model.get_state(env.grid_map, 0)
            state[0] = 0

            next_state, reward, done, _ = env.step(joint_action, Model)

            reward_sum += reward
            print('reward sum: ' + str(reward_sum))

            #next_state = torch.tensor(next_state.T, dtype=torch.float)
            #state = torch.tensor(state.T, dtype=torch.float)

            #reward = torch.tensor([reward], device=device)

            # Store the transition in memory for each car
            for i in range(num_cars):
                action = joint_action[i]
                state_i = state
                state_i[i] = 1
                state_i_tensor = torch.tensor(state_i.T, dtype=torch.float)
                next_state_tensor = torch.tensor(next_state.T, dtype=torch.float)
                reward_tensor = torch.tensor(np.array([reward]), dtype=torch.float)
                Model.memory.push(state_i_tensor, action.T, next_state_tensor, reward_tensor)

            # Maybe change this to memory of single actions

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            Model.optimize_model()

            if done:
                episode_durations.append(t + 1)
                #plot_durations()
                break
        '''
        # Update the target network, copying all weights and biases in DQN
        if i_episode % Model.target_update == 0:
            Model.target_net.load_state_dict(Model.policy_net.state_dict())
        '''

    print('Complete')


if __name__ == '__main__':
    main()
