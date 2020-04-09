import numpy as np
from algorithm import PairAlgorithm
import time

class Environment:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.prev_num_cars_idle = self.grid_map.num_cars
        #self.algorithm = PairAlgorithm()
        self.num_total_steps = 0

    def reset(self):
        self.grid_map.reset()
        self.prev_num_cars_idle = self.grid_map.num_cars+1#+1 to get it to start simulating
        self.num_total_steps = 0

    # Get number of idle cars in env
    def get_number_idle_cars(self):
        num_idle = 0
        for car in self.grid_map.cars:
            if car.status == 'idle':
                #print('car idle')
                num_idle += 1
            #else:
                #print('car not idle')
        return num_idle

    def get_passengers_dropped(self):
        passengers_dropped = []
        for p in self.grid_map.passengers:
            if p.status == 'dropped':
                passengers_dropped.append(p)
        return passengers_dropped


    # implement this for joint action
    def step(self,joint_action, Model):
        info = None 
        num_cars = self.grid_map.num_cars
        num_passengers = self.grid_map.num_passengers

        # Get passengers dropped before simulating
        passengers_dropped = self.get_passengers_dropped()

        # Joint_action is List of length c with each elem vec of length (p+1)
        # Pairing Step with action
        for i in range(num_cars):
            action = joint_action[i]
            c = self.grid_map.cars[i]
            p_idx = np.argmax(action)

            # If action is not 'do nothing' then pair if car is free and passenger not paired yet
            if p_idx != num_passengers and c.status == 'idle': # need to check if passenger is already car
                p = self.grid_map.passengers[p_idx]
                if p.status == 'wait_pair':
                    c.pair_passenger(p)
                    pick_up_path = self.grid_map.plan_path(c.position, p.pick_up_point)
                    drop_off_path = self.grid_map.plan_path(p.pick_up_point, p.drop_off_point)
                    c.assign_path(pick_up_path, drop_off_path)

        # Simulation step       
        # If need to simulate, Simulate (simulator_step) until the next car changes to idle
        num_cars_idle = self.get_number_idle_cars()
        reward = 0
        done = False
        info = {}
        sim_count = 0
        # Keep running simulate step
        while True:
            episode_done = True
            has_passenger = False
            has_car = False
            for p in self.grid_map.passengers:
                if p.status != 'dropped':
                    episode_done = False
                if p.status == 'wait_pair':
                    has_passenger = True
            for c in self.grid_map.cars:
                if c.status == 'idle':
                    has_car = True

            if episode_done or (has_passenger and has_car):
                break

            # Simulator Step and update params
            #print(sim_count)
            self.simulator_step()
            sim_count+=1

        # Finished simulation steps
        # Get Reward from number of passenger waiting steps if passengers finished trip
        # Get passengers dropped after simulating
        new_passengers_dropped = self.get_passengers_dropped()
        gamma = .005
        for p in new_passengers_dropped:
            if p not in passengers_dropped:
                reward += 1 - (gamma * p.waiting_steps)

        obs = Model.get_state(self.grid_map, 0)
        # Set indicator to all 0's by turning first element to 0
        obs[0] = 0

        info['sim_count'] = sim_count
        return obs, reward, episode_done, info

    # Small step
    def simulator_step(self):
        for passenger in self.grid_map.passengers:
            if passenger.status == 'wait_pair' or passenger.status == 'wait_pick':
                passenger.waiting_steps += 1

        # move car according to status
        for car in self.grid_map.cars:
            if car.status == 'idle':
                continue

            # init require step
            if car.required_steps is None:  # init
                car.required_steps = self.grid_map.map_cost[(car.position, car.path[0])]

            # pick up or drop off will take one step
            if car.status == 'picking_up' and car.position == car.passenger.pick_up_point: # picking up
                car.pick_passenger()
            elif car.status == 'dropping_off' and car.position == car.passenger.drop_off_point:  # dropping off
                car.drop_passenger()
            else:
                # try to move
                if car.required_steps > 0:  # decrease steps
                    car.required_steps -= 1
                elif car.required_steps == 0: # move
                    car.move()
                    if car.path:
                        car.required_steps = self.grid_map.map_cost[(car.position, car.path[0])]

