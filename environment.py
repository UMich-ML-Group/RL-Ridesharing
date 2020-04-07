import numpy as np
from algorithm import PairAlgorithm

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


    '''
    def step(self):
        # pairing
        # Status changed here
        self.algorithm.greedy_fcfs(self.grid_map)
        # TODO switch to reinforcement algorithm
        self.algorithm.deep_q_learning(self.grid_map)

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
    '''

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
        #print('step')
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
        #print('num idle: ' + str(num_cars_idle))
        #need_to_simulate = num_cars_idle != self.prev_num_cars_idle # Is this right
        need_to_simulate=True
        # 2 scenarios.
        # 1. Don't need any simulation step
        # 2. We need simulation step - at least 1 pass hasnt been served and at least 1 available car
        
        # 

        #need_to_simulate = num_cars_idle == 0 # not right
        reward = 0
        done = False
        info = None

        # Check if simulation needed at all
        if need_to_simulate:
            sim_count = 0
            
            #passengers_finishing = []

            # While no cars finish trips, simulate
            # while num_cars_idle == self.prev_num_cars_idle:
            # while num_cars_idle == 0: # not right

            need_to_simulate = True

            # Keep running simulate step
            #while num_cars_idle == self.prev_num_cars_idle:
            while need_to_simulate:

                # If any passengers are about to finish, keep track of them for reward
                '''
                for c in self.grid_map.cars:
                    if c.passenger is not None:
                        if c.passenger.status == 'picked_up' and len(c.path) == 0 and c.required_steps == 1 || c.required_steps == 0:
                            if (c.passenger not in passengers_finishing):
                                passengers_finishing.append(c.passenger)
                '''
                
                # Simulator Step and update params
                self.simulator_step()
                self.prev_num_cars_idle = num_cars_idle
                num_cars_idle = self.get_number_idle_cars()
                sim_count+=1
                #print('sim count: ' + str(sim_count))
                need_to_simulate = num_cars_idle == self.prev_num_cars_idle and sim_count < 40 # and num_pass_remaining != prev_num_pass_remaining # and sim_count < 20

            #self.prev_num_cars_idle = self.get_number_idle_cars()

            # Finished simulation steps
            # Get Reward from number of passenger waiting steps if passengers finished trip

            # Get passengers dropped after simulating
            updated_passengers_dropped = self.get_passengers_dropped()
            gamma = .005
            for p in updated_passengers_dropped:
                if p not in passengers_dropped:
                    #print('id: ' + str(id(p)) )
                    reward += 1 - (gamma * p.waiting_steps)

            '''
            gamma = .001
            for p in passengers_finishing:
                print('id: ' + str(id(p)) )
                reward += 1 - (gamma * p.waiting_steps)
            '''

            # Figure out if done simulating
            done = True
            num_remaining=0
            for p in self.grid_map.passengers:
                if p.status != 'dropped':
                    num_remaining+=1
                    done = False
            #print('num pass remaining: ' + str(num_remaining))

        obs = Model.get_state(self.grid_map, 0)
        # Set indicator to all 0's by turning first element to 0
        obs[0] = 0

        return obs, reward, done, info

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

