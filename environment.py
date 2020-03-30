

from algorithm import PairAlgorithm

class Environment:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.algorithm = PairAlgorithm()

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



    # Consider two scenarios

    # implement this for joint action
    def step(self,joint_action, Model):
        info = None 
        num_cars = self.grid_map.num_cars
        num_passengers = self.grid_map.num_passengers

        # Joint_action is List of length c with each elem vec of length (p+1)
        # Pairing Step with action
        for i in range(num_cars):
            action = joint_action[i]
            c = self.grid_map.cars[i]
            p_idx = np.argmax(action)

            # If action is not 'do nothing' then pair if car is free
            if p_idx != num_passengers and c.status == 'idle':
                p = grid_map.passengers[p_idx]
                c.pair_passenger(p)
                pick_up_path = grid_map.plan_path(c.position, p.pick_up_point)
                drop_off_path = grid_map.plan_path(p.pick_up_point, p.drop_off_point)
                c.assign_path(pick_up_path, drop_off_path)


        # Next TO-DO        
        # If need to simulate, Simulate (simulator_step) until the next car changes state
            # While no cars change states
            #   simulator_step

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

