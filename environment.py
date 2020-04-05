#from algorithm import PairAlgorithm

class Environment:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        #self.algorithm = PairAlgorithm()

    def step(self):
        # paring
        self.algorithm.greedy_fcfs(self.grid_map)
        # TODO switch to reinforcement algorithm
        #self.algorithm.deep_q_learning(self.grid_map)

        for passenger in self.grid_map.passengers:
            if passenger.status == 'wait_pair' or passenger.status == 'wait_pick':
                passenger.waiting_steps += 1

        # move car
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
                 
                 
    def reset(self):
        
        self.grid_map.map_cost = {}
        self.grid_map.cars = []
        self.grid_map.passengers = []
        self.grid_map.add_passenger(self.grid_map.num_passengers)
        self.grid_map.add_cars(self.grid_map.num_cars)
        self.grid_map.init_map_cost()    
          
                        
    def dqn_step(self, action):
        
        grid_map = self.grid_map
        cars = grid_map.cars
        passengers = grid_map.passengers
        reward = [0]*len(cars)
        
        for i, act in enumerate(action):
            if act < len(passengers) and passengers[act].status == 'wait_pair':
                car = cars[i]
                passenger = passengers[act]
                car.pair_passenger(passenger)
                pick_up_path = grid_map.plan_path(car.position, passenger.pick_up_point)
                drop_off_path = grid_map.plan_path(passenger.pick_up_point, passenger.drop_off_point)
                car.assign_path(pick_up_path, drop_off_path)
                
        for passenger in passengers:
            if passenger.status == 'wait_pair' or passenger.status == 'wait_pick':
                passenger.waiting_steps += 1

        # move car

        for i,car in enumerate(cars):

            if car.status == 'idle':
                reward[i] -= 1
                continue

            # init require step
            if car.required_steps is None:  # init
                car.required_steps = self.grid_map.map_cost[(car.position, car.path[0])]

            # pick up or drop off will take one step
            if car.status == 'picking_up' and car.position == car.passenger.pick_up_point: # picking up
                reward[i] += 10
                car.pick_passenger()
            elif car.status == 'dropping_off' and car.position == car.passenger.drop_off_point:  # dropping off
                car.drop_passenger()
            else:
                # try to move
                reward[i] -= 1
                if car.required_steps > 0:  # decrease steps
                    car.required_steps -= 1
                elif car.required_steps == 0: # move
                    car.move()
                    if car.path:
                        car.required_steps = self.grid_map.map_cost[(car.position, car.path[0])]

        done = False not in [passenger.status == "dropped" for passenger in passengers]        
                        
        return reward, done
            

                    


