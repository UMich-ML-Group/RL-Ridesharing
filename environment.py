#from algorithm import PairAlgorithm

class Environment:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        #self.algorithm = PairAlgorithm()
                 
                 
    def reset(self):
        
        self.grid_map.cars = []
        self.grid_map.passengers = []
        self.grid_map.add_passenger(self.grid_map.num_passengers)
        self.grid_map.add_cars(self.grid_map.num_cars) 
          
                        
    def step(self, action, mode):
        
        grid_map = self.grid_map
        cars = grid_map.cars
        passengers = grid_map.passengers
        reward = [0]*len(passengers)
        done = False
        
        duration = 0
        
        while not done:
            
            # print("Action: ", action)
            # print(self.grid_map)
            # input("Press enter to step")
            # self.grid_map.visualize()
            
        
            for i, act in enumerate(action[0]):
                if cars[act].status == "idle" and passengers[i].status == "wait_pair":
                    car = cars[act]
                    passenger = passengers[i]
                    car.pair_passenger(passenger)
                    pick_up_path = grid_map.plan_path(car.position, passenger.pick_up_point)
                    drop_off_path = grid_map.plan_path(passenger.pick_up_point, passenger.drop_off_point)
                    car.assign_path(pick_up_path, drop_off_path)
                    
            

            for i,car in enumerate(cars):

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
                            
            for passenger in passengers:
                if passenger.status == 'wait_pair' or passenger.status == 'wait_pick':
                    passenger.waiting_steps += 1
            
            done = False not in [passenger.status == "dropped" for passenger in passengers] 
            
            duration += 1
        
        if mode == "dqn":
            reward = [-passenger.waiting_steps for passenger in passengers]
        
        if mode == "qmix" or mode == "iql":
            #reward = sum(reward)/1000 #len(passengers)
            reward = -duration
            

        
                    
        return reward, duration
    
    
            

                    


