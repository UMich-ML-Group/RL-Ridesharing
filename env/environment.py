

from env.algorithm import PairAlgorithm

class Environment:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.algorithm = PairAlgorithm()

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

