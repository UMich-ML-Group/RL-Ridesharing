
import math
from gridmap import GridMap
from passenger import Passenger
from car import Car
from util import Util
from dqn import DQN

class PairAlgorithm:


    def greedy_fcfs(self, grid_map):
        passengers = grid_map.passengers
        cars = grid_map.cars
        action = [0]*len(passengers)
        for i, p in enumerate(passengers):
            min_dist = math.inf
            assigned_car = None
            for j, c in enumerate(cars):
                dist = Util.cal_dist(p.pick_up_point, c.position)
                if dist < min_dist:
                    min_dist = dist
                    assigned_car = j
            action[i] = assigned_car

        return action
                    


if __name__ == '__main__':
    algorithm = PairAlgorithm()
    grid_map = GridMap(0, (5,5), 3, 3)
    grid_map.init_map_cost()
    grid_map.visualize()
    print(grid_map)
    algorithm.greedy_fcfs(grid_map)
    print(grid_map)
