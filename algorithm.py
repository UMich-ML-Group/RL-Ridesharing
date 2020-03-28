
import math
from gridmap import GridMap
from passenger import Passenger
from car import Car
from util import Util

class PairAlgorithm:
    def assign_random(self, cars, passengers):
        cars = random.shuffle(cars)
        passengers = random.shuffle(passengers)

        car_count = 0
        for i, car in enumerate(cars):
            p = passengers[i] if i < len(passengers) and car.status == 'idle' else None
            car.set_passenger(p)

    def greedy_fcfs(self, grid_map):
        passengers = grid_map.passengers
        cars = grid_map.cars
        for p in passengers:
            min_dist = math.inf
            assigned_car = None
            if p.status == 'wait_pair':
                for c in cars:
                    if c.status == 'idle':
                        dist = Util.cal_dist(p.pick_up_point, c.position)
                        if dist < min_dist:
                            min_dist = dist
                            assigned_car = c
                if assigned_car is not None:
                    assigned_car.pair_passenger(p)
                    pick_up_path = grid_map.plan_path(assigned_car.position, p.pick_up_point)
                    drop_off_path = grid_map.plan_path(p.pick_up_point, p.drop_off_point)
                    assigned_car.assign_path(pick_up_path, drop_off_path)

if __name__ == '__main__':
    algorithm = PairAlgorithm()
    grid_map = GridMap(0, (5,5), 3, 3)
    grid_map.init_map_cost()
    grid_map.visualize()
    print(grid_map)
    algorithm.greedy_fcfs(grid_map)
    print(grid_map)
