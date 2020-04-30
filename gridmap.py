
import math
import random

from util import Util
from car import Car
from passenger import Passenger

class GridMap:
    def __init__(self, seed, size, num_cars, num_passengers):
        random.seed(seed)
        self.seed = seed
        self.size = size #(row, col)
        self.num_cars = num_cars
        self.num_passengers = num_passengers
        self.map_cost = {}
        self.cars = []
        self.passengers = []
        self.add_passenger(num_passengers)
        self.add_cars(num_cars)
        self.init_map_cost()

    def __repr__(self):
        message = 'cls:' + type(self).__name__ + \
                ', size:' + str(self.size) + ', seed:' + str(self.seed )+ '\n'
        for c in self.cars:
            message += repr(c) + '\n'
        for p in self.passengers:
            message += repr(p) + '\n'
        return message

    def is_valid(self, p1):
        if p1[0]<0 or p1[1]<0 or p1[0]>self.size[0] or p1[1]>self.size[1]:
            return False
        return True

    def is_adjacent(self, p1, p2):
        assert p1[0]<0 or p2[0]<0 or p1[1]<0 or p2[1]<0, "map point is out of boundary"
        assert p1[0]>self.size[0] or p2[0]>self.size[0] or p1[1]>self.size[1] or p2[1]>self.size[1],\
                "map point is out of boundary"

        if (abs(p1[0]-p2[0]) == 1 and abs(p1[1]-p2[1]) == 0) or \
                  (abs(p1[0]-p2[0]) == 0 and abs(p1[1]-p2[1]) == 1):
          return True

    def init_map_cost(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                p = (row, col)
                # for up
                p_up = (row-1, col)
                if self.is_valid(p_up): self.map_cost[(p, p_up)] = random.randint(0,9)
                p_right = (row, col+1)
                if self.is_valid(p_right): self.map_cost[(p, p_right)] = random.randint(0,9)
                p_down = (row+1, col)
                if self.is_valid(p_down): self.map_cost[(p, p_down)] = random.randint(0,9)
                p_left = (row, col-1)
                if self.is_valid(p_left): self.map_cost[(p, p_left)] = random.randint(0,9)
                
    def init_zero_map_cost(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                p = (row, col)
                # for up
                p_up = (row-1, col)
                if self.is_valid(p_up): self.map_cost[(p, p_up)] = 0
                p_right = (row, col+1)
                if self.is_valid(p_right): self.map_cost[(p, p_right)] = 0
                p_down = (row+1, col)
                if self.is_valid(p_down): self.map_cost[(p, p_down)] = 0
                p_left = (row, col-1)
                if self.is_valid(p_left): self.map_cost[(p, p_left)] = 0
        

    def add_cars(self, num_cars):
        assert num_cars <= self.size[0]*self.size[1], 'number of cars is larger than number of grids'
        car_set = set()
        while True:
            if len(car_set) == num_cars:
                break
            p = (random.randint(0,self.size[0]-1), random.randint(0,self.size[1]-1))
            if p not in car_set:
                car_set.add(p)

        for s in car_set:
            self.cars.append(Car(s))

    def add_passenger(self, num_passengers):
        assert num_passengers <= self.size[0]*self.size[1], 'number of passengers is larger than number of grids'
        passenger_set = set()
        while True:
            if len(passenger_set) == num_passengers:
                break
            p = (random.randint(0,self.size[0]-1), random.randint(0,self.size[1]-1))
            if p not in passenger_set:
                passenger_set.add(p)

        for s in passenger_set:
            # generate destination point
            while True:
                d = (random.randint(0,self.size[0]-1), random.randint(0,self.size[1]-1))
                if d != s:
                    break
            self.passengers.append(Passenger(s, d))

    def plan_path(self, start_point, end_point):
        def check_optim(next_pos, min_dist, optim_next_pos):
            if self.is_valid(next_pos):
                dist = Util.cal_dist(next_pos, end_point)
                if dist < min_dist:
                    min_dist = dist
                    optim_next_pos = next_pos
            return min_dist, optim_next_pos
        path = []
        curr_pos = start_point
        while curr_pos != end_point:
            min_dist = math.inf
            optim_next_pos = None
            # up
            min_dist, optim_next_pos = check_optim((curr_pos[0]-1, curr_pos[1]), min_dist, optim_next_pos)
            # down
            min_dist, optim_next_pos = check_optim((curr_pos[0]+1, curr_pos[1]), min_dist, optim_next_pos)
            # left
            min_dist, optim_next_pos = check_optim((curr_pos[0], curr_pos[1]-1), min_dist, optim_next_pos)
            # right
            min_dist, optim_next_pos = check_optim((curr_pos[0], curr_pos[1]+1), min_dist, optim_next_pos)
            assert optim_next_pos is not None, 'no valid position for optim_next_pos'
            path.append(optim_next_pos)
            curr_pos = optim_next_pos
        return path

    def visualize(self):
        m = [["     " for i in range(self.size[1])] for j in range(self.size[0])]
        for p in self.passengers:
            if p.status == 'wait_pair' or p.status == 'wait_pick':
                # passenger
                m[p.pick_up_point[0]][p.pick_up_point[1]] = "p" + str(id(p))[-2:] + "  "
        for c in self.cars:
            if c.status == 'dropping_off':
                # car with passenger
                m[c.position[0]][c.position[1]] = "x" + str(id(c.passenger))[-2:] + ":" + str(c.required_steps)[0]
            elif c.status == 'picking_up':
                # car assigned passenger
                m[c.position[0]][c.position[1]] = "c" + str(id(c.passenger))[-2:] + ":" + str(c.required_steps)[0]
            else:
                # car is idle
                m[c.position[0]][c.position[1]] = "c--  "

        for i in range(len(m)):
          print(m[i])


if __name__ == '__main__':
    m = GridMap(0, (10,10), 3, 3)
    print(m)
    print('path from (0,0) to (5,5):')
    path = m.plan_path((0,0),(5,5))
    print(path)
    m.visualize()
