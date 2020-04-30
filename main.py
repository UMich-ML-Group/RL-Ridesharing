#!/usr/bin/python3

import time
from IPython.display import clear_output
from gridmap import GridMap
from environment import Environment

def main():
    grid_map = GridMap(1, (7,7), 3, 4)
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
        step_count += 1
        time.sleep(1)
        if finished:
            print('step cost:', step_count)
            break

if __name__ == '__main__':
    main()
