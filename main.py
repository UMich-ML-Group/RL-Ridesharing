#!/usr/bin/python3

import time
#from IPython.display import clear_output
from env.gridmap import GridMap
from env.environment import Environment

def main():
    grid_map = GridMap(1, (10,10), 2, 7)
    env = Environment(grid_map)

    total_iter = 100
    total_step = 0
    for i in range(total_iter):
        env.reset()
        step_count = 0
        while True:
            finished = True
            for p in grid_map.passengers:
                if p.status != 'dropped':
                  finished = False
            #clear_output()
            #grid_map.visualize()
            #time.sleep(.1)
            #print('-'*10)
            env.step()
            step_count += 1
            if finished:
                print('step cost:', step_count)
                total_step += step_count
                break

    print('avg steps:', total_step/total_iter)

if __name__ == "__main__":
    main()
