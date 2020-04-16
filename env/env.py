
import numpy as np
from collections import namedtuple

from env.util import Util
from env.gridmap import GridMap
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.envs.base import Env, EnvSpaces, EnvStep
from rlpyt.utils.collections import is_namedtuple_class

# TODO not sure what this two column means
EnvInfo = namedtuple("EnvInfo", ["timeout", "traj_done"])

class TrackingEnv(Env):
    def __init__(self):
        self._target_point = np.array([0,0], dtype=np.float32)
        self._curr_point = np.random.uniform(low=-10.0, high=10.0, size=(2,)).astype(np.float32)
        self._step_count = 0

        #self._action_space = FloatBox(low=-1, high=1, shape=(2,))
        self._action_space = IntBox(low=0, high=5)
        self._observation_space = FloatBox(low=-10, high=10, shape=(2,))

    def step(self, action):
        self._curr_point += action
        r = 10 - np.linalg.norm(self._curr_point-self._target_point)

        # add some noise
        #self._curr_point += (np.matlib.rand(2)-0.5)*2
        timeout = None

        obs = self._curr_point
        d = True if self._step_count >= 100 else False
        info = EnvInfo(timeout=timeout, traj_done=d)
        print('action:', action)
        print('reward:', r)
        print('obs:', obs)
        return EnvStep(obs, r, d, info)


    def reset(self):
        self._curr_point = np.random.uniform(low=-10.0, high=10.0, size=(2,)).astype(np.float32)
        self._step_count = 0
        obs = self._curr_point
        return obs


class DispatchEnv(Env):

    def __init__(self, seed=1, map_size=10, num_cars=2, num_passengers=2):
        self.gridmap_env = GridMapEnv(seed, (map_size, map_size), num_cars, num_passengers)
        self.map_size = map_size
        self.num_cars = num_cars
        self.num_passengers = num_passengers

        # SAC only allow 1-dim action space
        self._action_space = FloatBox(low=0, high=map_size, shape=(num_cars*3))
        # SAC need flaot observe
        self._observation_space = FloatBox(low=-1, high=map_size, shape=(num_cars+num_passengers, 5))

    def step(self, action):
        # parssing action space
        # TODO assume input action is perfect so far, which is not true
        '''
            input is an np float array with dim (#cars, 3)
            2nd dim for := (dest x, dest y, assign range)
        '''
        obs = None
        r = None
        d = None
        info = None

        print(action)
        self.set_action(action)

        # move gridmap_env to next dispatch moment
        d = False
        need_dispatch = True
        timeout = False
        sim_count = 0
        while need_dispatch:
            self.gridmap_env.step()
            d, need_dispatch = self.gridmap_env.need_dispatch()
            sim_count += 1
            if sim_count > 100:
                timeout = True
                print('force break')
                break

        # collect info to return
        r = -self.gridmap_env.collect_waiting_steps()
        obs = self.get_observation()
        info = EnvInfo(timeout=timeout, traj_done=d)
        print('reward:', r)
        print('env obs:\n', obs)
        return EnvStep(obs, r, d, info)

    def reset(self):
        self.gridmap_env.reset()
        obs = self.get_observation()
        return obs

    def get_observation(self):
        '''
            return a np int array with dimension (#car+#pass, 5),
            with value range [-1, map_size]
            2nd dim for cars := (x, y, is_idel, null, null)
            2nd dim for pass := (curr x, curr y, drop x, drop y, is_wait)
        '''
        obs = np.zeros((self.num_cars+self.num_passengers, 5), dtype=np.float32)

        idx = 0
        for c in self.gridmap_env.all_cars():
            x, y = c.position
            obs[idx, 0] = x
            obs[idx, 1] = y
            obs[idx, 2] = 1 if c.status == 'idle' else 0
            obs[idx, 3] = -1 # unuse
            obs[idx, 4] = -1 # unuse
            idx += 1

        for p in self.gridmap_env.all_passengers():
            px, py = p.pick_up_point
            dx, dy = p.drop_off_point
            obs[idx, 0] = px
            obs[idx, 1] = py
            obs[idx, 2] = dx
            obs[idx, 3] = dy
            obs[idx, 4] = 1 if p.status == 'wait_pair' else 0
            idx += 1

        return obs

    def set_action(self, action):
        '''
            input is an np float array with dim (#cars*3)
            we reshape to (#cars, 3)
            with value range [0, map_size]
            2nd dim for := (dest x, dest y, assign range)
        '''
        #print('action:', action)
        action_mat = action.reshape(self.num_cars, 3)
        print('action mat:\n', action_mat)
        for i, c in enumerate(self.gridmap_env.all_cars()):
            if c.status != 'idle':
                continue

            predict_point = (action_mat[i, 0], action_mat[i, 1])
            assign_range = action_mat[i, 2]

            for p in self.gridmap_env.all_passengers():
                if p.status != 'wait_pair':
                    continue

                # if distance between car's predict position and passenger's pick up position
                # smaller than assign range, then pair up
                if Util.cal_dist(predict_point, p.pick_up_point) < assign_range:
                    self.gridmap_env.pair(c, p)
                    print('===== pair =====')
                    print(c)
                    print(p)

    #@property
    #def horizon(self):
    #    pass

class GridMapEnv:

    def __init__(self, seed=1, map_size=(10,10), num_cars=10, num_passengers=10):
        # TODO remove hardcode parameter for gridmap
        self.grid_map = GridMap(seed, map_size, num_cars, num_passengers)

    def need_dispatch(self):
        episode_done = True
        has_passenger = False
        has_car = False
        for p in self.grid_map.passengers:
            if p.status != 'dropped':
                episode_done = False
            if p.status == 'wait_pair':
                has_passenger = True
        for c in self.grid_map.cars:
            if c.status == 'idle':
                has_car = True

        return (episode_done, (has_car and has_passenger))

    def all_cars(self):
        for c in self.grid_map.cars:
            yield c

    def all_passengers(self):
        for p in self.grid_map.passengers:
            yield p

    def reset(self):
        self.grid_map.reset_car_and_passenger()

    def render(self):
        self.grid_map.visualize()

    def pair(self, car, passenger):
        car.pair_passenger(passenger)
        pick_up_path = self.grid_map.plan_path(car.position, passenger.pick_up_point)
        drop_off_path = self.grid_map.plan_path(passenger.pick_up_point, passenger.drop_off_point)
        car.assign_path(pick_up_path, drop_off_path)

    def collect_waiting_steps(self):
        total_waiting_steps = 0
        for p in self.grid_map.passengers:
            total_waiting_steps += p.waiting_steps
            p.waiting_steps = 0
        return total_waiting_steps

    def step(self):
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

