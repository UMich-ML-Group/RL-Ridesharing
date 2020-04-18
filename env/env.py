
import time
import numpy as np
from collections import namedtuple

from env.util import Util
from env.gridmap import GridMap
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.envs.base import Env, EnvSpaces, EnvStep
from rlpyt.utils.collections import is_namedtuple_class
from rlpyt.samplers.collections import TrajInfo

# TODO not sure what this two column means

# This env is only testing the convergence of algorithm
# The maximum trajectory reward should be 100
#EnvInfo = namedtuple("EnvInfo", ["timeout", "traj_done"])
EnvInfo = namedtuple("EnvInfo", ["timeout", "traj_done", "sim_steps"])
class TrackingEnv(Env):
    def __init__(self):
        self._target_point = np.array([0,0], dtype=np.float32)
        self._curr_point = np.random.uniform(low=-5.0, high=5.0, size=(2,)).astype(np.float32)
        self._action_space = FloatBox(low=-5, high=5, shape=(2,))
        self._observation_space = FloatBox(low=-100, high=100, shape=(2,))
        self._step_count = 0

    def step(self, action):
        print('action:', action)
        self._curr_point += action
        r = 10 - np.linalg.norm(self._curr_point-self._target_point)
        obs = self._curr_point
        #obs = np.array([0,0], dtype=np.float32)
        d = False
        if np.linalg.norm(self._curr_point-self._target_point)>20 or self._step_count >= 10:
            d = True
        info = EnvInfo(timeout=False, traj_done=d)
        self._step_count += 1
        return EnvStep(obs, r, d, info)

    def reset(self):
        self._step_count = 0
        self._curr_point = np.random.uniform(low=-10.0, high=10.0, size=(2,)).astype(np.float32)
        obs = self._curr_point
        return obs

class DispatchTrajInfo(TrajInfo):
    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self.SimSteps = 0
        self._cur_discount = 1

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self.SimSteps += env_info.sim_steps
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self

class DispatchEnv(Env):

    def __init__(self, seed=1, map_size=10, num_cars=2, num_passengers=7):
        self.gridmap_env = GridMapEnv(seed, (map_size, map_size), num_cars, num_passengers)
        self.map_size = map_size
        self.num_cars = num_cars
        self.num_passengers = num_passengers

        # SAC only allow 1-dim action space
        self._action_space = FloatBox(low=0, high=map_size, shape=(num_cars*3,))
        # SAC need flaot observe
        self._observation_space = FloatBox(low=-1, high=map_size, shape=((num_cars+num_passengers)*5,))

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

        #print(action)
        self.set_action(action)

        # move gridmap_env to next dispatch moment
        d = False
        need_dispatch = False
        timeout = False
        sim_count = 0

        while not need_dispatch:
            self.gridmap_env.step()
            d, need_dispatch = self.gridmap_env.need_dispatch()
            sim_count += 1

            # debug section
            #self.gridmap_env.render()
            #time.sleep(.1)
            #print('need_dispatch:', need_dispatch)
            #print('-'*10)

            if d:
                break

            if sim_count > 1000:
                timeout = True
                d = True
                #print('timeout:', timeout, ' done:', d)
                break

        # collect info to return
        r = -self.gridmap_env.collect_waiting_steps()
        obs = self.get_observation()
        env_info = EnvInfo(timeout=timeout, traj_done=d, sim_steps=sim_count)
        #print('reward:', r)
        #print('env:', env_info)
        #time.sleep(1)
        env_step = EnvStep(obs, r, d, env_info)
        #print(env_step)
        return env_step

    def reset(self):
        #print('**reset**')
        self.gridmap_env.reset()
        obs = self.get_observation()
        return obs

    def get_observation(self):
        '''
            return a np int array with dimension (#car+#pass, 5),
            with value range [-1, map_size]
            2nd dim for cars := (x, y, required_steps, is_idel, null)
            2nd dim for pass := (curr x, curr y, drop x, drop y, is_wait)
        '''
        obs = np.zeros((self.num_cars+self.num_passengers, 5), dtype=np.float32)

        idx = 0
        for c in self.gridmap_env.all_cars():
            x, y = c.position
            obs[idx, 0] = x
            obs[idx, 1] = y
            obs[idx, 2] = 1 if c.status == 'idle' else 0
            obs[idx, 3] = 0 if not c.required_steps else c.required_steps
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

        #print('obs:\n', obs)
        return obs.flatten()

    def set_action(self, action):
        '''
            input is an np float array with dim (#cars*3)
            we reshape to (#cars, 3)
            with value range [0, map_size]
            2nd dim for := (dest x, dest y, assign range)
        '''
        action_mat = (action+self.map_size/2).reshape(self.num_cars, 3)
        #print('act:\n', action_mat)
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
                    #print('===== pair =====')
                    #print(c)
                    #print(p)
                    break

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

