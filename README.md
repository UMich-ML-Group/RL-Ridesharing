# RL-Ridesharing
Effcient Ridesharing Dispatch Using Reinforcement Learning

Shared google doc:
https://docs.google.com/document/d/1eSJ6UNhpUenhdOfjNK9OoH-TnjUoNzNX6BgmGwwovZM/edit

--------------------

# Class Structure:

* Util
* Enviroment:
  * GridMap
    * Car
    * Passenger
  * Pair_Algorithm

# Feature Representation

In order to reduce the action space, so the action now is a pair decision for one car, meaning we need an indicator to point out which car we are pairing now.

### Observation: = [indicator, cars, passengers]

indicator: one-hot encoding to indicate which car we are pairing now

cars: [car_1, car_2, ..., car_max] where car_n = (position_x, position_y)

passengers: [person_1, person_2, ..., person_max] where person_n = (pick_up_x, pick_up_y, destination_x, destination_y)

* Example:

car_max = 3, passenger_max = 4

observation will be a vector with dimension: 3+3\*2+4\*4 = 25


### Action: = [passengers_q, ignore]

passengers_q: [person_q_1, person_q_2, ..., person_q_max]

ignore: is a scalar, meaning the q-value of choosing do nothing.

* Example:

car_max = 3, passenger_max = 4

action will be a vector with dimension: 4+1 = 5

### Reward: = (constant_drop_off_reward - \#steps_waiting_for_pick_up), which is a scalar

reward is given to the environment when the passenger has been drop off.

We can use a callback (traceback) function to relate the reward to a certain previous action caused the reward, and then put into replay buffer.

* Example

Let constant_drop_off_reward = 100
A passenger is picked up 23 environment steps after.
The reward from this passenger is 100-23=77 and is given after drop off .

### Environment step: represent world timestamp

For one environment step, either car move foward a grid, require_step minus 1, pick up, or drop off.

### Algorithem step: Whenever the pair is required

The length of alforithem step is varied in terms of environment steps, and is lasting until the next alforithem happen or termination.

condition1: there is at least 1 passenger request which hasn't be served.

condition2: there is at least 1 avaliable car

### Environment initialization:
Randomly assign costs of road.

### Episode Start:
Uniformly generate cars and passangers on the map.

### Episode Termination:
All the passangers have been served.

For now, there is no passenger will be generated during the episode. (We can test more complicated senario later)

