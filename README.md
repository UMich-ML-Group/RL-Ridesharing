# RL-Ridesharing
Effcient Ridesharing Dispatch Using Reinforcement Learning

Shared google doc:
https://docs.google.com/document/d/1eSJ6UNhpUenhdOfjNK9OoH-TnjUoNzNX6BgmGwwovZM/edit

--------------------
#Class Structure:

* Util
* Enviroment:
  * GridMap
    * Car
    * Passenger
  * Pair_Algorithm

# Feature Representation (Discussing)

* Observation: 
  Type: spaces.Dict
  {"Cars": [...], "Passengers": [...]}

  max_num_of_cars=3, max_num_of_passengers=3

  len of vector will be max_num_of_cars+max_num_of_passengers

  [car1_x, car1_y, car2_x, car2_y, ...., passenger1_x, passenger1_y, ....]

  If the num of cars is smaller than max number, than set car1_x=-1, car1_y=-1.
Same as the passenger

* Actions:
  Type: Discrete(len(Cars))

  max_num_of_cars=3
  len of vector will be max_num_of_cars

  [2,1,3]

  The list corresponds to an assigment from the car at that every index to a   passanger with the corresponding number.

  Each number should be a interger between -1 and the number of passangers.

  -1 means that the corresponding car has no assignment.

* Reward:
  Reward is 1 when a passanger is dropped and -0.1 for every step taken.

  Starting State:
  Uniform random initialization of cars, passangers and road costs.

  Episode Termination:
  The episode ends after every passanger is picked up.
