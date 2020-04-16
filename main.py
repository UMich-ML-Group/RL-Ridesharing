#!/usr/bin/python3

import time
from IPython.display import clear_output
from env.gridmap import GridMap
from env.env import DispatchEnv
from env.env import TrackingEnv

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.collections import TrajInfo
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.pg.ppo import PPO
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.agents.pg.gaussian import GaussianPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

def build_and_train(run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        #EnvCls=DispatchEnv,
        EnvCls=TrackingEnv,
        TrajInfoCls=TrajInfo,  # default traj info + GameScore
        env_kwargs={},
        eval_env_kwargs={},
        batch_T=1,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=1,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN()
    #algo = SAC()  # Run with defaults for other params.
    #algo = PPO()
    agent = DqnAgent(
            ModelCls=DuelingHeadModel,
            model_kwargs=dict(
                input_size=4,
                hidden_sizes=256,
                output_size=20))
    #agent = SacAgent()
    #agent = GaussianPgAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        #n_steps=50e6,
        n_steps=1e6,
        log_interval_steps=1e2,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = None
    name = "Ridesharing"
    log_dir = "Ridesharing"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

def _build_and_train(game="pong", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    algo = DQN(min_steps_learn=1e3)  # Run with defaults.
    agent = AtariDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

def main():
    from env.environment import Environment
    grid_map = GridMap(1, (7,7), 3, 3)
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

if __name__ == "__main__":
    #main()
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
