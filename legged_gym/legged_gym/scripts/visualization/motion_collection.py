import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.control.action_scale = 0.25
    env_cfg.curriculum.pull_force = False
    env_cfg.domain_rand.use_random = False  
    env_cfg.env.test = True


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    os.makedirs('./visualization/data', exist_ok=True)
    root_state = []
    for i in range(10*int(env.max_episode_length)):
        result = env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if env.real_episode_length_buf[0] >= env.unactuated_time:
            root_state.append(env.rigid_body_states[0].detach().cpu().numpy())

        if len(root_state) == 200:
            np.save(f'./visualization/data/{train_cfg.runner.experiment_name}_root_state.npy', np.array(root_state), allow_pickle=True)
            break
    

if __name__ == '__main__':
    args = get_args()
    play(args)
