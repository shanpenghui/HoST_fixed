import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import time

import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Process, Value


class EvalLogger:
    def __init__(self, env, num_episodes, device='cuda:0'):
        self.env = env
        self.num_episodes = num_episodes
        self.device = device
        self.curr_episode = 0
        
        self.reset_buffer()

        # real metric
        self.success_buffer = torch.zeros((env.num_envs, num_episodes) , dtype=torch.bool)
        self.feet_movement = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.smoothness = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.motion_tracking_error = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.power_all = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)
        self.smoothness_before_standingup = torch.zeros((env.num_envs, num_episodes), dtype=torch.float, device=self.device)

    def log(self):
        # base movement
        new_base_xy = self.env.root_states[:, :2].clone()
        base_xy_movement = torch.sum(torch.square(new_base_xy - self.base_xy), dim=-1)
        self.base_xy[:] = new_base_xy.clone()
        self.base_xy_movement[:] += base_xy_movement.clone() * self.ever_standingup

        # feet movement
        new_left_feet_xyz = self.env.rigid_body_states[:, self.env.left_foot_indices, :3].clone().squeeze(1) * 100
        new_right_feet_xyz = self.env.rigid_body_states[:, self.env.right_foot_indices, :3].clone().squeeze(1) * 100
        feet_xyz_movement = torch.sum(torch.square(new_left_feet_xyz - self.left_feet_xyz), dim=-1) + torch.sum(torch.square(new_right_feet_xyz - self.right_feet_xyz), dim=-1)
        self.left_feet_xyz[:] = new_left_feet_xyz.clone()
        self.right_feet_xyz[:] = new_right_feet_xyz.clone()
        self.feet_xyz_movement[:] += feet_xyz_movement.clone() * self.ever_standingup

        # smoothness
        self.action_smooth[:] = torch.sum(torch.square(self.env.dof_pos - self.env.last_dof_pos - self.env.last_dof_pos + self.env.last_last_dof_pos), dim=1) 
        self.smoothness[:, self.curr_episode] += self.action_smooth.clone()

        # pose tracking error
        pose_mse = torch.sum(torch.square(self.env.dof_pos[:, self.env.upper_body_joint_indices] - self.env.target_dof_pos[:, self.env.upper_body_joint_indices]), dim=-1)
        self.motion_tracking_error[:, self.curr_episode] += pose_mse.clone() * self.ever_standingup

        # power & smoothness before standing up
        self.power[:] += torch.sum(torch.abs(self.env.dof_vel) * torch.abs(self.env.torques), dim=-1)  * 0.02 *  ~self.ever_standingup #base_vel
        self.smoothness_before_standingup[:, self.curr_episode] += self.action_smooth.clone() * ~self.ever_standingup

        # common
        self.base_height_buffer[:] = self.env.relative_root_state[:, 2].clone()
        self.ever_standingup[:] = self.ever_standingup | (self.base_height_buffer > 0.6) & (self.env.real_episode_length_buf > self.env.unactuated_time)
        self.falldown_after_standup[:] = self.falldown_after_standup | (self.ever_standingup & (self.base_height_buffer < 0.1))
        self.standingup_times += self.ever_standingup
        self.action_times += self.env.real_episode_length_buf > self.env.unactuated_time
        self.before_standingup_times[:] +=  ~self.ever_standingup.clone()
        # print(self.ever_standingup[:])

    def compute_metric(self):
        # real metric
        self.success_buffer[:, self.curr_episode] = self.ever_standingup.clone() & ~self.falldown_after_standup.clone()
        self.feet_movement[:, self.curr_episode] = self.feet_xyz_movement.clone() / (self.standingup_times + 1)
        self.smoothness[:, self.curr_episode] /=  self.action_times
        self.motion_tracking_error[:, self.curr_episode] /= self.standingup_times + 1 
        self.power_all[:, self.curr_episode] = self.power.clone() #/ self.before_standingup_times
        self.smoothness_before_standingup[:, self.curr_episode] /= (self.before_standingup_times + 1)

    def reset_buffer(self):
        self.base_height_buffer = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.ever_standingup = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.falldown_after_standup = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        self.standingup_times = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)
        self.action_times = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)
        self.before_standingup_times = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.device)

        self.base_xy = torch.zeros((self.env.num_envs, 2), dtype=torch.float, device=self.device)
        self.base_xy_movement = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.left_feet_xyz = torch.zeros((self.env.num_envs, 3), dtype=torch.float, device=self.device)
        self.right_feet_xyz = torch.zeros((self.env.num_envs, 3), dtype=torch.float, device=self.device)
        self.feet_xyz_movement = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.action_smooth = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.power = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def reset(self):
        print('before_standingup_times', self.before_standingup_times.float().mean())
        self.compute_metric()
        self.reset_buffer()
        self.curr_episode += 1

    def plot_terminal_metrics(self):
        print('Success rate: ', torch.mean(self.success_buffer.float() * 100).item(), torch.std(torch.mean(self.success_buffer.float() * 100, dim=0)).item())

        print('Average feet movement: ', torch.mean(self.feet_movement * (self.feet_movement > 0)).item(), torch.std(torch.mean(self.feet_movement * (self.feet_movement > 0), dim=0)).item())
        print('Average smoothness: ', torch.mean(self.smoothness * 180 / np.pi).item(), torch.std(torch.mean(self.smoothness * 180 / np.pi, dim=0)).item())
        print('Average motion tracking error: ', torch.mean(self.motion_tracking_error).item(), torch.std(torch.mean(self.motion_tracking_error, dim=0)).item())

        print('Average times before standing up: ', torch.mean(self.before_standingup_times.float()).item(),  torch.std(self.before_standingup_times.float()).item())
        print('Average energy: ', torch.mean(self.power_all).item(), torch.std(torch.mean(self.power_all, dim=0)).item())
        print('Average smoothness before standing up: ', torch.mean(self.smoothness_before_standingup * 180 / np.pi).item(), torch.std(torch.mean(self.smoothness_before_standingup * 180 / np.pi, dim=0)).item())


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 100
    env_cfg.terrain.num_rows = 11
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.env.episode_length_s = 5
    env_cfg.control.action_scale = 0.3
    env_cfg.curriculum.pull_force = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    num_episodes = 5
    evalogger = EvalLogger(env, num_episodes)
    evalogger.log()
    for i in range(num_episodes):
        for j in range(int(env.max_episode_length + 1)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            if j != int(env.max_episode_length): evalogger.log()
        evalogger.reset()

    evalogger.plot_terminal_metrics()


if __name__ == '__main__':
    args = get_args()
    play(args)
