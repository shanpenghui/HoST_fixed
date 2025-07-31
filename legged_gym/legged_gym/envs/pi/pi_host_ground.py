from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .pi_config_ground import PiCfg

from legged_gym.envs.g1.g1_utils import (
    MotionLib, 
    load_imitation_dataset,
    compute_residual_observations,
    tolerance
)
from legged_gym.utils.math import (
    quat_rotate,
    euler_xyz_to_quat,
    quat_apply_yaw,
    quat_apply_yaw_inverse,
    quat_mul_yaw_inverse,
    quat_conjugate,
    quat_apply,
    quat_mul,
    quat_to_euler_xyz,
    quat_to_rot6d,
    quat_to_angle_axis,
    torch_rand_float,
    quat_conjugate,
)


class LeggedRobot_Pi(BaseTask):
    def __init__(self, cfg: PiCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # 保存传入的配置、仿真参数
        self.cfg = cfg
        self.sim_params = sim_params
        # 如果是 rough terrain 高度图测量，会在后面设置
        self.height_samples = None
        # 是否启用调试可视化
        self.debug_viz = False
        # 初始化完成标志，用于防止重复初始化
        self.init_done = False
        # 解析配置参数。这个函数内部会从 cfg 中读取一些变量并赋值为类成员变量。
        # 包括 reward scale、terminate 条件、控制参数等
        self._parse_cfg(self.cfg)
        # 从配置中读取机器人关节自由度数量（如 12 DOFs for 四足或人形）
        self.num_real_dofs = cfg.env.num_dofs

        # 调用父类（一般是 LeggedRobot) 的构造函数，完成仿真环境、机器人资源（URDF）、地形等创建。
        # 内部会调用 create_sim() → 创建多个并行环境（envs）、加载机器人模型
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 单步观测维度（如 base 速度 + 姿态 + 每个关节角/速/力 = 43）
        self.num_one_step_obs = self.cfg.env.num_one_step_observations #if not self.cfg.env.add_force else self.cfg.env.num_one_step_observations + 1
        # 状态历史帧数（如 6），表示每一步将保留最近 6 帧观测用于时序信息输入
        self.actor_history_length = self.cfg.env.num_actor_history
        # 整个输入 observation 的总维度（如 43 × 6 = 258）, 用于 LSTM/Transformer 等时序建模
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length

        # print(f"[INFO] num_one_step_obs = {self.num_one_step_obs}")
        # print(f"[INFO] actor_history_length = {self.actor_history_length}")
        # print(f"[INFO] actor_proprioceptive_obs_length = {self.actor_proprioceptive_obs_length}")

        # 若不是 headless 模式（即带可视化 GUI），则设置摄像机位置与焦点
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # 初始化 PyTorch 张量缓冲区（如 reset_buf、episode_length_buf、contact_forces、obs_buf 等）。
        # 这些 buffer 在每步 step() 中都会更新，是强化学习过程的核心数据结构。
        self._init_buffers()
        # 初始化奖励函数，包括设置 reward name → index 的映射。
        # 初始化每个 reward 的 scale，用于后续 reward 加权。
        self._prepare_reward_function()
        self.init_done = True
        # unactuated_timesteps 表示前多少步不触发 base_vel 等 reset 条件，允许“自由落体”或无动作
        self.unactuated_time = self.cfg.env.unactuated_timesteps
        # 乘以 0.02 / self.dt 是将 policy 的步数换算为仿真步数（因为通常每步动作会重复 decimation 次）。
        # 其中 0.02 是 PPO 默认的 policy timestep（20ms），除以仿真 timestep 可得出实际 sim steps。
        self.unactuated_time *= 0.02 / self.dt
        # 用于控制奖励/约束是否使用 Gaussian 形式
        # 若为 True，某些 reward 项将使用 exp(-x^2 / sigma) 形式
        self.is_gaussian = cfg.rewards.is_gaussian

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if torch.isnan(actions).any() or actions.abs().max() > 10:
            print(f"[ERROR][step-input] Abnormal raw actions: min={actions.min().item()}, max={actions.max().item()}")
        if torch.isnan(self.actions).any() or self.actions.abs().max() > 10:
            print(f"[ERROR][step-clipped] Abnormal clipped actions: min={self.actions.min().item()}, max={self.actions.max().item()}")

        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.actions *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            if self.actions.abs().max() > 10 or torch.isnan(self.actions).any():
                print(f"[ERROR][step-masked] Masked actions abnormal: max={self.actions.abs().max().item()}")
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            if torch.isnan(self.torques).any() or self.torques.abs().max() > 1000:
                print(f"[ERROR][step-torques] Abnormal torques: min={self.torques.min().item()}, max={self.torques.max().item()}")

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            # vertical pulling force
            if self.cfg.curriculum.pull_force:
                force_tensor = torch.zeros([self.num_envs, self.num_bodies, 3], device=self.device)
                force_tensor[:, self.base_indices, 2] = self.force 

                force_tensor *= (self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time).unsqueeze(1)
                if not self.cfg.curriculum.no_orientation:
                    force_tensor *= (self.projected_gravity[:, 2] < -0.8).unsqueeze(1).unsqueeze(1)
                force_tensor = gymtorch.unwrap_tensor(force_tensor)
                self.gym.apply_rigid_body_force_tensors(self.sim, force_tensor)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.real_episode_length_buf += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.init_rpy = None
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # === 调试：检查是否有非法物理状态 ===
        if torch.isnan(self.dof_vel).any() or torch.isinf(self.dof_vel).any() or (self.dof_vel.abs() > 1e5).any():
            bad_env_ids = torch.where(torch.isnan(self.dof_vel).any(dim=1) |
                                      torch.isinf(self.dof_vel).any(dim=1) |
                                      (self.dof_vel.abs() > 1e5).any(dim=1))[0]
            print("[ERROR] Abnormal dof_vel in envs:", bad_env_ids.cpu().numpy())
            print("  → dof_vel min/max:", self.dof_vel.min().item(), self.dof_vel.max().item())

        if torch.isnan(self.base_lin_vel).any() or torch.isinf(self.base_lin_vel).any() or (self.base_lin_vel.abs() > 1e5).any():
            bad_env_ids = torch.where(torch.isnan(self.base_lin_vel).any(dim=1) |
                                      torch.isinf(self.base_lin_vel).any(dim=1) |
                                      (self.base_lin_vel.abs() > 1e5).any(dim=1))[0]
            print("[ERROR] Abnormal base_lin_vel in envs:", bad_env_ids.cpu().numpy())
            print("  → base_lin_vel min/max:", self.base_lin_vel.min().item(), self.base_lin_vel.max().item())

        if torch.isnan(self.base_ang_vel).any() or torch.isinf(self.base_ang_vel).any() or (self.base_ang_vel.abs() > 1e5).any():
            bad_env_ids = torch.where(torch.isnan(self.base_ang_vel).any(dim=1) |
                                      torch.isinf(self.base_ang_vel).any(dim=1) |
                                      (self.base_ang_vel.abs() > 1e5).any(dim=1))[0]
            print("[ERROR] Abnormal base_ang_vel in envs:", bad_env_ids.cpu().numpy())
            print("  → base_ang_vel min/max:", self.base_ang_vel.min().item(), self.base_ang_vel.max().item())

        if torch.isnan(self.base_pos).any() or torch.isinf(self.base_pos).any() or (self.base_pos.abs() > 1e5).any():
            bad_env_ids = torch.where(torch.isnan(self.base_pos).any(dim=1) |
                                      torch.isinf(self.base_pos).any(dim=1) |
                                      (self.base_pos.abs() > 1e5).any(dim=1))[0]
            print("[ERROR] Abnormal base_pos in envs:", bad_env_ids.cpu().numpy())
            print("  → base_pos min/max:", self.base_pos.min().item(), self.base_pos.max().item())

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_last_dof_pos[:] = self.last_dof_pos[:]
        self.last_dof_pos[:] = self.dof_pos[:]

    def check_termination(self):
        """ Check if environments need to be reset """

        # 1. 接触力过大：如摔倒触地
        contact_force = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        contact_reset = torch.any(contact_force > 1., dim=1)
        self.reset_buf = contact_reset
        if contact_reset.any():
            print("[TERMINATE] contact_force trigger", contact_force[contact_reset])

        # 2. 时间超时
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        if self.time_out_buf.any():
            print("[TERMINATE] timeout trigger")

        # 3. DOF速度爆炸
        max_dof_vel = torch.abs(self.dof_vel.max(dim=1).values)
        self.dof_vel_out = (max_dof_vel > self.cfg.curriculum.dof_vel_limit) & (self.real_episode_length_buf > self.unactuated_time)
        self.reset_buf |= self.dof_vel_out
        if self.dof_vel_out.any():
            print(f"[TERMINATE] dof_vel trigger: max_dof_vel = {max_dof_vel[self.dof_vel_out]}")

        # 4. base速度过大（如掉下来冲击）
        base_vel_norm = torch.norm(self.base_lin_vel[:, :3], dim=-1)
        self.base_vel_out = (base_vel_norm > self.cfg.curriculum.base_vel_limit) & (self.real_episode_length_buf > self.unactuated_time)
        self.reset_buf |= self.base_vel_out
        if self.base_vel_out.any():
            print(f"[TERMINATE] base_vel trigger: base_vel_norm = {base_vel_norm[self.base_vel_out]}")

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # 放在 reset_idx() 的最前面
        if not hasattr(self, "_debug_first_reset_step"):
            self._debug_first_reset_step = None

        if self._debug_first_reset_step is None and len(env_ids) > 0:
            self._debug_first_reset_step = int(self.episode_length_buf.max().item())
            print(f"[DEBUG] First reset triggered at step: {self._debug_first_reset_step}")

        print(f"[DEBUG] Called reset_idx, env_ids = {env_ids}")
        # assert len(env_ids) > 0, "[FATAL] reset_idx got empty env_ids!"
        if len(env_ids) == 0:
            print(f"[WARN] No envs to reset! Skipping reset_idx.")
            return
            
        self.extras["episode"] = {}
        self.extras['episode']['base_height'] = self.old_headheight[env_ids].mean()

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.update_force_curriculum(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_last_dof_pos[env_ids] = 0
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.real_episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.old_headheight[env_ids] = 0
        self.max_headheight[env_ids] = 0
        self.feet_ori[env_ids] = 0
        # fill extras
        self.delay_buffer[:, env_ids, :] = 0.
        
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
        # self._reset_motions(env_ids)
        self.extras['episode']["force"] = self.force.mean()
        self.extras['episode']['action_scale'] = self.action_rescale

        # reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx[env_ids] = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(len(env_ids), ), device=self.device)

        # === 调试点 3：验证 dof_state 是否同步成功 ===
        with torch.no_grad():
            # 将 dof_state 恢复为 [num_envs, num_dof, 2]
            dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)

            pos_ok = torch.allclose(dof_state_view[env_ids, :, 0], self.dof_pos[env_ids], atol=1e-4)
            vel_ok = torch.allclose(dof_state_view[env_ids, :, 1], self.dof_vel[env_ids], atol=1e-4)

            if not pos_ok or not vel_ok:
                print(f"[ERROR][reset_idx] dof_state sync failed for some envs in {env_ids}")
                print("  → pos diff max:", (dof_state_view[env_ids, :, 0] - self.dof_pos[env_ids]).abs().max().item())
                print("  → vel diff max:", (dof_state_view[env_ids, :, 1] - self.dof_vel[env_ids]).abs().max().item())
            else:
                print(f"[OK][reset_idx] dof_state is in sync with dof_pos/dof_vel ✓")


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # 如果不是 Gaussian 风格的 reward 聚合方式，暂不支持
        if not self.is_gaussian:
            raise NotImplementedError
        else:
            # 初始化 rew_buf 全为 0（每个 reward group 的总和）
            self.rew_buf[:, :] = 0
            # 找到 'task' 所在 reward group 的索引（乘积聚合放在 task reward 中）
            task_group_index = self.reward_groups.index('task')
            # 将 task reward 的初始值设为 1，便于之后逐项乘法
            self.rew_buf[:, task_group_index] = 1
            # 遍历所有启用的 reward function
            for i in range(len(self.reward_functions)):
                # 当前 reward 的名字（如 head_height）
                name = self.reward_names[i]
                # 调用 reward 函数并乘以对应的权重
                rew = self.reward_functions[i]() * self.reward_scales[name]
                # 如果返回的是 shape = [n, 1]，就 squeeze 到 shape = [n]
                if len(rew.shape) == 2 and rew.shape[1] == 1:
                    rew = rew.squeeze(1)
                # 乘法聚合：与当前 task reward 相乘（Gaussian 聚合方式）
                # 参考论文 Learning to Get Up: "product of reward terms"
                self.rew_buf[:, task_group_index] *= rew # follow "Learning to Get Up"
                # 保存当前 reward 项的累计值（用于 TensorBoard 分析）
                self.episode_sums[name] += rew

        # 处理所有约束项（如 smoothness、joint limit 等），用于正则项 reward
        for i in range(len(self.constraints)):
            # 当前正则项的名称
            name = self.constraint_names[i]
            # 将其划归对应 reward group，如 'regu'、'style'
            reward_group_name = name.split('_')[0]
            # 计算该约束 reward 并缩放
            rew = self.constraints[i]() * self.constraints_scales[name]
            # 找到对应 reward group 的索引
            task_group_index = self.reward_groups.index(reward_group_name)

            # 加和（正则项 reward 多为负值，采用加法聚合）
            self.rew_buf[:, task_group_index] += rew
            self.episode_sums[name] += rew
            # 如果开启了 only_positive_rewards，则 clip 正则项 reward 到最小为0（避免 reward 为负）
            if self.cfg.constraints.only_positive_rewards:
                self.rew_buf[:, task_group_index] = torch.clip(self.rew_buf[:, task_group_index], min=0.)

        # 若 constraints 中包含 'termination' 项，单独处理终止惩罚（通常为负值）
        if "termination" in self.constraints_scales:
            rew = self._reward_termination() * self.constraints_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        # 每个 reward group 的最终值，累加到 episode_sums 中（用于 TensorBoard）
        for rg in self.reward_groups:
            idx = self.reward_groups.index(rg)
            self.episode_sums[rg] = self.rew_buf[:, idx]

    def compute_observations(self):
        """ Computes observations with debugging logs to locate potential explosion. """

        # ========== 逐项检查输入 ==========
        def check_tensor(name, tensor, threshold=1e5):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any() or tensor.abs().max() > threshold:
                bad_envs = torch.where(
                    torch.isnan(tensor).any(dim=1) |
                    torch.isinf(tensor).any(dim=1) |
                    (tensor.abs() > threshold).any(dim=1)
                )[0]
                print(f"[ERROR] {name} abnormal in envs:", bad_envs.cpu().numpy())
                print(f"  → {name} min/max:", tensor.min().item(), tensor.max().item())
            else:
                print(f"[OK] {name} range: {tensor.min().item()} ~ {tensor.max().item()}")

        check_tensor("base_ang_vel", self.base_ang_vel)
        check_tensor("projected_gravity", self.projected_gravity)
        check_tensor("dof_pos", self.dof_pos)
        check_tensor("dof_vel", self.dof_vel)
        check_tensor("actions", self.actions)
        check_tensor("action_rescale", self.action_rescale)

        # ========== 拼接观测向量 ==========
        current_obs = torch.cat((
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.dof_pos * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.action_rescale + (torch.rand_like(self.action_rescale) - 0.5) * 0.05,
        ),dim=-1)

        check_tensor("current_obs (before noise)", current_obs)

        # ========== 加噪声 ==========
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec

        # ========== 忽略起始阶段 ==========
        current_obs *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time

        check_tensor("current_obs (after noise & unactuated mask)", current_obs)

        # ========== 拼接历史观测 ==========
        self.obs_buf = torch.cat((
            self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length],
            current_obs
        ), dim=-1)

        check_tensor("final obs_buf", self.obs_buf)
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
    
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # hard limits
                self.dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_link_index].mass = self.default_rigid_body_mass[self.torso_link_index] + self.payload[env_id, 0]

        if self.cfg.domain_rand.randomize_com_displacement:
            props[self.torso_link_index].com = self.default_com_torso + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(0, len(props)):
                if i == self.torso_link_index:
                    pass
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # if self.cfg.domain_rand.push_robots and (
        #     self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        # ):
        #     self._push_robots()
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy

        # 添加调试：生成的随机速度
        push_vel = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device)
        print("[DEBUG][push_robots] generated push_vel min/max:",
              push_vel.min().item(), push_vel.max().item())

        # 写入线速度
        self.root_states[:, 7:10] = push_vel
        print("[DEBUG][push_robots] updated root_states[:, 7:10] lin_vel min/max:",
              self.root_states[:, 7:10].min().item(), self.root_states[:, 7:10].max().item())

        # 写入 sim
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        # print('push_robots',self.root_states[:, 7:10])
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.action_rescale
        # print('actions_scaled',actions_scaled)
        self.joint_pos_target = self.dof_pos + actions_scaled
        # self.cfg.domain_rand.delay=False
        if self.cfg.domain_rand.delay:
            self.delay_buffer = torch.concat((self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0)
            self.joint_pos_target = self.dof_pos + self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
        else:
            self.joint_pos_target = self.dof_pos + actions_scaled
        # print('joint_pos_target',self.joint_pos_target)
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains *  self.Kd_factors * self.dof_vel
            # print('p_gains:',self.p_gains,'Kp_factors',self.Kp_factors,'joint_pos_target',self.joint_pos_target,'dof_pos',self.dof_pos,'d_gains',self.d_gains,'Kd_factors',self.Kd_factors,'dof_vel',self.dof_vel)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # print('torques1',torques)
        torques = self.motor_strength *  torques + self.actuation_offset
        # print('torques',torques)
        # ================= 异常值检测 & 定向打印 =================
        # 条件：dof_vel 或 torques 出现异常大值，打印对应 env_id 的数据
        dof_vel_abs = self.dof_vel.abs().max(dim=1).values  # [num_envs]
        torque_abs = torques.abs().max(dim=1).values        # [num_envs]

        threshold_vel = 1000.0
        threshold_torque = 1000.0

        abnormal_envs = torch.nonzero((dof_vel_abs > threshold_vel) | (torque_abs > threshold_torque)).squeeze(-1)

        if len(abnormal_envs) > 0:
            print("=" * 60)
            print(f"[DEBUG][compute_torques] ⚠️ Abnormal envs detected: {abnormal_envs.tolist()}")
            for eid in abnormal_envs.tolist():
                print(f"--- Env {eid} ---")
                print("  → actions_scaled:", actions_scaled[eid].cpu().numpy())
                print("  → dof_pos:", self.dof_pos[eid].cpu().numpy())
                print("  → dof_vel:", self.dof_vel[eid].cpu().numpy())
                print("  → joint_pos_target:", self.joint_pos_target[eid].cpu().numpy())
                print("  → torques:", torques[eid].cpu().numpy())
            print("=" * 60)

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device) + int(self.cfg.domain_rand.random_pose) * torch_rand_float(-1, 1, (len(env_ids), self.num_dof), device=self.device) 
            self.dof_vel[env_ids] = 0.

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # === 调试点 1：检查 dof_pos / dof_vel 数值是否正常 ===
        with torch.no_grad():
            dof_pos_val = self.dof_pos[env_ids]
            dof_vel_val = self.dof_vel[env_ids]
            if not torch.isfinite(dof_pos_val).all():
                print(f"[ERROR][reset_dofs] NaN/INF in dof_pos! env_ids = {env_ids}")
            if not torch.isfinite(dof_vel_val).all():
                print(f"[ERROR][reset_dofs] NaN/INF in dof_vel! env_ids = {env_ids}")
            print(f"[DEBUG][reset_dofs] dof_pos range: {dof_pos_val.min():.2e} ~ {dof_pos_val.max():.2e}")
            print(f"[DEBUG][reset_dofs] dof_vel range: {dof_vel_val.min():.2e} ~ {dof_vel_val.max():.2e}")

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            # print("[DEBUG][reset_root_states] Step 1 base_init_state assigned:", self.root_states[env_ids])

            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # print("[DEBUG][reset_root_states] Step 2 added env_origins:", self.root_states[env_ids, :3])

            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)
            # print("[DEBUG][reset_root_states] Step 3 added random xy offset:", self.root_states[env_ids, :2])
        else:
            self.root_states[env_ids] = self.base_init_state
            # print("[DEBUG][reset_root_states] Step 1 base_init_state assigned:", self.root_states[env_ids])

            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # print("[DEBUG][reset_root_states] Step 2 added env_origins:", self.root_states[env_ids, :3])

        # 可选：打印所有 root_state（pos+quat+vel）
        # print("[DEBUG][reset_root_states] Final root_states[env_ids]:", self.root_states[env_ids])

        # 正式写入 sim 环境中
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # === 调试点 2：检查 root_states 数值是否正常 ===
        with torch.no_grad():
            rs = self.root_states[env_ids]
            if not torch.isfinite(rs).all():
                bad_envs = env_ids[~torch.isfinite(rs).all(dim=1)]
                print(f"[ERROR][reset_root] NaN/INF in root_states! bad_envs = {bad_envs}")
            print(f"[DEBUG][reset_root] root_states pos range: {rs[:, :3].min():.2e} ~ {rs[:, :3].max():.2e}")
            print(f"[DEBUG][reset_root] root_states vel range: {rs[:, 7:13].min():.2e} ~ {rs[:, 7:13].max():.2e}")

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_force_curriculum(self, env_ids):
        if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
            self.force[env_ids] = (self.force[env_ids] - 20).clamp(0, np.inf)
            self.action_rescale[env_ids] = (self.action_rescale[env_ids] - 0.02).clamp(0.25, np.inf)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        start_index = 6
        noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)# if not self.cfg.env.add_force else torch.zeros(start_index + 3 * self.num_actions + 1, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level

        noise_vec[start_index:start_index+self.num_real_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[start_index+self.num_real_dofs:start_index+2*self.num_real_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[start_index+2*self.num_real_dofs:start_index+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
            初始化用于存储仿真状态和处理结果的 Torch 张量
        """
        # get gym GPU state tensors
        # 获取所有 actor 的根状态（位置、姿态、线速度、角速度），shape = (num_actors, 13)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取关节状态（角度、速度），shape = (num_dofs, 2)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取每个刚体上的接触力（用于判断落地/支撑等信息）
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # 获取每个刚体的位置、旋转、速度等状态信息
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # 刷新张量，使得值是最新的（注意这些是 PhysX 内部的数据）
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 用 gymtorch.wrap_tensor 将原始 gym 张量包装成 torch.Tensor，便于计算
        # 所有机器人 base 的状态信息
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # 所有关节的角度和速度
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 每个环境中每个刚体的状态
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        # 当前关节角度
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # 当前关节角速度
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # [角度, 速度] 二者合并
        self.dof_states = self.dof_state.view(self.num_envs, self.num_dof, 2)
        # base 的四元数姿态
        self.base_quat = self.root_states[:, 3:7]
        # 转换为欧拉角（roll, pitch, yaw）
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        # base 的位置（x, y, z）
        self.base_pos = self.root_states[:self.num_envs, 0:3]

        # 每个刚体的 contact force（向量 xyz）
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        # 足端位置
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        # 足端姿态（四元数）
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        # 足端速度
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        # initialize some data used later on
        # 初始化其他关键缓存
        # 当前 step 数，用于判断训练时间等
        self.common_step_counter = 0
        # 存储额外信息供 PPO 等模块使用
        self.extras = {}
        # 观测噪声大小因子，用于训练时加噪增强鲁棒性
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 重力向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 向前向量（X轴）
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # 控制相关缓存
        # 用于存储关节控制力
        self.torques = torch.zeros(self.num_envs, self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # 每个关节的 P 增益
        self.p_gains = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # 每个关节的 D 增益
        self.d_gains = torch.zeros(self.num_real_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # 当前策略输出动作
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 上一次策略动作
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 上上次策略动作（用于正则化 smoothness）
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 上一次的关节速度
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        # 上一次 base 的速度信息
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # 上一次的关节位置
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        # 上上次的关节位置
        self.last_last_dof_pos = torch.zeros_like(self.dof_pos)
        # 控制指令（x vel, y vel, yaw vel）
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # 每项命令缩放因子
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        # 记录每只脚离地持续时间
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        # 记录上一步是否接触地面
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)

        # 计算 base 的局部速度（四元数旋转到 robot local frame）
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 重力向量投影到 base local frame
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 记录头部上一时刻的高度（可用于 reward 中计算上升趋势）
        self.old_headheight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        # 记录历史最大头高
        self.max_headheight = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        # 足部朝向，用于 reward 中判断落地时姿态
        self.feet_ori = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)

        # curriculum 助力大小
        self.force = self.cfg.curriculum.force * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        # 动作缩放因子
        self.action_rescale = self.cfg.control.action_scale * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1)
        # 用于模拟动作延迟的环形缓冲区
        self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # joint positions offsets and PD gains
        # 设置每个关节的默认位置和目标位置（站立姿态）
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_dof_pos = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            self.target_dof_pos[:, i] = self.cfg.init_state.target_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys(): # 遍历查找是否指定了增益
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name] 
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        # shape: (1, dof)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        #randomize kp, kd, motor strength
        # 随机化控制相关参数（domain randomization）
        self.Kp_factors = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # actuation delay offset
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        # observation noise for height
        self.height_noise_offset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # 电机强度
        self.motor_strength = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.domain_rand.randomize_kp:
            # 每个关节采样 Kp 随机因子
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            # 每个关节采样 Kd 随机因子
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_dofs), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            # 在 base 上加载不同 payload
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            # 放大位移量，增加扰动
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 4
            self.com_displacement[:, 1] = self.com_displacement[:, 1] * 4
            self.com_displacement[:, 2] = self.com_displacement[:, 2] * 2
        if self.cfg.domain_rand.delay:
            # 每个环境设置不同的动作延迟 index（从 delay buffer 中取动作）
            self.delay_idx = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(self.num_envs,), device=self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= 1 #self.dt
        
        for key in list(self.constraints_scales.keys()):
            scale = self.constraints_scales[key]
            if scale==0:
                self.constraints_scales.pop(key) 
            else:
                self.constraints_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.reward_functions.append(getattr(self, name))

        self.constraints = []
        self.constraint_names = []
        for name, scale in self.constraints_scales.items():
            # import ipdb; ipdb.set_trace()
            self.constraint_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.constraints.append(getattr(self, name))        

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        for name in self.constraint_names:
            self.episode_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        print(f"Loaded asset: {asset_file}")
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s and 'auxiliary' not in s]
        penalized_contact_names = []
        # import ipdb; ipdb.set_trace()
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.torso_link_index = body_names.index("base_link")

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            self.com_displacement[:, 0] = self.com_displacement[:, 0] * 4
            self.com_displacement[:, 1] = self.com_displacement[:, 1] * 4
            self.com_displacement[:, 2] = self.com_displacement[:, 2] * 2

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

            if i == 0:
                # self.default_com = copy.deepcopy(body_props[0].com)
                self.default_com_torso = copy.deepcopy(body_props[self.torso_link_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
            
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        left_foot_names = [s for s in body_names if self.cfg.asset.left_foot_name in s and 'keyframe' not in s and 'auxiliary' not in s]
        right_foot_names = [s for s in body_names if self.cfg.asset.right_foot_name in s and 'keyframe' not in s and 'auxiliary' not in s]
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])

        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        base_name = [s for s in body_names if self.cfg.asset.base_name in s]
        self.base_indices = torch.zeros(len(base_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_name)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name[i])

        # import ipdb; ipdb.set_trace()
        left_knee_names = [s for s in body_names if self.cfg.asset.left_knee_name in s and 'keyframe' not in s]
        right_knee_names = [s for s in body_names if self.cfg.asset.right_knee_name in s and 'keyframe' not in s]
        self.left_knee_indices = torch.zeros(len(left_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_knee_names)):
            self.left_knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_knee_names[i])
        self.right_knee_indices = torch.zeros(len(right_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_knee_names)):
            self.right_knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_knee_names[i])

        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])

        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        

        self.keyframe_names = [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.head_names = [s for s in body_names if self.cfg.asset.head_name in s]
        # import ipdb; ipdb.set_trace()
        self.head_indices = torch.zeros(len(self.head_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.head_names):
            self.head_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.left_hip_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))

        self.left_hip_roll_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_roll_joints)):
            self.left_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_roll_joints[i])
            
        self.right_hip_roll_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_roll_joints)):
            self.right_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_roll_joints[i])
            
        self.hip_roll_joint_indices = torch.cat((self.left_hip_roll_joint_indices, self.right_hip_roll_joint_indices))

        self.left_knee_joint_indices = torch.zeros(len(self.cfg.asset.left_knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_knee_joints)):
            self.left_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_knee_joints[i])
            
        self.right_knee_joint_indices = torch.zeros(len(self.cfg.asset.right_knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_knee_joints)):
            self.right_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_knee_joints[i])


        self.left_hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_pitch_joints)):
            self.left_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_pitch_joints[i])
            
        self.right_hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_pitch_joints)):
            self.right_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_pitch_joints[i])
            
        self.hip_pitch_joint_indices = torch.cat((self.left_hip_pitch_joint_indices, self.right_hip_pitch_joint_indices))

        self.all_hip_joint_indices = torch.cat([self.hip_pitch_joint_indices, self.hip_roll_joint_indices, self.hip_joint_indices])


        # left
        self.left_leg_joints_indices = torch.zeros(len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joints_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])
        self.right_leg_joints_indices = torch.zeros(len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joints_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])
        


        # Remove arm and upper body related code
        self.lower_body_joint_indices = torch.cat([self.all_hip_joint_indices, self.knee_joint_indices, self.ankle_joint_indices])

        left_lower_body_names = []
        for target_name in self.cfg.asset.left_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    left_lower_body_names.append(source_name)
        self.left_lower_body_indices = torch.zeros(len(left_lower_body_names), dtype=torch.long, device=self.device)
        self.left_lower_body_names = left_lower_body_names
        for i, name in enumerate(left_lower_body_names):
            self.left_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_lower_body_names = []
        for target_name in self.cfg.asset.right_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name and 'aux' not in source_name:
                    right_lower_body_names.append(source_name)
        self.right_lower_body_indices = torch.zeros(len(right_lower_body_names), dtype=torch.long, device=self.device)
        self.right_lower_body_names = right_lower_body_names
        for i, name in enumerate(right_lower_body_names):
            self.right_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        left_ankle_names = []
        for target_name in self.cfg.asset.left_ankle_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name:
                    left_ankle_names.append(source_name)
        self.left_ankle_indices = torch.zeros(len(left_ankle_names), dtype=torch.long, device=self.device)
        self.left_ankle_names = left_ankle_names
        for i, name in enumerate(left_ankle_names):
            self.left_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        right_ankle_names = []
        for target_name in self.cfg.asset.right_ankle_names:
            for source_name in body_names:
                if target_name in source_name and 'keyframe' not in source_name:
                    right_ankle_names.append(source_name)
        self.right_ankle_indices = torch.zeros(len(right_ankle_names), dtype=torch.long, device=self.device)
        self.right_ankle_names = right_ankle_names
        for i, name in enumerate(right_ankle_names):
            self.right_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    # 参数 cfg 是环境配置对象（如 ZqCfg），它包含控制参数、奖励函数设置、指令范围
    def _parse_cfg(self, cfg):
        # self.sim_params.dt 是单步物理仿真的时间间隔（如 0.005s）
        # self.cfg.control.decimation 是每个策略动作重复多少个 sim step
        # print(f"[INFO] decimation = {self.cfg.control.decimation}, sim_dt = {self.sim_params.dt}, policy_dt = {self.dt}")
        # [INFO] decimation = 10, sim_dt = 0.004999999888241291, policy_dt = 0.04999999888241291
        # sim_dt = 0.005, 表示仿真时间频率 f = 1/0.005 = 200 Hz
        # decimation = 10, 表示策略频率为仿真频率的 1/10, f' = 200/10 = 20 Hz, 也就是: 每 0.05 秒执行一次策略网络前向推理（action 输出）, 每输出一次 action，会连续执行 10 个仿真子步来应用该动作
        self.dt = self.cfg.control.decimation * self.sim_params.dt

        # 保存观测值的归一化尺度（一般用于标准化 observation 中的位置、速度等）。
        # 这有助于加速训练收敛并提升数值稳定性。
        self.obs_scales = self.cfg.normalization.obs_scales

        # 将 cfg.rewards.scales 类结构转成字典。
        # 每个奖励项都会有一个权重，这里把它们提取出来备用，便于在计算 reward 时逐项加权。
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)

        # 同样将 constraints 中的每项惩罚（软约束）项的权重也转为字典保存
        self.constraints_scales = class_to_dict(self.cfg.constraints.scales)

        # 提取机器人控制任务中的指令范围，如目标速度、方向等上下限。
        # 一般用于训练 phase 中的 random command sampling。
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        # 环境最大 episode 时间（单位：秒），比如设定一次训练最多 10 秒
        self.max_episode_length_s = self.cfg.env.episode_length_s

        # 将最大 episode 时间（秒）转换为最大步数（以策略步为单位）。
        # 比如：如果 episode 长度为 10 秒，dt = 0.02 → 共 500 个 policy steps。
        # np.ceil 向上取整，确保不会提前结束。
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # 把 curriculum 中**扰动间隔（单位：秒）**转换成策略步数单位。
        # 也就是说每隔这么多步，就可能触发一次随机推力扰动。
        # 举例：如果 push_interval_s = 10 秒、dt = 0.02 → 实际为每 500 步检查是否扰动。
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws target body position and orientation
        """
        self.gym.clear_lines(self.viewer)
        self._refresh_tensor_state()
        terrain_sphere = gymutil.WireframeSphereGeometry(0.02, 5, 5, None, color=(1, 1, 0))
        marker_sphere = gymutil.WireframeSphereGeometry(0.03, 5, 5, None, color=(0, 1, 1))
        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in range(self.num_envs):
            base_pos = self.base_pos[i].cpu().numpy()

            motion_body_pos = self.motion_dict["keyframe_pos"][i].clone()
            motion_body_pos = motion_body_pos.cpu().numpy()
            motion_body_quat = self.motion_dict["keyframe_quat"][i].cpu().numpy()

            for j in range(len(self.keyframe_indices)):
                x, y, z = motion_body_pos[j, 0], motion_body_pos[j, 1], motion_body_pos[j, 2]
                a, b, c, d = motion_body_quat[j, 0], motion_body_quat[j, 1], motion_body_quat[j, 2], motion_body_quat[j, 3]
                target_sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(a, b, c, d))
                gymutil.draw_lines(marker_sphere, self.gym, self.viewer, self.envs[i], target_sphere_pose)
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_sphere_pose)

    def _refresh_tensor_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
    #-----------------------------task rewards-----------------------------
    def _reward_orientation(self):
        if not self.is_gaussian:
            mse_error = torch.sum(torch.square(self.projected_gravity - torch.tensor([0, 0 ,1], device=self.device)), dim=-1)
            reward = torch.exp(mse_error/self.cfg.rewards.orientation_sigma) * (self.root_states[:, 2] > 0.4)
        else:
            base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
            reward = tolerance(-self.projected_gravity[:, 2], [self.cfg.rewards.orientation_threshold, np.inf], 1., 0.05) #-1
        print(f"[REWARD] _reward_orientation: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward

    def _reward_head_height(self):
        if not self.is_gaussian:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            reward = head_height.squeeze(1).clamp(0, 1)
            print(f"[REWARD] _reward_head_height (non-Gaussian): mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
            return reward
        else:
            head_height = self.rigid_body_states[:, self.head_indices, 2].clone()
            feet_height = self.rigid_body_states[:, self.feet_indices, 2].clone().mean(-1).unsqueeze(-1)
            head_height -= feet_height
            reward = tolerance(head_height, (self.cfg.rewards.target_head_height, np.inf), self.cfg.rewards.target_head_margin, 0.1)
            delta_max_headheight = head_height - self.max_headheight
            delta_headheight = head_height - self.old_headheight
            self.max_headheight = torch.max(torch.cat((head_height, self.old_headheight), dim=1), dim=1)[0].unsqueeze(-1)
            self.old_headheight = head_height
            print(f"[REWARD] _reward_head_height (Gaussian): mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
            return reward


    #-----------------------------regularization rewards-----------------------------
    def _reward_dof_acc(self):
        acc = (self.last_dof_vel - self.dof_vel) / self.dt
        reward = torch.sum(torch.square(acc), dim=1)

        # 基本统计信息打印
        print(f"[REWARD] _reward_dof_acc: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")

        # 异常加速度范围检测（可选）
        if torch.any(torch.abs(acc) > 1e3):
            print("[DEBUG] _reward_dof_acc abnormal acc detected")
            print(f"        acc max: {acc.max().item():.2e}, acc min: {acc.min().item():.2e}")
            print(f"        dof_vel max: {self.dof_vel.max().item():.2e}, min: {self.dof_vel.min().item():.2e}")
            print(f"        last_dof_vel max: {self.last_dof_vel.max().item():.2e}, min: {self.last_dof_vel.min().item():.2e}")

        return reward

    def _reward_action_rate(self):
        reward = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        print(f"[REWARD] _reward_action_rate: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward

    # 二阶动作变化（即 jerk）
    def _reward_smoothness(self):
        # second order smoothness
        jerk = self.actions - 2 * self.last_actions + self.last_last_actions
        reward = torch.sum(torch.square(jerk), dim=1)
        print(f"[REWARD] _reward_smoothness: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward
    def _reward_torques(self):
        reward = torch.sum(torch.square(self.torques), dim=1)
        print(f"[REWARD] _reward_torques: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward
    def _reward_joint_power(self):
        #Penalize high power
        reward = torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)
        print(f"[REWARD] _reward_joint_power: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward
    def _reward_dof_vel(self):
        # Penalize dof velocities
        reward = torch.sum(torch.square(self.dof_vel), dim=1)
        print(f"[REWARD] _reward_dof_vel: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward

    def _reward_joint_tracking_error(self):
        reward = torch.sum(torch.square(self.joint_pos_target - self.dof_pos), dim=-1)
        print(f"[REWARD] _reward_joint_tracking_error: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        reward = torch.sum(out_of_limits, dim=1)
        print(f"[REWARD] _reward_dof_pos_limits: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        reward = torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
        print(f"[REWARD] _reward_dof_vel_limits: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        reward = torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
        print(f"[REWARD] _reward_torque_limits: mean={reward.mean().item():.4f}, max={reward.max().item():.4f}, min={reward.min().item():.4f}")
        return reward

    #-----------------------------style rewards-----------------------------
    def _reward_waist_deviation(self):
        wrist_dof = self.dof_pos[:, self.waist_joint_indices]
        reward = (torch.abs(wrist_dof) > 1.4).float()
        print(f"[REWARD] _reward_waist_deviation: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward.squeeze(1)

    def _reward_hip_yaw_deviation(self):
        hip_yaw_dof = self.dof_pos[:, self.hip_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 1.4) | (torch.min(torch.abs(self.dof_pos[:, self.hip_joint_indices]), dim=-1)[0] > 0.9)
        reward_float = reward.float()
        print(f"[REWARD] _reward_hip_yaw_deviation: mean={reward_float.mean():.4f}, max={reward_float.max():.1f}, min={reward_float.min():.1f}")
        return reward

    def _reward_hip_roll_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.hip_roll_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] >  1.4) | (torch.min(torch.abs(self.dof_pos[:, self.hip_roll_joint_indices]), dim=-1)[0] > 0.9)
        reward_float = reward.float()
        print(f"[REWARD] _reward_hip_roll_deviation: mean={reward_float.mean():.4f}, max={reward_float.max():.1f}, min={reward_float.min():.1f}")
        return reward

    def _reward_left_foot_displacement(self):
        base_xy = self.root_states[:, :2].clone()
        left_foot_xy = self.rigid_body_states[:, self.left_foot_indices, :2].squeeze(1)
        mse=torch.sum(torch.square(base_xy - left_foot_xy), dim=-1)
        mse_error = mse.clamp(0.3, np.inf)
        # mse_error = mse.clamp(0.025, np.inf) #better standing style
        # mse_error -=0.02 #better standing style
        reward = torch.exp(mse_error * self.cfg.rewards.left_foot_displacement_sigma) *  (self.rigid_body_states[:, self.left_foot_indices, 2] < 0.3).squeeze(1)

        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        # standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2 #better standing style
        print(f"[REWARD] _reward_left_foot_displacement: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return reward * standup

    def _reward_right_foot_displacement(self):
        base_xy = self.root_states[:, :2].clone()
        right_foot_xy = self.rigid_body_states[:, self.right_foot_indices, :2].squeeze(1)
        mse=torch.sum(torch.square(base_xy - right_foot_xy), dim=-1)
        mse_error = mse.clamp(0.3, np.inf)
        # mse_error = mse.clamp(0.025, np.inf) #better standing style
        # mse_error -=0.02 #better standing style
        reward = torch.exp(mse_error * self.cfg.rewards.right_foot_displacement_sigma) * (self.rigid_body_states[:, self.right_foot_indices, 2] < 0.15).squeeze(1)

        # standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase2 #better standing style
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        print(f"[REWARD] _reward_right_foot_displacement: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward * standup

    def _reward_knee_deviation(self):
        hip_roll_dof = self.dof_pos[:, self.knee_joint_indices]
        reward = (torch.max(torch.abs(self.dof_pos[:, self.knee_joint_indices]), dim=-1)[0] > 2.85) | (torch.min(self.dof_pos[:, self.knee_joint_indices], dim=-1)[0] < -0.06)
        reward_float = reward.float()
        print(f"[REWARD] _reward_knee_deviation: mean={reward_float.mean():.4f}, max={reward_float.max():.4f}, min={reward_float.min():.4f}")
        return reward

    def _reward_shank_orientation(self):
        left_knee_pos = self.rigid_body_states[:, self.left_knee_indices, :3].clone()
        right_knee_pos = self.rigid_body_states[:, self.right_knee_indices, :3].clone()
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()

        left_feet_orientation = (left_knee_pos - left_foot_pos)[:, :, 2] / torch.norm(left_knee_pos - left_foot_pos, dim=-1)
        right_feet_orientation = (right_knee_pos - right_foot_pos)[:, :, 2] / torch.norm(right_knee_pos - right_foot_pos, dim=-1)

        feet_orientation = torch.mean(torch.concat([left_feet_orientation, right_feet_orientation], dim=-1), dim=-1)

        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        reward = tolerance(feet_orientation, [0.8, np.inf], 1, 0.1) * base_height#.unsqueeze(1) 

        if self.cfg.constraints.post_task:
            standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        print(f"[REWARD] _reward_shank_orientation: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return reward

    def _reward_ground_parallel(self):
        left_ankle_pos = self.rigid_body_states[:, self.left_ankle_indices, 2].clone() * 10
        right_ankle_pos = self.rigid_body_states[:, self.right_ankle_indices, 2].clone() * 10
        var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
        var = torch.mean(torch.concat([left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1), dim=-1)
        reward = var < 0.05

        if self.cfg.constraints.post_task:
            standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
            reward = reward * ~standup + torch.ones_like(reward) * standup
        print(f"[REWARD] _reward_ground_parallel: mean={reward.float().mean():.4f}, max={reward.float().max():.1f}, min={reward.float().min():.1f}")

        return reward

    def _reward_feet_distance(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=-1)
        reward = tolerance(feet_distances, [0, 0.4], 0.3, 0.05)
        # return (feet_distances > 0.33).squeeze(1) #better standing style
        print(f"[REWARD] _reward_feet_distance: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return (feet_distances > 0.45).squeeze(1)


    def _reward_style_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1
        reward = torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height
        print(f"[REWARD] _reward_style_ang_vel_xy: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward
    def _reward_soft_symmetry_action(self):
        left_body_action = self.actions[:, self.left_leg_joints_indices] # [num_envs, 6]
        right_body_action = self.actions[:, self.right_leg_joints_indices] # [num_envs, 6]
        # negative_indices = torch.tensor([1, 2, 5, 16, 17], device=self.device, dtype=torch.int64)
        negative_indices = torch.tensor([1, 2, 5], device=self.device, dtype=torch.int64)
        left_body_action[:, negative_indices] *= -1
        body_symmetry = torch.norm(left_body_action - right_body_action, dim=-1)
        standup =self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        body_symmetry[~standup] *= 0
        body_symmetry = body_symmetry * torch.clamp(-self.projected_gravity[:, 2], 0, 0.9) / 0.9
        reward = body_symmetry
        print(f"[REWARD] _reward_soft_symmetry_action: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward
    def _reward_soft_symmetry_body(self):
        left_body = self.dof_pos[:, self.left_leg_joints_indices] # [num_envs, 6]
        right_body = self.dof_pos[:, self.right_leg_joints_indices] # [num_envs, 6]
        # negative_indices = torch.tensor([1, 2, 5, 16, 17], device=self.device, dtype=torch.int64)
        negative_indices = torch.tensor([1, 2, 5], device=self.device, dtype=torch.int64)
        left_body[:, negative_indices] *= -1
        error=(torch.abs(left_body-right_body)-0.08).clip(min=0)
        body_symmetry = torch.norm(error, dim=-1)
        standup =self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        body_symmetry[~standup] *= 0
        reward = torch.exp(-body_symmetry/0.025)

        reward[~standup] *= 0
        reward = reward * torch.clamp(-self.projected_gravity[:, 2], 0, 0.9) / 0.9
        print(f"[REWARD] _reward_soft_symmetry_body: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return reward

    #--------------------------post-task rewards-----------------------------
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward=torch.exp(torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * -2) * base_height
        print(f"[REWARD] _reward_ang_vel_xy: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return reward

    def _reward_lin_vel_xy(self):
        # Penalize z axis base linear velocity
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward=torch.exp(torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) * -5) * base_height
        print(f"[REWARD] _reward_lin_vel_xy: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return reward

    def _reward_feet_height_var(self):
        left_foot_height = self.rigid_body_states[:, self.left_foot_indices, 2].clone() * 10
        right_foot_height = self.rigid_body_states[:, self.right_foot_indices, 2].clone() * 10
        feet_distance = torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward=torch.exp(feet_distance * -2) * standup
        print(f"[REWARD] _reward_feet_height_var: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")

        return reward

    def _reward_target_orientation(self):
        # Penalize non flat base orientation
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * -5) * standup
        print(f"[REWARD] _reward_target_orientation: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward
    def _reward_target_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        standup  = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(torch.abs(base_height - self.cfg.rewards.base_height_target) * -20) * standup
        print(f"[REWARD] _reward_target_base_height: mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        return reward
    def _reward_target_lower_dof_pos(self):
        mse = torch.sum(torch.square(self.dof_pos[:, :] - self.target_dof_pos[:, :]), dim=-1)
        standup =self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
        reward = reward * standup
        print(f"[REWARD] _reward_target_lower_dof_pos: reward mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        print(f"[REWARD] _reward_target_lower_dof_pos: standup activated={standup.float().mean():.4f}, max={standup.float().max():.1f}, min={standup.float().min():.1f}")
        return reward
    def _reward_target_upper_dof_pos(self):
        # if self.is_gaussian:
        mse = torch.sum(torch.square(self.dof_pos[:, self.upper_body_joint_indices] - self.target_dof_pos[:, self.upper_body_joint_indices]), dim=-1)
        standup =self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = torch.exp(mse * self.cfg.rewards.target_dof_pos_sigma)
        reward = reward * standup
        print(f"[REWARD] _reward_target_upper_dof_pos: reward mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        print(f"[REWARD] _reward_target_upper_dof_pos: standup activated={standup.float().mean():.4f}, max={standup.float().max():.1f}, min={standup.float().min():.1f}")
        return reward

    def _reward_target_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        reward = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        base_height = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase1

        reward = reward * base_height
        print(f"[REWARD] _reward_target_feet_stumble: reward mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        print(f"[REWARD] _reward_target_feet_stumble: base_height activated={base_height.float().mean():.4f}, max={base_height.float().max():.1f}, min={base_height.float().min():.1f}")
        return reward

    def _reward_target_knee_angle(self):
        knee_angles = self.dof_pos[:, self.knee_joint_indices]
        reward = torch.all(knee_angles > 0, dim=1).float()
        standup = self.root_states[:, 2] > self.cfg.rewards.target_base_height_phase3
        reward = reward * standup

        print(f"[REWARD] _reward_target_knee_angle: reward mean={reward.mean():.4f}, max={reward.max():.4f}, min={reward.min():.4f}")
        print(f"[REWARD] _reward_target_knee_angle: standup activated={standup.float().mean():.4f}, max={standup.float().max():.1f}, min={standup.float().min():.1f}")
        return reward
        