from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ZqCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.351] # x,y,z [m], updated to match Piwaist
        rot = [0.0, -1, 0, 1.0] # x,y,z,w [quat]
        # ZQ robot:
        # leg_l1_joint: hip_roll
        # leg_l2_joint: hip_yaw
        # leg_l3_joint: hip_pitch
        # leg_l4_joint: knee_pitch
        # leg_l5_joint: ankle_pitch
        # leg_l6_joint: ankle_roll
        # PI robot:
        # l_hip_pitch_joint: hip_pitch
        # l_hip_roll_joint: hip_roll
        # l_thigh_joint: knee_yaw
        # l_calf_joint: knee_pitch
        # l_ankle_pitch_joint: ankle_pitch
        # l_ankle_roll_joint: ankle_roll
        target_joint_angles = { # = target angles [rad] when action = 0.0
            # left leg (6 dof)
            "leg_l1_joint": 0.0,  # "l_hip_roll_joint"
            "leg_l2_joint": 0.0,  # hip_yaw -> "l_thigh_joint": knee_yaw
            "leg_l3_joint": -0.0, # "l_hip_pitch_joint"
            "leg_l4_joint": 0.0,  # "l_calf_joint": knee_pitch
            "leg_l5_joint": -0.0, # "l_ankle_pitch_joint"
            "leg_l6_joint": 0,    # "l_ankle_roll_joint"
            # right leg (6 dof)
            "leg_r1_joint": 0.0,  # "r_hip_roll_joint"
            "leg_r2_joint": 0.0,  # hip_yaw -> "r_thigh_joint": knee_yaw
            "leg_r3_joint": -0.0, # "r_hip_pitch_joint"
            "leg_r4_joint": 0.0,  # "r_calf_joint": knee_pitch
            "leg_r5_joint": -0.0, # "r_ankle_pitch_joint"
            "leg_r6_joint": 0,    # "r_ankle_roll_joint"
        }

        default_joint_angles = {
            # left leg (6 dof)
            "leg_l1_joint": 0.0,
            "leg_l2_joint": 0.0,
            "leg_l3_joint": -0.24,
            "leg_l4_joint": 0.48,
            "leg_l5_joint": -0.24,
            "leg_l6_joint": 0,
            # right leg (6 dof)
            "leg_r1_joint": 0.0,
            "leg_r2_joint": 0.0,
            "leg_r3_joint": -0.24,
            "leg_r4_joint": 0.48,
            "leg_r5_joint": -0.24,
            "leg_r6_joint": 0,
        } 

    class env(LeggedRobotCfg.env):
        num_one_step_observations= 43#3+3+3*12+1
        num_actions = 12
        num_dofs = 12
        num_actor_history = 6
        num_observations = num_actor_history * num_one_step_observations
        episode_length_s = 10 # episode length in seconds
        unactuated_timesteps = 30

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'1_joint': 50, '2_joint': 50,'3_joint': 70,
                     '4_joint': 70, '5_joint': 20, '6_joint': 20}
        damping = {'1_joint': 5.0, '2_joint': 5.0,'3_joint': 7.0,
                   '4_joint': 7.0, '5_joint': 2, '6_joint': 2}
        # action scale: target angle = actionRescale * action + cur_dof_pos
        action_scale = 0.25 #1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10

    class terrain:
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.3
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 1 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        terrain_proportions = [1, 0., 0, 0, 0]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/zq_humanoid/urdf/zq_sa01.urdf"
        name = "zqsa01"
        left_foot_name = "leg_l6_link"
        right_foot_name = "leg_r6_link"
        left_knee_name = 'leg_l4_link'
        right_knee_name = 'leg_r4_link'
        foot_name = "6_link"
        penalize_contacts_on = ['base_link']
        terminate_after_contacts_on = ['base_link', "4_link"]


        left_leg_joints = [ 'leg_l1_joint', 'leg_l2_joint','leg_l3_joint', 'leg_l4_joint', 'leg_l5_joint', 'leg_l6_joint']
        right_leg_joints = [  'leg_r1_joint','leg_r2_joint', 'leg_r3_joint','leg_r4_joint', 'leg_r5_joint', 'leg_r6_joint']
       
        left_hip_joints = ['leg_l2_joint']
        right_hip_joints = ['leg_r2_joint']
        left_hip_roll_joints = ['leg_l1_joint']
        right_hip_roll_joints = ['leg_r1_joint']
        left_hip_pitch_joints = ['leg_l3_joint']
        right_hip_pitch_joints = ['leg_r3_joint']

        

        left_knee_joints = ['leg_l4_joint']
        right_knee_joints = ['leg_r4_joint']

        left_arm_joints = ['l_shoulder_pitch_joint', 'l_shoulder_roll_joint', 'l_shoulder_yaw_joint', 'l_elbow_joint', 'l_wrist_roll_joint']
        right_arm_joints = ['r_shoulder_pitch_joint', 'r_shoulder_roll_joint', 'r_shoulder_yaw_joint', 'r_elbow_joint', 'r_wrist_roll_joint']
        waist_joints = ["waist_yaw_joint"]
        knee_joints = ['leg_l4_joint', 'leg_r4_joint']
        ankle_joints = [ 'leg_l5_joint', 'leg_l6_joint', 'leg_r5_joint', 'leg_r6_joint']

        keyframe_name = "keyframe"
        head_name = 'keyframe_head'

        trunk_names = ["base_link"]
        base_name = 'base_link'

      
        left_lower_body_names = ['leg_l3', 'leg_l6', 'leg_l4']
        right_lower_body_names = ['leg_r3', 'leg_r6', 'leg_r4']

        left_ankle_names = ['leg_l6']
        right_ankle_names = ['leg_r6']

        density = 0.001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        base_height_target = 0.827  # updated to match ZQ
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        orientation_sigma = 1
        is_gaussian = True
        target_head_height = 1  # 参考宇树G1 # updated to match ZQ head_height_target (base_height + 0.08)
        target_head_margin = 1
        target_base_height_phase1 = 0.45  # 参考宇树G1 第一个阶段的结束高度门槛（从完全趴倒→翻身）# updated to match Piwaist
        target_base_height_phase2 = 0.45  # 参考宇树G1 多用于奖励逻辑复用 # 0.05 updated to 0.05 to get better standing style
        target_base_height_phase3 = 0.65  # 参考宇树G1 第二阶段的结束门槛（从跪立→站立）# updated to match Piwaist
        orientation_threshold = 0.99
        left_foot_displacement_sigma = -2  #-200 updated to get better standing style
        right_foot_displacement_sigma = -2 #-200 updated to get better standing style
        target_dof_pos_sigma = -0.1
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        reward_groups = ['task', 'regu', 'style', 'target']
        num_reward_groups = len(reward_groups)
        reward_group_weights = [2.5, 0.1, 1, 1]

        class scales:
            task_orientation = 1
            task_head_height = 1

    class constraints( LeggedRobotCfg.rewards ):
        is_gaussian = True
        target_head_height = 1
        target_head_margin = 1
        orientation_height_threshold = 0.9
        target_base_height = 0.45  # 参考宇树G1 # updated to match Piwaist

        left_foot_displacement_sigma = -2  #-200 updated to get better standing style
        right_foot_displacement_sigma = -2 #-200 updated to get better standing style
        hip_yaw_var_sigma = -2
        target_dof_pos_sigma = -0.1
        post_task = False
        
        class scales:
            # regularization reward
            regu_dof_acc = -2.5e-7
            regu_action_rate = -0.01
            regu_smoothness = -0.01 
            regu_torques = -2.5e-6
            regu_joint_power = -2.5e-5
            regu_dof_vel = -1e-3
            regu_joint_tracking_error = -0.00025
            regu_dof_pos_limits = -100.0
            regu_dof_vel_limits = -1 

            # style reward
            # style_waist_deviation = -10
            style_hip_yaw_deviation = -10
            style_hip_roll_deviation = -10
            # style_shoulder_roll_deviation = -2.5
            style_left_foot_displacement = 2.5 #7.5 updated to get better standing style
            style_right_foot_displacement = 2.5 #7.5  updated to get better standing style
            style_knee_deviation = -0.25
            # style_shank_orientation = 10
            style_ground_parallel = 20
            style_feet_distance = -10
            style_style_ang_vel_xy = 1
            # style_soft_symmetry_action=-10  #  updated to get better standing style
            # style_soft_symmetry_body=2.5 # updated to get better standing style

            # post-task reward
            target_ang_vel_xy = 10
            target_lin_vel_xy = 10
            target_feet_height_var = 2.5
            # target_target_lower_dof_pos = 30  #  updated to get better standing style
            target_target_orientation = 10
            target_target_base_height = 10
            # target_target_knee_angle = 10 #  updated to get better standing style

    class domain_rand:
        use_random = True

        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_motor_strength = use_random
        motor_strength_range = [0.9, 1.1]

        randomize_payload_mass = use_random
        payload_mass_range = [-2, 3]

        randomize_com_displacement = use_random
        com_displacement_range = [-0.03, 0.03]

        randomize_link_mass = use_random
        link_mass_range = [0.8, 1.2]
        
        randomize_friction = use_random
        friction_range = [0.1, 1]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.85, 1.15]
        
        randomize_kd = use_random
        kd_range = [0.85, 1.15]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_scale = [0.9, 1.1]
        initial_joint_pos_offset = [-0.1, 0.1]
        
        push_robots = True 
        push_interval_s = 10
        max_push_vel_xy = 0.5

        delay = use_random
        max_delay_timesteps = 5
    
    class curriculum:
        pull_force = True
        # vertical pulling force F 在训练初期，为了帮助机器人更容易探索起身动作，
        # 在 base link 上向上施加一个垂直辅助力，模拟“托起”动作；随着训练进行，逐渐减小，最终为 0
        # Unitree G1 的训练中用了 F = 200N 而代码中实际设置的是 force=100，是因为该项目结构中 力同时施加在两个 link 上（torso+base）→ 总力=100×2=200
        # 如果代码结构是双 link 同时施力（如 torso + base），你应将 force 设置为总力一半（如 force = 50 → 实际 100N）
        force = 150 # 100*2=200 is the actuatl force because of a extra keyframe torso link
        dof_vel_limit = 300
        base_vel_limit = 20
        threshold_height = 0.9 # 参考宇树G1
        no_orientation = True  # 参考宇树G1

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class ZqCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # smoothness
        value_smoothness_coef = 0.1
        smoothness_upper_bound = 1.0
        smoothness_lower_bound = 0.1
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'Zq_ground'
        algorithm_class_name = 'PPO'
        init_at_random_ep_len = True
        max_iterations = 40000 # number of policy updates