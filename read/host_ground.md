1. _reward_target_base_height ✓ -> the same
2. _reward_target_orientation ✓ -> the same
3. _reward_target_upper_dof_pos ✓ -> the same
4. _reward_feet_height_var ✓ -> the same
5. _reward_lin_vel_xy ✓ -> the same
6. _reward_ang_vel_xy ✓ -> the same


Diff:
1. _create_envs
2. _reward_shoulder_roll_deviation, pi里面没有这个函数
3. _reward_left_foot_displacement
    pi_host 中对 mse_error 的 clamp 下限为 0.3，落脚高度阈值 < 0.15。
    host 中同样的 mse_error，高度阈值变为 < 0.3，数值更宽松。
4. _reward_right_foot_displacement
    同上，对右脚的判定更宽松（高度阈值从 < 0.15 改为 < 0.3）。
5. _reward_knee_deviation
    判定角度范围更宽：最大值从 > 1.65 变成 > 2.85，最小值从 < -0.6 到 < -0.06。
6. _reward_feet_distance
    reward 计算中 tolerance 函数参数从 mid=0.3 改为 mid=0.38。
    feet 距离从要求 >0.45 变为 >0.9，更宽松。
7. 只在 pi_host_ground.py 中存在的函数（共 6 个）

    | 函数名                            |
    | ------------------------------ |
    | `_push_robots`                 |
    | `_reward_soft_symmetry_action` |
    | `_reward_soft_symmetry_body`   |
    | `_reward_target_lower_dof_pos` |
    | `_reward_target_feet_stumble`  |
    | `_reward_target_knee_angle`    |

8. 只在 host_ground.py 中存在的函数（共 1 个）

    | 函数名                               |
    | --------------------------------- |
    | `_reward_shoulder_roll_deviation` |




