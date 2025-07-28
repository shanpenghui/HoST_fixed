reset_idx()
│
├─> print: dof_vel / obs 清零是否成功？
│
compute_observations()
│
├─> print: obs_buf 是否含爆炸？
│
actor 推理
│
├─> print: actor 输出是否爆炸？
│
step()
│
├─> print: reward_dof_acc 内 dof_vel 跳变？
│
└─> print: torque / dof_vel / joint_power 是否爆炸？
