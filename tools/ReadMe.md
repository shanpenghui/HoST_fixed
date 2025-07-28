

Put the 'export_tensorboard_log.py' in the train log folder, such as 'legged_gym/logs/Zq_ground/Jul27_23-59-58_test_zq_ground'

Change the folder path in file 'log_export.py' , important!!!

Run the command in conda environment, such as 'conda activate host'
```shell
python log_export.py 
```

```shell
 Episode_action_scale.csv                    Episode_rew_regu_torques.csv                    Episode_rew_target_feet_height_var.csv           model_0.pt      model_700.pt
 Episode_base_height.csv                     Episode_rew_style.csv                           Episode_rew_target_lin_vel_xy.csv                model_1000.pt   model_800.pt
 Episode_force.csv                           Episode_rew_style_feet_distance.csv             Episode_rew_target_target_base_height.csv        model_100.pt    model_900.pt
 Episode_rew_regu_action_rate.csv            Episode_rew_style_ground_parallel.csv           Episode_rew_target_target_orientation.csv        model_1100.pt  'Perf_collection time.csv'
 Episode_rew_regu.csv                        Episode_rew_style_hip_roll_deviation.csv        Episode_rew_task.csv                             model_1200.pt   Perf_learning_time.csv
 Episode_rew_regu_dof_acc.csv                Episode_rew_style_hip_yaw_deviation.csv         Episode_rew_task_head_height.csv                 model_1300.pt   Perf_total_fps.csv
 Episode_rew_regu_dof_pos_limits.csv         Episode_rew_style_knee_deviation.csv            Episode_rew_task_orientation.csv                 model_1400.pt   Policy_mean_noise_std.csv
 Episode_rew_regu_dof_vel.csv                Episode_rew_style_left_foot_displacement.csv    events.out.tfevents.1753631998.unitree.29062.0   model_200.pt    Train_mean_episode_length.csv
 Episode_rew_regu_dof_vel_limits.csv         Episode_rew_style_right_foot_displacement.csv   log_export.py                                    model_300.pt    Train_mean_episode_length_time.csv
 Episode_rew_regu_joint_power.csv            Episode_rew_style_style_ang_vel_xy.csv          Loss_learning_rate.csv                           model_400.pt    Train_mean_reward.csv
 Episode_rew_regu_joint_tracking_error.csv   Episode_rew_target_ang_vel_xy.csv               Loss_surrogate.csv                               model_500.pt    Train_mean_reward_time.csv
 Episode_rew_regu_smoothness.csv             Episode_rew_target.csv                          Loss_value_function.csv                          model_600.pt
```
