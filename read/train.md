python train.py --task zq_ground
│
├── get_args()  → args.task = "zq_ground"
│
├── task_registry.make_env("zq_ground")
│   └── 返回 LeggedRobot_Zq(env_cfg=ZqCfgGround())
│
├── task_registry.make_alg_runner("zq_ground")
│   └── 返回 OnPolicyRunner(train_cfg=ZqCfgPPOGround())
│
└── ppo_runner.learn()
└── 正式开始训练

python train.py --task zq_ground --run_name test_zq_ground
│
├── get_args()  → args = Namespace(task="zq_ground", run_name="test_zq_ground")
│
├── train(args)
│   ├── task_registry.make_env("zq_ground", args)
│   │   ├── env_cfg = ZqCfgGround()
│   │   ├── env_cfg ← update_cfg_from_args(env_cfg, args)
│   │   └── env = LeggedRobot_Zq(env_cfg)
│   │
│   ├── task_registry.make_alg_runner("zq_ground", args)
│   │   ├── train_cfg = ZqCfgPPOGround()
│   │   ├── train_cfg ← update_cfg_from_args(train_cfg, args)
│   │   └── ppo_runner = OnPolicyRunner(env, train_cfg)
│   │
│   └── ppo_runner.learn(...)

