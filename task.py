from pathlib import Path
import time
from legged_gym.envs import *
from legged_gym.utils import  get_args,  task_registry
from terrain_base.config import terrain_config

import torch
import faulthandler

def evaluate(file_path: str, user_token: str, robot_type: str) -> None:
    file_path = Path(file_path)           # ← 已是绝对路径

    faulthandler.enable()
    args = get_args()
    args.exptid = 'test'

    # robot_type可能是这里
    args.task = 'h1_2_fix'

    exptid = args.exptid

    # 这里我们用test路径下的模型
    log_pth = file_path.parent
    # log_pth = "./legged_gym/logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1000
    env_cfg.commands.resampling_time = 60
    env_cfg.rewards.is_play = False

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = 1

    env_cfg.terrain.height = [0.01, 0.02]

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args,
                                                                   train_cfg=train_cfg, return_log_dir=True)
    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 19, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(10 * int(env.max_episode_length)):

        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                obs_student[:, 6:8] = 0
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 6:8] = 1.5 * yaw

        else:
            depth_latent = None

        if hasattr(ppo_runner.alg, "depth_actor"):
            actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)

        obs, _, rews, dones, infos = env.step(actions.detach())

        id = env.lookat_id

    print(f"[Task] done: {file_path} by {user_token}")