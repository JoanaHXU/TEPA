import numpy as np
import torch
import gym
import argparse
import os
import sys
import time
import copy

from collections import defaultdict
from itertools import count

# TensorBoard
import tensorboardX
import datetime

# Configuration
from yacs.config import CfgNode as CN
yaml_name='config/config_default.yaml'
fcfg = open(yaml_name)
config = CN.load_cfg(fcfg)
config.freeze()

SEQ_LEN = config.AE.SEQ_LEN
EMBEDDING_SIZE = config.AE.EMBEDDING_SIZE
MEMORY_SIZE = config.AE.MEMORY_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attack package
from utils import utils_buf, utils_op, utils_attack, utils_log
from attack.DDPG import DDPG
from envs.target_def import TARGET

# Environment object
from envs.env3D_4x4 import GridWorld_3D_env
env = GridWorld_3D_env()
INIT_T = env.T.copy()

# Victim object
from victim.system import System


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_dir", default=None)               # TensorBoard folder
    
    parser.add_argument("--policy", default="DDPG")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_episodes", default=100, type=int)  # Time steps initial random policy is used
    parser.add_argument("--max_timesteps", default=12, type=int)   # Max time steps to run environment
    parser.add_argument("--max_episodes_num", default=1000, type=int)   # Max episodes to run environment
    parser.add_argument("--eval_freq_episode", default=50, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=512, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    
    parser.add_argument("--victim_n_episodes", default=60, type=int)  # number of episodes for victim's updated in poisoned Env
    parser.add_argument("--ae_n_epochs", default=10, type=int)         # number of training epoch
    
    args = parser.parse_args()

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ''' ..... Tensorboard Settings ..... '''

    # Set run dir
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"results_{date}"
    
    model_name = args.model_dir or default_model_name
    model_dir = utils_log.get_model_dir(model_name)

    # Load loggers and Tensorboard writer
    txt_logger = utils_log.get_txt_logger(model_dir)
    csv_file, csv_logger = utils_log.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)
    
    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))
    
    ''' ..... Attack Network ..... '''
    # Input / Output size
    state_dim = EMBEDDING_SIZE + env.nS
    action_dim = env.Attack_ActionSpace.shape[0]
    max_action = float(env.Attack_ActionSpace.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize Policy
    Policy = DDPG(**kwargs)
    Buffer = utils_buf.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    
    txt_logger.info("Log: Attack Strategy Initialize")
    
    ''' ..... Victim ..... '''
    
    victim_args = {
        "env": env, 
        "MEMORY_SIZE": MEMORY_SIZE,
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    ae_enc_in_size = SEQ_LEN*2
    ae_enc_out_size = EMBEDDING_SIZE
    ae_dec_in_size = 1+EMBEDDING_SIZE
    ae_dec_out_size = action_dim
    
    ae_args = {
        "enc_in_size": ae_enc_in_size, 
        "enc_out_size": ae_enc_out_size, 
        "dec_in_size": ae_dec_in_size, 
        "dec_out_size": ae_dec_out_size, 
        "n_epochs": 50, 
        "lr": 0.001, 
    }
    
    system = System(victim_args, ae_args)
        
        
    ''' ..... Training ..... '''

    for i_episode in range(args.max_episodes_num):
        txt_logger.info(f"\n--------- Episode: {i_episode} ----------")
        tic_episode = time.time()
        cumulative_cost = 0

        # reset victim's env and Q
        env.reset_altitude()
        system.victim.reset()

        # Initialize the attacker's state
        victim_info = np.zeros((1,EMBEDDING_SIZE))
        victim_tensor = torch.from_numpy(victim_info)
        victim_tensor_4d = victim_tensor.unsqueeze(0).unsqueeze(0)

        env_info = env.altitude.copy()
        env_tensor = torch.from_numpy(env_info)
        env_tensor = env_tensor.view(1, env.nS)
        env_tensor_4d = env_tensor.unsqueeze(0).unsqueeze(0)

        x = torch.cat((victim_tensor_4d, env_tensor_4d), 3)
        
        for t in range(args.max_timesteps):

            # Select attack_action
            if i_episode < args.start_episodes:
                u = env.Attack_ActionSpace.sample()
            else:
                u = (
                    Policy.select_action(np.array(x))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Step: implement attack_action
            env.Attack_Env(u)

            # Step: victim updates = get next_x
            system.train(args.victim_n_episodes, args.ae_n_epochs)
            ### ... updated victim.Q
            next_victim_info = system.ae.Embedding(system.victim.MEM)
            next_victim_tensor = torch.from_numpy(next_victim_info[-1]).unsqueeze(0)
            next_victim_tensor_4d = next_victim_tensor.unsqueeze(0).unsqueeze(0)
            ### ... updated env altitude
            next_env_info = system.victim.env.altitude.copy()
            next_env_tensor = torch.from_numpy(next_env_info)
            next_env_tensor = next_env_tensor.view(1, env.nS)
            next_env_tensor_4d = next_env_tensor.unsqueeze(0).unsqueeze(0)
            ### ... next_state
            next_x = torch.cat((next_victim_tensor_4d, next_env_tensor_4d), 3)
            
            # Step: cost
            cost = (-1)*utils_attack.Attack_Cost_Compute(env, INIT_T, system.victim.Q, TARGET)
            done, rate = utils_attack.Attack_Done_Identify(env, TARGET, system.victim.Q)
            ### log
            cumulative_cost += cost
            txt_logger.info(f"episode = {i_episode+1} | t = {t+1} | cost: {cost:.4f} | done: {done} | success: {rate:.3f}")

            # Replay buffer
            Buffer.add(x.view(state_dim), u, next_x.view(state_dim), cost, done)

            # Update state 
            x = copy.deepcopy(next_x)

            # Attack_Policy Update
            if i_episode >= args.start_episodes:
                Policy.train(Buffer, args.batch_size)
                
            if done: 
                # training time of episode
                toc_episode = time.time()
                time_episode = toc_episode - tic_episode
                # txt_logger
                txt_logger.info(f"...Done... Episode={i_episode+1} | Timestep={t+1} | Running_Time={time_episode:.3f} | Cost={cumulative_cost:.3f}")
                # csv log
                data = [t+1, -cumulative_cost]
                csv_logger.writerow(data)
                csv_file.flush()
                # tensorboard log
                tb_writer.add_scalar('Timesteps/train', t+1, i_episode+1)
                tb_writer.add_scalar('Cost/train', -cumulative_cost, i_episode+1)

                break

            if t+1==args.max_timesteps:
                # training time of episode
                toc_episode = time.time()
                time_episode = toc_episode - tic_episode
                # txt_logger
                txt_logger.info(f"...Done... Episode={i_episode+1} | Timestep={t+1} | Running_Time={time_episode:.3f} | Cost={cumulative_cost:.3f}")
                # csv log
                data = [t+1, -cumulative_cost]
                csv_logger.writerow(data)
                csv_file.flush()
                # tensorboard log
                tb_writer.add_scalar('Timesteps/train', t+1, i_episode+1)
                tb_writer.add_scalar('Cost/train', -cumulative_cost, i_episode+1)

                break
                
        ''' save Attack_Policy '''
        if (i_episode + 1) % args.eval_freq_episode == 0:
            model_no = str(i_episode + 1)
            Policy.save(f"./{model_dir}/{model_no}")
            system.ae.save(f"./{model_dir}/{model_no}")
