#!/bin/bash

python main.py --model_dir=Q_0817_v2 \
        --max_timesteps=15 \
        --max_episodes_num=1000 \
        --start_episodes=100 \
        --eval_freq_episode=50 \
        --batch_size=256 \
        --discount=0.95 \
        --victim_type=Q \
        --victim_n_episodes=80 \