#!/bin/bash

python main.py --model_dir=MC_0817 \
        --max_timesteps=15 \
        --max_episodes_num=1000 \
        --start_episodes=100 \
        --eval_freq_episode=50 \
        --batch_size=256 \
        --discount=0.99 \
        --victim_type=MC \
        --victim_n_episodes=80 \