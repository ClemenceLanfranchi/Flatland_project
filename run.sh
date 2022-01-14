#!/bin/bash

# uncomment the desired line
# python baselines/run.py --checkpoint baselines/checkpoints/ppo-default-smallv0-9800.pth --tree_obs_type default
python baselines/run.py --checkpoint baselines/checkpoints/ppo-default-test0-11900.pth --tree_obs_type default
# python baselines/run.py --checkpoint baselines/checkpoints/ppo-deadlock-smallv0-5200.pth --tree_obs_type deadlock
# python baselines/run.py --checkpoint baselines/checkpoints/ppo-deadlock-test0-3000.pth --tree_obs_type deadlock
# python baselines/run.py --checkpoint baselines/checkpoints/ppo-judge-smallv0-800.pth --tree_obs_type judge
# python baselines/run.py --checkpoint baselines/checkpoints/ppo-judge-test0-900.pth --tree_obs_type judge
