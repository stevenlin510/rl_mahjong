# Project: Taiwanese Mahjong
This project aims to modify the existing Mahjong environment in RLCard to implement Taiwanese Mahjong rules. The adaptation will enable reinforcement learning agents to learn and play Mahjong according to traditional Taiwanese scoring systems and gameplay mechanics.

# Installation

Install Pytorch
```bash
pip install torch
```
Install this repo
```bash
pip install -e . 
```
# Usage
Train a single DQN Agent
```bash
python run_dqn.py --log_dir 'experiments/mahjong_ppo_result/'
```

Train multiple DQN Agents
```bash
python run_dqn_multi.py --log_dir 'experiments/mahjong_ppo_result/'
```

Play Demo with trained agent
```bash
python demo.py
```
# Acknowledgements

[rl_card](https://github.com/datamllab/rlcard)
[mahjong_rl](https://github.com/sharedcare/mahjong_rl)