''' A toy example of playing against rule-based bot on UNO
'''
import os
import torch
import numpy as np
import rlcard
from rlcard import models
from human_agent import HumanAgent, _print_action
from rlcard.agents import DQNAgent

# Make environment
env = rlcard.make('mahjong')
human_agent = HumanAgent(env.num_actions)
device = torch.device("cpu")

model_path = os.path.join('./experiments/1226_final_mg/', 'checkpoint_dqn.pt')
agent = DQNAgent.from_checkpoint(checkpoint=torch.load(model_path, weights_only=False))
agents = [agent]
for _ in range(1, env.num_players):
        agents.append(agent)
env.set_agents(agents)

print(">> Mahjong ")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action(pair[1])
        print('')

    print('===============     Result     ===============')
    winner = np.argmax(payoffs) 
    print(f"Reward: {payoffs}")
    if payoffs.any() == 0:
        print(f'No player win the game')
    else:
        print(f'Player {winner} win!')
        
    print('')
    input("Press any key to continue...")