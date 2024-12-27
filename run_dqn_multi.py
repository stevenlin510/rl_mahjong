import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    MultiLogger,
    plot_curves,
    plot_loss
)

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        if args.load_checkpoint_path != "":
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64,64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        )

    env.set_agents(agents)

    # Start training
    with MultiLogger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for i in range(len(agents)):
                if isinstance(agents[i], DQNAgent):  # Only train DQN agents
                    for ts in trajectories[i]:
                        agents[i].feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                for i in range(len(agents)):
                    reward = tournament(env, args.num_eval_games)[i]  # Get reward for each agent
                    logger.log_performance(episode, i, reward)  # Log performance with agent ID

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    for i in range(len(agents)):
        plot_loss(agents[i].training_losses, os.path.join(args.log_dir, f'agent_{i}_loss.png'))


    plot_curves(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='mahjong',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/mahjong/',
    )
    
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=-1)

    args = parser.parse_args()

    train(args)

