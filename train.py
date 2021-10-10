import gym
import numpy as np
import argparse
import frozen_lake
from agents.monte_carlo import MonteCarlo
from agents.TD import QLearning, SARSA

AGENTS = {'monte_carlo': MonteCarlo, 'q_learning': QLearning, 'sarsa': SARSA}
ENVS = {'frozen_lake': 'frozen_lake:frozen-lake-v0', 'frozen_lake_large': 'frozen_lake:frozen-lake-large-v0'}

def parse_args():
    parser = argparse.ArgumentParser()
    # Agent parameters
    parser.add_argument('--agent', default='monte_carlo', type=str, help='RL agent: monte_carlo, q_learning or sarsa')
    parser.add_argument('--epsilon', default=0.1, type=float, help='epsilon')
    parser.add_argument('--epsilon_decay', default=0, type=float, help='epsilon decay rate')
    parser.add_argument('--gamma', default=0.95, type=float, help='discount rate')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate, for q_learning and sarsa')
    # Environment parameters
    parser.add_argument('--env', default='frozen_lake', type=str, help='environment: frozen_lake or frozen_lake_large')
    parser.add_argument('--hide_video', default=False, action='store_true', help='hide frozen lake simulation window')
    # train
    parser.add_argument('--train_log', default=100, type=int, help='Train log frequency')
    parser.add_argument('--train_episodes', default=10000, type=int, help='Number of training episodes')
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)

    args = parser.parse_args()
    return args

def make_agent(args, env):
    if args.agent == 'monte_carlo':
        return AGENTS[args.agent]\
            (env.observation_space,
             env.action_space,
             args.train_log,
             args.train_episodes,
             args.eval_freq,
             args.epsilon,
             args.epsilon_decay,
             args.gamma)
    elif args.agent == 'q_learning' or args.agent == 'sarsa':
        return AGENTS[args.agent]\
            (env.observation_space,
             env.action_space,
             args.train_log,
             args.train_episodes,
             args.eval_freq,
             args.lr,
             args.epsilon,
             args.epsilon_decay,
             args.gamma)
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    env = gym.make(ENVS[args.env], kwargs={"video":args.hide_video})
    agent = make_agent(args, env)

    agent.train(env)

if __name__ == '__main__':

    main()