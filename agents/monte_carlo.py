import numpy as np
import matplotlib.pyplot as plt


def e_soft_policy(Q, s, eps=0.1):
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[2])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    possible_actions = []
    best = np.argmax(Q[s[0]][s[1]])
    # Select a random best action if the best action is tied
    for a in range(len(Q[s[0]][s[1]])):
        if Q[s[0]][s[1]][a] == Q[s[0]][s[1]][best]:
            possible_actions.append(a)
    return np.random.choice(possible_actions)


def softmax(Q, s):
    possible_actions = [i for i in range(Q.shape[2])]
    values = Q[s[0]][s[1]]
    e_x = np.exp(values) / sum(np.exp(values))
    return np.random.choice(possible_actions, p=e_x)


def evaluate(env, agent, i):
    total_reward = 0
    state = env.reset()

    done = False
    step = 0
    while not done:
        # select a greedy action
        next_state, rew, done, _ = env.step(greedy(agent.Q, state))

        state = next_state
        total_reward += rew

        if not env.hide_video:
            env.render(episode=i)

        # done if the agent enters an infinite cycle
        if step > 1000:
            done = True

        if done:
            state = env.reset()
            if step > 1000:
                total_reward = -1

        step += 1

    return total_reward


class MonteCarlo():

    def __init__(self, obs_shape, action_shape, train_log=100, num_episodes=10000, eval_freq=1000, eps=0.1, eps_decay=0,
                 gamma=0.95):

        # Initialize the Q matrix
        # Q: matrix r*c*A for row of grid, column of grid, and action taken at that state
        r, c, a = obs_shape[0].n, obs_shape[1].n, action_shape.n
        self.Q = np.zeros((r, c, a)) # row, column, action

        self.train_log = train_log
        self.num_episodes = num_episodes
        self.eval_freq = eval_freq
        self.eps = eps
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.episode = []

        print("Training Monte Carlo agent without ES")

    def train(self, env):

        rewards_total = []
        eval_rewards_total = []
        for i in range(self.num_episodes):
            state = env.reset()
            self.episode = []
            self.returns = [[[[] for i in range(env.action_space.n)] for j in range(env.observation_space[1].n)] for k in
                            range(env.observation_space[0].n)]

            step = 0
            done = False
            # adaptive epsilon in e-greedy policy
            if self.eps > 0.01:
                self.eps -= self.eps_decay
            while not done:
                step += 1

                # observe (s', reward) from action
                action = e_soft_policy(self.Q, state, self.eps)
                next_state, reward, done, info = env.step(action)

                # append (s, a, r) to episode
                self.episode.append({'state':state, 'action':action, 'reward':reward})

                state = next_state
                if not env.hide_video:
                    env.render(episode=i)

                # done if the agent enters an infinite cycle
                # if step > 1000:
                #     done = True

                if done and i % self.train_log == 0:
                    rewards_total.append(reward)

            G = 0
            t = step-1
            while t >= 0:
                G = self.gamma*G + self.episode[t]['reward']

                # update state-action values based on every-visit MC prediction
                s = self.episode[t]['state']
                a = self.episode[t]['action']
                self.returns[s[0]][s[1]][a].append(G)
                self.Q[s[0]][s[1]][a] = sum(self.returns[s[0]][s[1]][a])/len(self.returns[s[0]][s[1]][a])
                t -= 1

            if i % self.eval_freq == 0:
                eval_score = evaluate(env, self, i)
                print("Episode {} average reward: {}".format(i, eval_score))
                eval_rewards_total.append(eval_score)

        x = np.arange(0, self.num_episodes, self.train_log)
        x2 = np.arange(0, self.num_episodes, self.eval_freq)
        plt.plot(x, rewards_total, label="training reward")
        plt.plot(x2, eval_rewards_total, label="eval reward")
        plt.title("Monte Carlo agent in frozen lake: evaluation reward")
        plt.legend(loc='lower right')
        plt.show()

        print("State-action matrix Q:\n", self.Q)