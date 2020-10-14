import numpy as np


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self, bandit, policy, prior=0, gamma=None):
        self.policy = policy
        self.k = bandit.k
        self.prior = prior
        self.gamma = gamma
        self._value_estimates = prior*np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g*(reward - q)
        self.t += 1

    @property
    def value_estimates(self):
        return self._value_estimates


class GradientAgent(Agent):
    """
    The Gradient Agent learns the relative difference between actions instead of
    determining estimates of reward values. It effectively learns a preference
    for one action over another.
    """
    def __init__(self, bandit, policy, prior=0, alpha=0.1, baseline=True):
        super(GradientAgent, self).__init__(bandit, policy, prior)
        self.alpha = alpha
        self.baseline = baseline
        self.average_reward = 0

    def __str__(self):
        return 'g/\u03B1={}, bl={}'.format(self.alpha, self.baseline)

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.baseline:
            diff = reward - self.average_reward
            self.average_reward += 1/np.sum(self.action_attempts) * diff

        pi = np.exp(self.value_estimates) / np.sum(np.exp(self.value_estimates))

        ht = self.value_estimates[self.last_action]
        ht += self.alpha*(reward - self.average_reward)*(1-pi[self.last_action])
        self._value_estimates -= self.alpha*(reward - self.average_reward)*pi
        self._value_estimates[self.last_action] = ht
        self.t += 1

    def reset(self):
        super(GradientAgent, self).reset()
        self.average_reward = 0