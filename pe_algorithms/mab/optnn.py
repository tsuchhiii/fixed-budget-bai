import numpy as np
import torch

from pe_algorithms.mab.uniform import Uniform

from fixed_budget_pjt.utils import load_model


class OptNN(Uniform):
    def __init__(self, model_path, n_arms, sigma, horizon, lazy_policy_update=False, seed=0):
        """ Optimal learning based on trained neural network
        if lazy_polity_update is True, then the algorithm
            forward empirical means and update policy only once per 100 step
        """
        super(OptNN, self).__init__(n_arms=n_arms, sigma=sigma, seed=seed)

        # load NN
        self.model = load_model(model_path)
        self.model.eval()

        self.policy = np.ones(n_arms) / n_arms  # for plotting purpose, this uniform policy is not actually used

        self.lazy_policy_update = lazy_policy_update
        self.lazy_step = 100  # todo

    def initialize(self, seed):
        super(OptNN, self).initialize(seed)
        self.policy = np.ones(self.n_arms) / self.n_arms  # for plotting purpose, this uniform policy is not used

    def select_arm(self):
        # pull each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # pull arm based on trained policy

        # emp_means = torch.FloatTensor(self.values).to(ENV.DEVICE)[None, :]
        # policy = self.model(emp_means).to('cpu').detach().numpy().copy()

        if not self.lazy_policy_update or self.step % self.lazy_step == 0:
            emp_means = torch.FloatTensor(self.values)[None, :]
            policy = self.model(emp_means).detach().numpy()

            self.policy = policy  # for plotting purpose
        else:
            policy = self.policy

        return np.argmax(self.step * policy - np.array(self.counts))


