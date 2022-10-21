import random
import numpy as np

from algorithms.base.base import MABAlgorithm


class Uniform(MABAlgorithm):
    def __init__(self, n_arms, sigma, seed=0):
        super(Uniform, self).__init__(n_arms=n_arms, sigma=sigma, seed=seed)

    def initialize(self, seed):
        super(Uniform, self).initialize(seed)

    def select_arm(self):
        return (self.step - 1) % self.n_arms

    def update(self, chosen_arm, reward):
        super(Uniform, self).update(chosen_arm_idx=chosen_arm, reward=reward)

    def predict_best_arm(self):
        return np.argmax(self.values)


class SuccessiveRejects(Uniform):
    def __init__(self, n_arms, sigma, horizon, seed=0):
        """ (Audibert, Bubeck & Munos, COLT2010) Best Arm Identification in Multi-Armed Bandits
        """
        super(SuccessiveRejects, self).__init__(n_arms=n_arms, sigma=sigma, seed=seed)
        log_over_n_arms = self.log_over()

        # compute n_k for k = 1,..., K-1 (no n_0 here)
        self.horizon = horizon
        self.n_list = [int(np.ceil((1./log_over_n_arms) * (horizon - n_arms) / (n_arms + 1 - k)))
                       for k in range(1, n_arms)]
        self.remained_arm_set = None
        self.n_list_pointer = None

    def log_over(self):
        """ Math: 1/2 + sum_{i=2}^n_arms 1/i """
        return 1./2. + sum([1./i for i in range(2, self.n_arms + 1)])

    def initialize(self, seed):
        super(SuccessiveRejects, self).initialize(seed)
        self.remained_arm_set = set(range(self.n_arms))
        self.n_list_pointer = 0

    def find_worst_arm_from_remaining(self):
        emp_worst_arm, emp_worst_reward = None, 99999
        for arm in self.remained_arm_set:
            if self.values[arm] < emp_worst_reward:
                emp_worst_arm = arm
                emp_worst_reward = self.values[arm]
        return emp_worst_arm

    def find_best_arm_from_remaining(self):
        emp_best_arm, emp_best_reward = None, -99999
        for arm in self.remained_arm_set:
            if self.values[arm] > emp_best_reward:
                emp_best_arm = arm
                emp_best_reward = self.values[arm]
        return emp_best_arm

    def select_arm(self):
        if len(self.remained_arm_set) == 1:
            # all done; for just fixing difference of a few steps
            return list(self.remained_arm_set)[0]

        #  todo: more efficient implementation
        counts_array = np.array(self.counts)
        remained_arm_list = list(self.remained_arm_set)
        tmp_min_idx_in_remained = counts_array[remained_arm_list].argmin()
        min_pull_idx_in_remained = remained_arm_list[tmp_min_idx_in_remained]
        min_pull_in_remained = self.counts[min_pull_idx_in_remained]
        if min_pull_in_remained < self.n_list[self.n_list_pointer]:
            return min_pull_idx_in_remained

        # now, all arms in remained_arm_set are pulled at least self.n_list[self.n_list_pointer] times
        self.n_list_pointer += 1

        # remove the lowest arm from remained arm set
        emp_worst_arm = self.find_worst_arm_from_remaining()
        self.remained_arm_set.remove(emp_worst_arm)

        return random.choice(tuple(self.remained_arm_set))

    def predict_best_arm(self):
        if self.step == self.horizon:
            assert len(self.remained_arm_set) == 1
            return list(self.remained_arm_set)[0]
        else:
            return self.find_best_arm_from_remaining()

