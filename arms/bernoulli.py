import random
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List


class Arm(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, seed):
        pass

    @abstractmethod
    def draw(self, time):
        pass


class ArmSet(metaclass=ABCMeta):
    def __init__(self):
        self.arms: List[Arm] = None

        self.expected_reward_list = None  # abstract attribute
        self.expected_rewards_matrix = None
        self.max_reward = None
        self.best_arm = None  # abstract attribute

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def initialize(self, seed):
        self.set_seed(seed)
        for arm in self.arms:
            arm.initialize(seed)

    def draw(self, arm_idx, time):
        """ Remark. draw method for class a set of arms """
        return self.arms[arm_idx].draw(time=time)


class BernoulliArm(Arm):
    def __init__(self, p, horizon):
        super(BernoulliArm, self).__init__()
        self.p = p
        self.horizon = horizon
        self.observed_rewards = None

    def initialize(self, seed):
        self.observed_rewards = np.random.binomial(n=1, p=self.p, size=self.horizon)

    def draw(self, time):
        # todo: refactor
        return self.observed_rewards[time]

'''
class BernoulliArm(Arm):
    def __init__(self, p):
        super(BernoulliArm, self).__init__()
        self.p = p

    def draw(self):
        """ Remark. draw method for class a single arm"""
        # todo: refactor using numpy function?
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0
'''


class BernoulliArmSet(ArmSet):
    """ refer LinearArm class and made this. """
    def __init__(self, p_list: np.ndarray, horizon: int):
        super(BernoulliArmSet, self).__init__()

        self.expected_reward_list = p_list
        self.arms = list(
            map(lambda p: BernoulliArm(p=p, horizon=horizon), self.expected_reward_list))

        # for using with adversarial environments
        self.expected_rewards_matrix = np.broadcast_to(p_list, (horizon, len(p_list))).T  # shape == (n_arms, horizon)

        self.max_reward = np.max(self.expected_reward_list)
        self.best_arm = np.argmax(self.expected_reward_list)

    # def initialize(self, seed):
    #     super(BernoulliArmSet, self).initialize(seed)
        # p_list = self.expected_reward_list
        # horizon = self.horizon
