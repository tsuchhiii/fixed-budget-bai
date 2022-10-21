import math
import numpy as np


def gen_pe_setting(n_arms, idx, ascending=False) -> np.ndarray:
    """ [Bubeck 2009]
    and easy version of the following gen_pe_hard_setting """
    assert n_arms >= 3

    if n_arms == 3:
        if idx == 1:
            reward_means = [0.5, 0.45, 0.3]   # hard for uniform (added in 2022.3.13)
        elif idx == 2:
            reward_means = [0.5, 0.45, 0.05]  # hard for SR
        elif idx == 3:
            reward_means = [0.5, 0.45, 0.45]  # somewhat good for SR
        elif idx == 4:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return np.array(reward_means)

