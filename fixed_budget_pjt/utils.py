import os
import random
from pathlib import Path

import numpy as np

import torch
from torch._six import inf

from fixed_budget_pjt.config import ENV
from fixed_budget_pjt.model import PolicyNet


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_kth_max(x: np.ndarray, k):
    """ k: 1-indexed """
    assert len(x.shape) == 2
    return np.partition(x, -k, axis=1)[:, -k]


def load_model(model_path: Path):
    print(f'==> Trying to load model from checkpoint {model_path}..')
    assert model_path.exists(), f'cannot find {model_path}'
    # model = torch.jit.load(str(model_path))  # not available in parallel computation
    # model = torch.jit.trace(model_path)
    model = PolicyNet(n_arms=3)  # todo  # only n_arms = 3 for now
    model.load_state_dict(torch.load(model_path))
    # # model = torch.load(model_path)
    return model


def load_optimizer(opt_path: Path, optimizer):
    print(f'==> Trying to load optimizer from {opt_path}..')
    assert opt_path.exists(), f'cannot find {opt_path}'
    optimizer.load_state_dict(torch.load(opt_path))
    return optimizer


def H_ndarray(means: np.ndarray, eps=1e-10):
    """ One of the most standard complexity in the fixed-confidence setting

    Args:
        means: one-dimensional ndarray

    Returns:
        math: sum_{i \neq i^*} 1 / max{mu_{i^*} - mu_i}, eps} ** 2
    """
    max_mean = means.max()
    return np.sum(1./np.maximum((max_mean - means), eps) ** 2, where=(means != max_mean))


def H_from_gap(gaps: np.ndarray, n_arms, eps=1e-10):
    """ REMARK. Added tmp for complexity pre-training # todo: ongoing
    Assuming that gaps is a shape of (n_arms - 1), NOT (n_arms,) """
    assert gaps.shape[0] == n_arms - 1
    return np.sum(1./np.maximum(gaps, eps) ** 2)
    # max_mean = means.max()
    # return np.sum(1./np.maximum((max_mean - means), eps) ** 2, where=(means != max_mean))


@torch.jit.script
def H_tensor(means: torch.FloatTensor, eps=torch.tensor(1e-10)):
    """ Tensor implementation of H_ndarray
    """
    max_mean, best_arm = torch.max(means, 0)
    return torch.sum(1./torch.maximum((max_mean - means[means != max_mean]), eps) ** 2)


# @torch.jit.script
def H_batch_tensor(means: torch.FloatTensor, eps=torch.tensor(1e-10)):
    """ Batch function of H_tensor, assuming unique optimal arm in means

    Args:
        means: two-dimensional tensor, (batch, n_arms)

    Returns:
        tensor with shape (batch, H)
    """
    n_data, n_arms = means.shape
    max_mean, best_arm = torch.max(means, dim=1)
    max_mean_broadcast = torch.broadcast_to(max_mean, (n_arms, n_data)).T
    diff = torch.where(max_mean_broadcast - means == 0, torch.tensor(inf).to(ENV.DEVICE), max_mean_broadcast - means)

    return torch.sum(1./torch.maximum(diff, eps)**2, dim=1)


def kl_func(p: torch.FloatTensor, q: torch.FloatTensor, arm_dist):
    if arm_dist == 'Ber':
        return kl_bernoulli_bernoulli(p, q)
    elif arm_dist == 'Normal':
        return kl_normal_normal(p, q)
    else:
        raise NotImplementedError


def kl_bernoulli_bernoulli(p: torch.FloatTensor, q: torch.FloatTensor):
    """ fixed version of https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py#L181
    """
    t1 = p * (p / q).log()
    t1[q == 0] = inf
    t1[p == 0] = 0
    t2 = (1 - p) * ((1 - p) / (1 - q)).log()
    t2[q == 1] = inf
    t2[p == 1] = 0
    return t1 + t2


@torch.jit.script
def kl_normal_normal(p: torch.FloatTensor, q: torch.FloatTensor):
    """ Only for Gaussian with a unit variance
    """
    return 0.5 * (p - q).pow(2)


if __name__ == '__main__':
    '''
    all unit tests are in test/test_utils.py
    '''
    print(H_batch_tensor.__name__)
    exit(0)

    print('-' * 100)
    print('==> Unit Testing H..')

    means_list = [
        [0.5, 0.4, 0.2],
        [0.1, 0.4, 0.2],
        [0.1, 0.4, 0.2, 0.39999],
    ]

    eps_list = [
        1e-5,
        1e-5,
        1e-2,
    ]

    # care int part
    expected_list = [
        1./(0.1)**2 + 1./(0.3)**2,
        1./(0.3)**2 + 1./(0.2)**2,
        1./(0.3)**2 + 1./(0.2)**2 + 1./(1e-2)**2,
    ]

    for idx in range(len(means_list)):
        means = means_list[idx]
        eps = eps_list[idx]
        expected = expected_list[idx]

        means_ndarray = np.array(means)
        means_tensor = torch.FloatTensor(means)

        H_a = H_ndarray(means_ndarray, eps=eps)
        H_t = H_tensor(means_tensor, eps=torch.tensor(eps))

        print(f'test case {idx}: '
              f'means {means}, '
              f'H_ndarray {H_a}, '
              f'H_tensor {H_t}, '
              f'expected {expected}')

        assert np.isclose(H_a, expected, atol=1e-1)
        assert torch.isclose(H_t, torch.tensor(expected), atol=1e-1)



