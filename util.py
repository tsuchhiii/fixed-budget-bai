import os
import numpy as np

'''
for pe-mab
'''

def get_pemab_save_basename(algo_name, setting, arm_dist, n_arms, noise_var):
    return f'{algo_name}_{setting}_{arm_dist}_{n_arms}_{noise_var}'


def get_pemab_save_path(algo_name, setting, arm_dist, n_arms, noise_var):
    model = 'pe-mab'
    basename = get_pemab_save_basename(algo_name, setting, arm_dist, n_arms, noise_var)
    return f'results/{model}/{basename}.csv'


def get_pemab_fig_basename(setting, arm_dist, n_arms, noise_var):
    """ for pe mab """
    return f'na_{n_arms}_setting_{setting}_dist_{arm_dist}_nv_{int(noise_var)}'




def get_save_basename(algo_name, setting, arm_dist, n_arms, dim, model, noise_var, horizon):
    """ for linear bandits """
    return f'{algo_name}_{setting}_{arm_dist}_{n_arms}_{dim}_{model}_{noise_var}_{horizon}'


def get_save_path(algorithm, setting, arm_dist, n_arms, dim, model, noise_var, horizon, log_save=False):
    basename = get_save_basename(algorithm, setting, arm_dist, n_arms, dim, model, noise_var, horizon)
    if log_save:
        return 'results/{}/{}=log.csv'.format(model, basename)
    else:
        return 'results/{}/{}.csv'.format(model, basename)


def get_fig_basename(setting, arm_dist, n_arms, dim, model, noise_var):
    """ for linear bandits """
    # return '{}_{}_{}_{}_{}'.format(setting, n_arms, dim, model, noise_var)
    return f's_{setting}_na_{n_arms}_d_{dim}_dist_{arm_dist}_m_{model}_nv_{noise_var}'


def split_save_path(save_path):
    basename_wo_ext = os.path.splitext(os.path.basename(save_path))[0]
    return basename_wo_ext.split('_')


def compute_kl_bernoulli(p_val, q_val):
    """
    compute KL divergence between
    p log(p/q) + (1-p) log((1-p)/(1-q)) (p is p_val, q is q_val)
    specifically:
    (i) p = 0 => log(1/(1-q))
    (ii) p = 1 => log(1/q)


    Args:
        p, q in [0,1]

    Remark:
        - 0 * log(0/qi) is 0 for any qi.
        - p << q is required
            (p should be abs. cont. w.r.t. q,
            since they are qit and Sip respectively)

    Returns:

    """
    p = np.array([p_val, 1.-p_val])
    q = np.array([q_val, 1.-q_val])

    # todo: refactor
    assert np.all(p >= -1e-5) and np.all(p <= 1. + 1e-5), 'p_val is {}'.format(p_val)
    assert np.all(q >= -1e-5) and np.all(q <= 1. + 1e-5)
    assert np.abs(np.sum(p) - 1) < 1e-5
    assert np.abs(np.sum(q) - 1) < 1e-5
    assert p.shape == q.shape

    ret = 0
    # todo: refactor
    for i in range(p.shape[0]):
        if p[i] == 0:
            ret += 0
        elif q[i] == 0:
            INF = 99999
            ret += INF
            # raise Exception('The condition p << q is not satisfied.')
        else:
            ret += p[i] * np.log(p[i] / q[i])

    return ret


def compute_kl_gaussian(mu1, mu2, sigma):
    return (mu1 - mu2)**2 / (2 * sigma**2)


def sample_from_categorical_dist(p):
    """ sample from categorical distribution, as there is not such function in numpy,,
    just for 1d array in probability simplex. """
    assert len(p.shape) == 1
    assert np.isclose(np.sum(p), 1.), f'p is {p}, np.sum is {np.sum(p)}'

    return np.random.choice(len(p), 1, p=p)[0]



