import argparse
import random
import numpy as np

from pathlib import Path

# from pe_algorithms.mab.uniform import Uniform, UCB1PE, SuccessiveRejects, UGapEb
from pe_algorithms.mab.uniform import Uniform, SuccessiveRejects
from pe_algorithms.mab.optnn import OptNN
from testing_framework.tests import test_pe_algorithm, test_pe_algorithm_parallel


from config import config_pe_mab

from arms.bernoulli import BernoulliArmSet
# from arms.normal import NormalArmSet

from fixed_budget_pjt.config import PJT_PATH

from mab_setting import gen_pe_setting
from util import get_pemab_save_path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_arms', '-na', required=False, type=int, default=5,
                        help='# of arms')
    parser.add_argument('--algo_name', '-a', required=False, type=str, default='Uniform',
                        choices=config_pe_mab.ALGO_CHOICES,
                        help='xxx')
    parser.add_argument('--arm_dist', '-ad', required=False, type=str, default='Ber',
                        choices=config_pe_mab.ARM_DIST_CHOICES,
                        help='xxx noise distribution (not parameters prior')
    parser.add_argument('--setting', '-s', required=False, type=int, default=None,
                        choices=config_pe_mab.SETTING_CHOICES,
                        help='arm setting')
    parser.add_argument('--horizon', '-ho', required=False, type=int, default=10**4,
                        help='horizon')
    parser.add_argument('--num_sims', '-ns', required=False, type=int, default=100,
                        help='# of simulations')
    parser.add_argument('--do_parallel', '-p', required=False, action='store_true',
                        help='# of simulations')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args)

    num_sims = args.num_sims

    n_arms = args.n_arms
    setting = args.setting
    algo_name = args.algo_name
    arm_dist = args.arm_dist

    test_algorithm = test_pe_algorithm_parallel if args.do_parallel else test_pe_algorithm

    noise_var = 1.0
    sigma_noise = np.sqrt(1.0)  # just for Gaussian arm_dist

    seed = 0  # seed for feat_mat and theta
    np.random.seed(seed)

    if args.setting is not None:
        means = gen_pe_setting(n_arms, setting)
    else:
        if n_arms == 3:
            means = np.array([0.5, 0.48, 0.4])
        elif n_arms == 5:
            means = np.array([0.499, 0.499, 0.499, 0.499, 0.5])
        else:
            raise NotImplementedError

    assert n_arms == means.shape[0]

    # arms = list(map(lambda mu: NormalArm(mu, 0.5), [.25, .55, .60, .62, .63]))

    horizon = args.horizon

    if arm_dist == 'Ber':
        # arms = BernoulliArmSet(p_list=means)
        arms = BernoulliArmSet(p_list=means, horizon=horizon)
    elif arm_dist == 'Normal':
        # arms = NormalArmSet(mu_list=means, sigma=sigma_noise)
        # arms = NormalArmSet(mu_list=means, sigma=sigma_noise, horizon=horizon)
        raise NotImplementedError
    else:
        raise NotImplementedError

    print('best arm (0-indexed) is', np.argmax(means))


    '''
    Run algorithms
    '''

    save_name = get_pemab_save_path(algo_name, setting, arm_dist, n_arms, noise_var)

    # todo: refactor
    if algo_name == 'Uniform':
        random.seed(seed)
        algo = Uniform(n_arms=n_arms, sigma=sigma_noise)
        algo.initialize(n_arms)

        results = test_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    # elif algo_name == 'UCB1PE':
    #     random.seed(seed)
    #     algo = UCB1PE(n_arms=n_arms, sigma=sigma_noise)
    #     algo.initialize(n_arms)

    #     results = test_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    elif algo_name == 'SR':
        random.seed(seed)
        algo = SuccessiveRejects(n_arms=n_arms, sigma=sigma_noise, horizon=horizon)
        algo.initialize(n_arms)

        results = test_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    # elif algo_name == 'UGapEb':
    #     random.seed(seed)
    #     algo = UGapEb(n_arms=n_arms, sigma=sigma_noise, horizon=horizon)
    #     algo.initialize(n_arms)

    #     results = test_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    elif algo_name == 'OptNN-H' or algo_name == 'OptNN-H-Lazy':
        random.seed(seed)

        if n_arms == 3 and args.arm_dist == 'Ber':
            model_path = Path(PJT_PATH / 'checkpoint' / f'{n_arms}_{arm_dist}' /
                              # NOTE: SETUP THE TRAINED POLICY MODEL name HERE
                              # e.g.,
                              '0' / 'policy-2_0_33_valloss-0.98.pth'   # <===== REWRITE HERE IF YOU WANT
                              )
        else:
            raise NotImplementedError
        print(f'==> Model path is {model_path}')
        if 'Lazy' not in algo_name:
            algo = OptNN(model_path=model_path, n_arms=n_arms, sigma=sigma_noise, horizon=horizon)
        else:
            algo = OptNN(model_path=model_path, n_arms=n_arms, sigma=sigma_noise, horizon=horizon, lazy_policy_update=True)
        algo.initialize(n_arms)

        results = test_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    elif algo_name == 'OptNN-OptH':
        random.seed(seed)
        if n_arms == 3 and args.arm_dist == 'Normal':
            model_path = Path(PJT_PATH / 'checkpoint' / f'{n_arms}_{arm_dist}' /
                              '0' /
                              'policy_1_02_valloss-0.71.pth'
                              )
        elif n_arms == 5 and args.arm_dist == 'Normal':
            model_path = Path(PJT_PATH / 'checkpoint' / f'{n_arms}_{arm_dist}' /
                              '0' /  # <- use same data and num_sampled for a lot
                              'policy_1_02_valloss-0.44.pth'
                              )
        else:
            raise NotImplementedError
        print(f'==> Model path is {model_path}')
        algo = OptNN(model_path=model_path, n_arms=n_arms, sigma=sigma_noise, horizon=horizon)
        algo.initialize(n_arms)

        results = test_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    else:
        raise NotImplementedError

    print(f'==> Saving to {save_name}')
    results.to_csv(save_name)


if __name__ == '__main__':
    main()