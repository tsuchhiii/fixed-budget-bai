import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from fixed_budget_pjt.utils import H_ndarray
from fixed_budget_pjt.utils import H_from_gap

from fixed_budget_pjt.utils import get_kth_max
from mab_setting import gen_pe_setting


# todo: refactor


class BatchTrueMeansDataset(torch.utils.data.Dataset):
    def __init__(self, n_arms, arm_dist, n_data, n_means_for_loss):
        """ Sample true arms means from corresponding distribution

        Args:
            n_arms:
            arm_dist:
            n_data: n_data * n_sampled_true_means == all used true means
            n_means_for_loss: true means for each loss
        """
        self.n_arms = n_arms
        self.n_means_for_loss = n_means_for_loss
        self.n_data = n_data

        all_true_means = \
            np.random.uniform(size=n_data * n_means_for_loss * n_arms).reshape(n_data, n_means_for_loss, n_arms)

        self.all_true_means = all_true_means

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        return self.all_true_means[idx]


class BatchEmpMeansDataset(object):  # todo update name
    """ sample emp_means for each best arm """
    # value is changed for each mini-batch or xxx
    def __init__(self, batch_true_means: np.ndarray, n_arms, arm_dist, n_sampled_emp_means=100):
        n_batch, n_means_for_loss, n_arms = batch_true_means.shape
        batch_true_means = batch_true_means.reshape(n_batch * n_means_for_loss, n_arms)

        self.n_arms = n_arms

        # Sample Q such that best arm is different from that of true means
        self.emp_means_per_true_means_best_list = []
        for arm in range(n_arms):
            tmp_multiple = n_arms * 3
            tmp_emp_means = np.random.uniform(size=tmp_multiple * n_sampled_emp_means * n_arms).reshape(
                tmp_multiple * n_sampled_emp_means, n_arms)

            # A. 0-optimal arm
            emp_means_for_bestarm = tmp_emp_means[np.argmax(tmp_emp_means, axis=1) != arm][:n_sampled_emp_means]  # todo
            assert emp_means_for_bestarm.shape == (n_sampled_emp_means, n_arms)
            self.emp_means_per_true_means_best_list.append(emp_means_for_bestarm)

        all_emp_means = [self.emp_means_per_true_means_best_list[best_arm] for best_arm in np.argmax(batch_true_means, axis=1)]
        all_emp_means = np.array(all_emp_means)
        all_emp_means = torch.FloatTensor(all_emp_means)
        assert all_emp_means.shape == (n_batch * n_means_for_loss, n_sampled_emp_means, n_arms)
        self.all_emp_means = all_emp_means.reshape(n_batch, n_means_for_loss, n_sampled_emp_means, n_arms)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.all_emp_means[idx]


class BatchEmpMeansDatasetForPlot(object):  # todo update name
    """ sample emp_means for each best arm """
    # value is changed for each mini-batch or something
    def __init__(self, batch_true_means: np.ndarray, n_arms, arm_dist, setting_idx, n_sampled_emp_means=100):
        """
        NOTE THAT this is special versions of BatchEmpMeansDataset class
            for plot: bool, (for plot_pe.py) sample n_sampled_emp_means number of samples in [0, 1] `for each arm`
                (only in this setting), and remove pairs where argmax of is different from the best arm
        """
        n_batch, n_means_for_loss, n_arms = batch_true_means.shape
        batch_true_means = batch_true_means.reshape(n_batch * n_means_for_loss, n_arms)

        self.n_arms = n_arms

        assert batch_true_means.shape == (1, n_arms)
        assert n_arms == 3

        # reward_means = np.array([0.5, 0.495, 0.49])  # todo, replace with true means for each setting
        reward_means = gen_pe_setting(n_arms, setting_idx)
        assert np.argmax(reward_means) == 0, 'only for this for now'
        max_reward = np.max(reward_means)
        min_reward = np.min(reward_means)
        # todo: see Carpentier for the safe range of values
        max_range = min(1., max_reward + (max_reward - min_reward))
        min_range = max(0., min_reward - (max_reward - min_reward))
        # q_range = max_reward - min_reward
        tmp_n_sampled_emp_means = int((max_range - min_range) / 0.005)  # TODO: NOW the interval is fixed regardless of min or max value
        x = np.linspace(min_range, max_range, tmp_n_sampled_emp_means)  # arm 1
        y = np.linspace(min_range, max_range, tmp_n_sampled_emp_means)  # arm 2
        z = np.linspace(min_range, max_range, tmp_n_sampled_emp_means)  # arm 3

        tmp_means_mesh = np.meshgrid(x, y, z)
        tmp_means_array = np.vstack(map(np.ravel, tmp_means_mesh)).T   # shape == (# of combination, n_arms)

        best_arm = np.argmax(batch_true_means)
        MACHINE_EPS = 1e-8  # todo
        all_emp_means = tmp_means_array[np.max(tmp_means_array, axis=1) > tmp_means_array[:, best_arm] + MACHINE_EPS]
        self.all_emp_means = all_emp_means.reshape(1, 1, -1, n_arms)

        # import ipdb; ipdb.set_trace()


        # make this for pseudo computing in loss.py
        self.emp_means_per_true_means_best_list = []
        for arm in range(n_arms):
            tmp_all_emp_means = tmp_means_array[np.max(tmp_means_array, axis=1) > tmp_means_array[:, arm] + MACHINE_EPS]
            self.emp_means_per_true_means_best_list.append(tmp_all_emp_means)

        # import ipdb; ipdb.set_trace()
        print(f'size is {self.all_emp_means.shape}')

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.all_emp_means[idx]


class BestArmEmpMeansDataset(torch.utils.data.Dataset):  # todo update name
    """ sample emp_means for each best arm """
    def __init__(self, n_arms, arm_dist, n_sampled_emp_means=100):
        self.n_arms = n_arms

        # Sample Q such that best arm is different from that of true means
        self.all_emp_means_list = []
        for arm in range(n_arms):
            tmp_multiple = n_arms * 3
            tmp_emp_means = np.random.uniform(size=tmp_multiple * n_sampled_emp_means * n_arms).reshape(
                tmp_multiple * n_sampled_emp_means, n_arms)

            # A. 0-optimal arm  ==> this seems better now
            all_emp_means = tmp_emp_means[np.argmax(tmp_emp_means, axis=1) != arm][:n_sampled_emp_means]  # todo
            # # B. hoge-optimal arm
            # cond = np.max(tmp_true_means, axis=1) > np.max(tmp_true_means[:, 1:], axis=1) + 1e-3
            # self.all_true_means = tmp_emp_means[np.max(tmp_true_means, axis=1) > np.max(tmp_true_means[:, 1:], axis=1) + hoge][:n_sampled_emp_means]
            assert all_emp_means.shape == (n_sampled_emp_means, n_arms)
            self.all_emp_means_list.append(all_emp_means)

    def __len__(self):
        # return len(self.all_emp_means)
        return self.n_arms

    def __getitem__(self, idx):
        # return self.all_emp_means[idx]
        return self.all_emp_means_list[idx]


class TrueMeansDataset(torch.utils.data.Dataset):
    def __init__(self, n_arms, arm_dist, n_sampled_true_means):
        """ Sample true arms means such that arm 1 is optimal

        Args:
            n_arms:
            arm_dist:
            n_sampled_true_means: approx value now
        """
        self.n_arms = n_arms
        self.n_sampled_true_means = n_sampled_true_means

        all_true_means = np.random.uniform(size=n_sampled_true_means * n_arms).reshape(n_sampled_true_means, n_arms)

        self.all_true_means = all_true_means

    def __len__(self):
        # todo: fix
        return self.all_true_means.shape[0]

    def __getitem__(self, idx):
        # todo: need efficient implementation
        return self.all_true_means[idx]


class EmpMeansDataset(torch.utils.data.Dataset):
    def __init__(self, true_means: np.ndarray, n_arms, arm_dist, n_sampled_emp_means=100):
        """ Sample empirical arms means such that arm 1 is NOT optimal where
        the emp_means is different from given true_means only in one arm
        (brutal force here)

        ==> updated for any best arm case

        Args:
            true_means:
            n_arms:
            arm_dist:
        """
        self.n_arms = n_arms

        '''
        Sample Q such that best arm is different from that of true means
        '''
        tmp_multiple = n_arms * 3
        tmp_emp_means = np.random.uniform(size=tmp_multiple * n_sampled_emp_means * n_arms).reshape(
            tmp_multiple * n_sampled_emp_means, n_arms)

        best_arm_of_true_means = np.argmax(true_means)

        self.all_emp_means = tmp_emp_means[np.argmax(tmp_emp_means, axis=1) != best_arm_of_true_means][:n_sampled_emp_means]  # todo

        assert self.all_emp_means.shape == (n_sampled_emp_means, n_arms)

    def __len__(self):
        return len(self.all_emp_means)

    def __getitem__(self, idx):
        return self.all_emp_means[idx]


def emp_means_generator(true_means: np.ndarray, n_arms, eps=1e-3):
    """ if eps is small, then w/ small prob. the kl can be H * kl can be negative
    """
    # for fast computation of EmpMeasDataset w/o loader
    best_arm = np.argmax(true_means)
    all_emp_means = np.array([true_means for _ in range(n_arms)])

    for arm_idx in range(n_arms):
        all_emp_means[arm_idx][arm_idx] = true_means[best_arm] + eps

    all_emp_means = np.delete(all_emp_means, best_arm, axis=0)
    assert all_emp_means.shape == (n_arms - 1, n_arms)

    return all_emp_means

    # for emp_means in all_emp_means:
    #     # yield emp_means
    #     yield torch.FloatTensor(emp_means[None, :])


# class H1Dataset(TrueMeansDataset):
class H1Dataset(object):
    def __init__(self, n_arms, arm_dist,
                 # n_sampled_true_means=1000,
                 n_sampled_reward_gaps=1000,
                 rotate_aug=False):
        """ Sample true arms means such that arm 1 is optimal
        Dataset class for pre-training complexity network

        Args:
            n_arms:
            n_sampled_true_means:
        """
        # todo: need to fix data generation process because the gap distribution is different from arm prior dist.
        self.n_arms = n_arms
        self.rotate_aug = rotate_aug
        self.n_sampled_reward_gaps = n_sampled_reward_gaps
        # tmp_multiple = 1.1
        tmp_multiple = 1.5
        n_tmp_sampled = int(np.ceil(tmp_multiple * n_sampled_reward_gaps))

        hoge = 1e-3
        one_over_gaps = np.random.uniform(low=1., high=1./hoge, size=n_tmp_sampled * n_arms).reshape(n_tmp_sampled, n_arms)
        gap_array = 1./np.sqrt(one_over_gaps)
        tmp_true_means = (1. - gap_array).reshape(n_tmp_sampled, n_arms)

        sorted_true_means = np.sort(tmp_true_means, axis=1)[:, ::-1]  # descending order
        valid_idx = np.max(sorted_true_means, axis=1) - np.max(sorted_true_means[:, 1:], axis=1) > hoge
        all_true_means = sorted_true_means[valid_idx][:n_sampled_reward_gaps]
        max_mean_broadcast = np.broadcast_to(np.max(all_true_means, axis=1), (n_arms, n_sampled_reward_gaps)).T
        self.all_reward_gaps = max_mean_broadcast - all_true_means

        self.all_true_means = all_true_means

        assert self.all_reward_gaps.shape == (n_sampled_reward_gaps, n_arms), \
            f'all_true_means {self.all_reward_gaps.shape} != ({n_sampled_reward_gaps},{n_arms - 1}),' \
            f'increase tmp_multiple'

    def __len__(self):
        return self.n_sampled_reward_gaps

    def __getitem__(self, idx):
        return self.all_true_means[idx], H_ndarray(self.all_true_means[idx])

