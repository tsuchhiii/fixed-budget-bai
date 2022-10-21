import numpy as np
import torch

from torch._six import inf

from fixed_budget_pjt.config import ENV, MODEL
from fixed_budget_pjt.dataset import BatchEmpMeansDataset, BatchEmpMeansDatasetForPlot
from fixed_budget_pjt.utils import H_batch_tensor, H_tensor, kl_func


class FixedBudgetLoss(object):
    def __init__(self, n_arms, arm_dist, policy_model, complexity_model):
        self.n_arms = n_arms
        self.arm_dist = arm_dist
        self.policy_model = policy_model
        self.complexity_model = complexity_model

        assert hasattr(complexity_model, 'training') or complexity_model.__name__ == 'H_batch_tensor'
        self.pretrain_policy = not hasattr(complexity_model, 'training')

    def compute_critical_point(self, batch_true_means, n_sampled_emp_means, _get_minmax_point=False, _for_plot_setting_idx=None):
        # Note. codes for other purposes are included here
        # batch_true_means.shape == (batch_size, num_true_means_for_loss, n_arms)

        # config
        n_arms = self.n_arms
        arm_dist = self.arm_dist
        policy_model = self.policy_model
        complexity_model = self.complexity_model

        # assert not policy_model.training  # commentted out just for find_hard_instance.py
        assert (hasattr(complexity_model, 'training') and not complexity_model.training) \
               or complexity_model.__name__ == 'H_batch_tensor'

        # forward in advance for fast computation
        n_batch, n_means_for_loss, n_arms = batch_true_means.shape  # todo: use global var
        flatten_true_means = batch_true_means.reshape(-1, n_arms)

        # 1. forward: compute complexity of true means H(P)
        if self.pretrain_policy:
            H_batch = complexity_model(flatten_true_means)  # no need to sort
        else:
            H_batch = complexity_model(torch.sort(flatten_true_means, dim=1, descending=True)[0])  # need sort because comp. nn assumes it
        H_batch = H_batch.reshape(-1, n_means_for_loss)
        assert H_batch.shape == (n_batch, n_means_for_loss)

        # 2. forward: compute policy for given empirical means pi(Q)
        # | Now all the same values are used for empirical means in mini-batch if best arm in true means are
        # | equal for fast computation of empirical means generation
        # | Note that the empirical means still differ for each mini-batch iteration
        # n_sampled_emp_means = MODEL.NUM_EMP_MEANS
        if _for_plot_setting_idx:
            emp_means_dataset = BatchEmpMeansDatasetForPlot(batch_true_means.to('cpu').detach().numpy().copy(),
                                                            n_arms=n_arms, arm_dist=arm_dist,
                                                            n_sampled_emp_means=n_sampled_emp_means,
                                                            setting_idx=_for_plot_setting_idx)
            n_sampled_emp_means = emp_means_dataset.all_emp_means.shape[2]   # todo: override this variable only for plotting purpose
        else:
            emp_means_dataset = BatchEmpMeansDataset(batch_true_means.to('cpu').detach().numpy().copy(),
                                                     n_arms=n_arms, arm_dist=arm_dist,
                                                     n_sampled_emp_means=n_sampled_emp_means)
        # policy_list = []
        policy_dict = dict()   # dict mapping from `best_arm_idx' to `output of policy_model'
        for best_arm_idx, emp_item in enumerate(emp_means_dataset.emp_means_per_true_means_best_list):
            emp_means_sampled = torch.FloatTensor(emp_item).to(ENV.DEVICE).float()
            policy = policy_model(emp_means_sampled)  # policy.shape == (n_sampled_emp_means, n_arms)
            policy_dict[best_arm_idx] = policy

        # 3. compute kl in batch manner by reshaping or extending given data to
        # the array with (n_batch, n_means_for_loss, n_sampled_emp_means, n_arms)
        flatten_emp_means = torch.FloatTensor(emp_means_dataset.all_emp_means.reshape(-1, n_arms)).to(ENV.DEVICE)  # flatten_emp_means.shape == (B * num_true_means_for_loss * n_sampled_emp_means, n_arms)
        extended_true_means = torch.broadcast_to(
            batch_true_means, (n_sampled_emp_means, n_batch, n_means_for_loss, n_arms)
        ).transpose(0, 1).transpose(1, 2)  # extend_true_means.shape == (B, num_true_means_for_loss, n_sampled_emp_means, n_arms)
        flatten_extended_true_means = extended_true_means.reshape(-1, n_arms)
        kl_vec_batch = kl_func(flatten_emp_means, flatten_extended_true_means, arm_dist=arm_dist)
        kl_vec_batch = kl_vec_batch.reshape(-1, n_means_for_loss, n_sampled_emp_means, n_arms)

        # 4. compute inf of negative exponent in probability of error P^min, Q^min(P^min) from forwarded values
        true_means_min_list = []
        emp_means_minmin_list = []

        if _get_minmax_point:
            true_means_max_list = []
            emp_means_minmax_list = []

        # for each item in minibatch (size of (n_batch, n_true_means_for_loss))
        for idx_minibatch, sampled_true_means in enumerate(batch_true_means):  # sampled_true_means.shape == (num_true_means_for_loss, n_arms)
            minmin_exponent = inf
            true_means_min, emp_means_minmin = None, None  # math: P^min, Q^min(P^min)

            if _get_minmax_point:
                maxmin_exponent = -inf
                true_means_max, emp_means_minmax = None, None  # math: P^max, Q^min(P^max)

                min_exponent_given_true_means = inf  # math: H(P) inf_Q \sum_k pi_k(Q) KL(Q||P(Q)) for given P
                emp_means_min_given_true_means = None  # math: Q^min(P)

            # for each item in sampled true means (in minibatch)
            # start of loop for compute loss of `one` data
            for idx_sample, true_means in enumerate(sampled_true_means):   # true_means.shape == (n_arms,)
                best_arm = torch.max(true_means, 0)[1]
                # tmp_exponent = compute_negative_exponent(true_means, emp_means, policy_model, complexity_model)
                tmp_H = H_batch[idx_minibatch][idx_sample]
                tmp_policy = policy_dict[best_arm.item()]
                # math:
                # | inf_policy_prod_kl -> inf_Q sum_k pi_k(Q) KL(Q||P)
                # | emp_means_min_idx -> Q which gives inf of above quantity
                inf_policy_prod_kl, emp_means_min_idx = \
                    torch.min((tmp_policy * kl_vec_batch[idx_minibatch][idx_sample]).sum(axis=1), dim=0)
                emp_means_min = emp_means_dataset.emp_means_per_true_means_best_list[best_arm][emp_means_min_idx]  # todo: refactor
                tmp_exponent = tmp_H * inf_policy_prod_kl

                if tmp_exponent < minmin_exponent:
                    minmin_exponent = tmp_exponent
                    true_means_min = true_means
                    emp_means_minmin = emp_means_min

                if _get_minmax_point and tmp_exponent < min_exponent_given_true_means:
                    min_exponent_given_true_means = tmp_exponent
                    emp_means_min_given_true_means = emp_means_min

                if _get_minmax_point and min_exponent_given_true_means > maxmin_exponent:
                    maxmin_exponent = min_exponent_given_true_means
                    true_means_max = true_means
                    emp_means_minmax = emp_means_min_given_true_means

            true_means_min_list.append(true_means_min)
            emp_means_minmin_list.append(torch.FloatTensor(emp_means_minmin).to(ENV.DEVICE))

            if _get_minmax_point:
                true_means_max_list.append(true_means_max)
                emp_means_minmax_list.append(torch.FloatTensor(emp_means_minmax).to(ENV.DEVICE))

        if _get_minmax_point:
            return true_means_min_list, emp_means_minmin_list, true_means_max_list, emp_means_minmax_list
        else:
            return true_means_min_list, emp_means_minmin_list

    def compute_negative_exponent(self, true_means: torch.FloatTensor, emp_means: torch.FloatTensor, no_H=False):
        """ math: + H(P) sum_k pi_k(Q) KL(Q_k(P)||P_k)
        if no_H, then return sum_k pi_k(Q) KL(Q_k(P)||P_k)
        """
        if self.pretrain_policy:
            H = self.complexity_model(true_means)  # no need to sort
        else:
            H = self.complexity_model(torch.sort(true_means, dim=1, descending=True)[0])
        policy = self.policy_model(emp_means)
        klvec = kl_func(emp_means, true_means, arm_dist=self.arm_dist)
        assert policy.shape == klvec.shape

        if no_H:
            return torch.sum(policy * klvec, axis=1)[:, None]  # shape == (batch_size, 1)

        return H * torch.sum(policy * klvec, axis=1)[:, None]  # shape == (batch_size, 1)


class PolicyLoss(FixedBudgetLoss):
    def __init__(self, n_arms, arm_dist, policy_model, complexity_model):
        """ Policy loss when the complexity measure is fixed """
        super(PolicyLoss, self).__init__(n_arms, arm_dist, policy_model, complexity_model)

    def compute_policy_loss(self, true_means_min_list, emp_means_minmin_list):
        """ math: mini-batch sum of - H(P) sum_k pi_k(Q) KL(Q_k(P)||P_k) """
        loss = torch.tensor(0.).to(ENV.DEVICE)
        for idx in range(len(true_means_min_list)):
            loss += - self.compute_negative_exponent(
                true_means_min_list[idx][None, :], emp_means_minmin_list[idx][None, :]
            )[0][0]

        if loss > 0:
            print(loss)
        return loss






