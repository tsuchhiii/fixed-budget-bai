"""
Plotting code for Pure Exploration
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
import torch
from torch.utils.data import DataLoader

from fixed_budget_pjt.dataset import BatchTrueMeansDataset
from fixed_budget_pjt.loss import PolicyLoss

sns.set()
sns.set_style('white')
palette = sns.color_palette("colorblind", n_colors=6)
sns.set_palette(palette)

import argparse

from util import split_save_path, get_pemab_fig_basename
from util import get_pemab_save_path, get_pemab_save_basename
from config import config_pe_mab
from fixed_budget_pjt.config import MODEL, ENV

import sys
from mab_setting import gen_pe_setting
from fixed_budget_pjt.utils import H_ndarray, H_batch_tensor, load_model


def change_name_for_plot(algo_name):
    return algo_name if algo_name != 'OptNN-H' else 'TNN'

is_log_scale = 0
noise_var = 1.0

assert noise_var in [0.01, 0.1, 1.0]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_arms', '-na', required=True, type=int,
                        help='# of arms')
    parser.add_argument('--arm_dist', '-ad', required=True, type=str,
                        help='arm distribution')
    parser.add_argument('--setting', '-s', required=True, type=int, default=None,
                        choices=config_pe_mab.SETTING_CHOICES)
    parser.add_argument('--horizon', '-ho', required=True, type=int, # default=10000,
                        help='horizon')
    parser.add_argument('--num_sims', '-ns', required=True, type=int, # default=10000,
                        help='horizon')
    args = parser.parse_args()

    return args


def plt_save(path):
    """ tmp func """
    directory, filename = os.path.split(path)
    if directory == '':
        directory = '.'
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_fig_path = os.path.join(directory, filename)
    plt.savefig(save_fig_path,
                bbox_inches='tight',
                pad_inches=0.2,
                )


args = get_args()
print(args)
n_arms = args.n_arms
arm_dist = args.arm_dist
setting = args.setting
horizon = args.horizon

''' 
configuration 
'''
model = 'pe-mab'  # pure exploration for MAB

# todo: need refactor
target_metrics = 'pseudo_regret' if 'pe' not in model else 'simple_regret'

algo_choices = config_pe_mab.ALGO_CHOICES


# todo: need refactoring
saved_path_list = [get_pemab_save_path(algo_name, setting, arm_dist, n_arms, noise_var)
                   for algo_name in algo_choices]

'''
plot setting
'''
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 9
plt.rcParams['figure.figsize'] = (3., 3.)
plt.rcParams['figure.dpi'] = 100

'''
plot each method performance
'''
fig = plt.figure()
axes = fig.add_subplot(111)

'''
for line-style setting
'''
linestyle_list = ['dashdot', 'dashed', 'solid', '-', '-', '-', '-', '-', '-', '-', '-'][:len(saved_path_list)]
marker_list = ['o', 'v', 'D', 'h', '^', '<', 'p', 's', '*'][:len(saved_path_list)]

algo_name_list = []
prob_of_error_list = []

for saved_name, linestyle in zip(saved_path_list, linestyle_list):
    algo_name, setting, _, _, _ = split_save_path(saved_name)
    setting = int(setting)

    if os.path.exists(saved_name):
        try:
            print(f'Trying to load {saved_name}..')
            results_df = pd.read_csv(saved_name)
        except:
            print(f'Cannot load {saved_name}')
            raise NotImplementedError
    else:
        print('!!! the file {} does not exist'.format(saved_name))
        continue

    results_df = results_df[results_df.times <= horizon]

    # ==
    saved_time_array = results_df.loc[:, 'times'].unique()
    num_save_step_per_sim = saved_time_array.shape[0]  # e.g., 1000
    # num_dataplot_per_sim = 20
    num_dataplot_per_sim = num_save_step_per_sim // 2  # todo: decide for clear plotting
    # ==

    regret_df = results_df.loc[:, ['times', target_metrics]]
    mean_regret_df = regret_df.groupby('times').mean()
    std_regret_df = regret_df.groupby('times').std()

    mean_regret_df.plot(# x='times',
                        y=target_metrics,
                        ax=axes,
                        label=change_name_for_plot(algo_name),
                        # colormap='Accent',
                        grid=True, legend=True,
                        alpha=0.8,
                        # marker=marker,
                        # markersize=1,
                        # markeredgecolor=None,
                        linestyle=linestyle,
                        )

    if is_log_scale:
        raise NotImplementedError
    else:
        tsav = np.linspace(0, num_save_step_per_sim-1, num_dataplot_per_sim, dtype=int)

    mean = np.array(mean_regret_df).flatten()[tsav]
    dev = np.array(std_regret_df).flatten()[tsav]
    plt.fill_between(saved_time_array[tsav], mean - dev, mean + dev,
                     mean + dev, # facecolor=mycolor,
                     alpha=0.125)

    '''
    plot probability of error 
    '''
    estimated_arm_series = results_df[results_df['times'] == horizon - 1].loc[:, 'estimated_best_arm_idxes']
    best_arm = 0  # todo
    print('REMARK. Assuming that best arm is {} for now'.format(best_arm) * 5)
    num_sims = len(estimated_arm_series)
    assert num_sims == args.num_sims
    algo_name_list.extend([algo_name] * num_sims)
    prob_of_error_array = (estimated_arm_series != best_arm).values.astype(int)
    prob_of_error_list.extend(prob_of_error_array)


plt.ylim(bottom=0.0)

# for log plot
if is_log_scale:
    axes.set_xscale('log')  # todo

if is_log_scale:
    axes.set_xlabel(r'logarithm of round $\log_{10} T$')
else:
    axes.set_xlabel(r'round $t$')

axes.set_ylabel(target_metrics.replace('_', ' '))
axes.grid(False)

fig_basename = get_pemab_fig_basename(setting, arm_dist, n_arms, noise_var)
if is_log_scale:
    save_fig_path = f'results/{model}/fig/{n_arms}_{arm_dist}/{fig_basename}_log-scale.pdf'
else:
    save_fig_path = f'results/{model}/fig/{n_arms}_{arm_dist}/{fig_basename}.pdf'
# todo: save for log plot
print(f'saving to {save_fig_path}')
plt_save(save_fig_path)
plt.close()


'''
prob of error 
'''
prob_of_error_df = pd.DataFrame({
    # 'algo_name': algo_name_list,
    'algo_name': map(change_name_for_plot, algo_name_list),
    'prob_of_error': prob_of_error_list,
})

# s = sns.barplot(x='algo_name', y='prob_of_error', data=prob_of_error_df, capsize=.1, errwidth=1.0)
s = sns.barplot(x='algo_name', y='prob_of_error', data=prob_of_error_df, capsize=.1, errwidth=1.0, ci=95)
save_poe_fig_path = f'results/{model}/fig/{n_arms}_{arm_dist}/poe_{fig_basename}.pdf'

s.set_xlabel('algorithm')
s.set_ylabel('probability of error')

print(f'saving to {save_poe_fig_path}')
# plt.savefig(save_poe_fig_path,
#             bbox_inches='tight',
#             pad_inches=0.2,
#             )
plt_save(save_poe_fig_path)

###########################
###########################
###########################
###########################
###########################
###########################

'''
horizon vs prob of error 
'''
'''
plotting time for x-axis, log of probability of error for y-axis
'''
fig = plt.figure()
axes = fig.add_subplot(111)
# here is just for one trial plot
# for saved_name in saved_name_list:
for saved_name, linestyle in zip(saved_path_list, linestyle_list):
    # method, setting = get_alg_and_setting(saved_name)
    algo_name, setting, _, _, _ = split_save_path(saved_name)
    setting = int(setting)

    if os.path.exists(saved_name):
        print(f'Loading from {saved_name}')
        results_df = pd.read_csv(saved_name)
    else:
        print('!!! the file {} does not exist'.format(saved_name))
        continue

    results_df = results_df[results_df.times <= horizon]

    ### partially taken from plot_linear_bandits.py
    saved_time_array = results_df.loc[:, 'times'].unique()
    num_save_step_per_sim = saved_time_array.shape[0]  # e.g., 1000
    # num_dataplot_per_sim = 20  # todo: decide for clear plotting
    num_dataplot_per_sim = num_save_step_per_sim // 2  # todo: decide for clear plotting
    ###
    '''
    plot time for x-axis, log of probability of error for y-axis
    '''
    serror_df = results_df.loc[:, ['times', 'error']]
    mean_serror_df = serror_df.groupby('times').mean()
    std_serror_df = serror_df.groupby('times').std()  # todo
    mean_serror_df.plot(# x='times',
        y='error',
        ax=axes,
        # title='cumulative regret',
        label=change_name_for_plot(algo_name),
        # colormap='Accent',
        grid=True, legend=True,
        alpha=0.8,
        linestyle=linestyle
    )
    if algo_name == 'Uniform':
        bottom_poe_2 = mean[-1] * 0.5   # todo: scale for now


# plot true tilt of eop
def compute_true_eop(time_array, setting):
    means = gen_pe_setting(n_arms, setting)
    H = H_ndarray(means)
    return np.exp(- time_array / H)

tsav = np.asarray(np.linspace(0, horizon-1, horizon//10), dtype=int)
# not plotting this for now
# plt.plot(tsav, compute_true_eop(tsav, setting), linewidth=0.5, linestyle='-.', color='k',
#          label=r'$\exp(-T/H)$')


def load_trained_policy_model(comp):
    # Remark. MUST be same with the name in test_pe_mab.py
    if comp == 'H':
        if n_arms == 3 and arm_dist == 'Ber':
            trained_policy_model_name = '0/policy-2_0_33_valloss-0.98.pth'
            # trained_policy_model_name = '2/policy-2_0_33_valloss-0.98.pth'   # TODO: need to specify here
        # elif n_arms == 3 and arm_dist == 'Normal':
        #     raise NotImplementedError
        #     trained_policy_model_name = '56/policy_0_33_valloss-0.19.pth'
        else:
            raise NotImplementedError
    elif comp == 'OptH':
        raise NotImplementedError
    else:
        raise NotImplementedError

    policy_model_name = f'{n_arms}_{arm_dist}/{trained_policy_model_name}'
    policy_model = load_model(MODEL.PRE_CHECKPOINT_PATH / policy_model_name)

    # todo: now Only complexity H
    return policy_model


# plot estimated slope by NN
def compute_estimated_eop(time_array, setting, policy_model):
    """ math: for given setting of P, compute H(P) inf_Q sum_k pi_k(Q) KL(Q_k||P_k)
     """
    # 0. get setting of P
    means = gen_pe_setting(n_arms, setting)
    true_means = torch.FloatTensor(means[None, None, :])
    print(f'Computing estimated negative exponent for given means: {means}')

    # 1. load policy model
    # policy_model = load_trained_policy_model(comp='H')
    # policy_model.eval()

    # 2. compute inf_Q sum_k pi_k(Q) KL(Q_k || P_k) for given true_means P
    policy_loss_inst = PolicyLoss(n_arms, arm_dist, policy_model, H_batch_tensor)
    true_means_minmin_list, emp_means_minmin_list = \
        policy_loss_inst.compute_critical_point(true_means, n_sampled_emp_means=101, _for_plot_setting_idx=setting)
    neg_exponent_no_H = policy_loss_inst.compute_negative_exponent(true_means_minmin_list[0][None, :], emp_means_minmin_list[0][None, :], no_H=True)
    print(f'{true_means_minmin_list[0][None, :]=}, {emp_means_minmin_list[0][None, :]=}')

    neg_exponent_no_H = neg_exponent_no_H.to('cpu').detach().numpy().copy().squeeze()
    print('inf_Q sum_i pi_i(Q) KL(Q_i||P_i) (neg exponent w/o H) is ', neg_exponent_no_H)

    return np.exp(- time_array * neg_exponent_no_H), neg_exponent_no_H, emp_means_minmin_list[0]


# todo: don't plot this for now
policy_model = load_trained_policy_model(comp='H')
policy_model.eval()
estimated_eop, neg_exponent_no_H, Q_inf = compute_estimated_eop(tsav, setting, policy_model)
tnn_intercept = 0.3
plt.plot(tsav, tnn_intercept * estimated_eop, linewidth=0.5, linestyle='-', color='k')

# end of new coding

plt.xlim(left=0.0, right=horizon)
# plt.ylim(bottom=bottom_poe_2, top=2.0)  # todo
plt.legend()

'''
horizon vs prob of error 
'''
# plt.ylim(bottom=0.0)

axes.set_yscale('log')  # todo

axes.set_xlabel(r'round $t$')
# axes.set_xlabel(r'round $T$')
axes.set_ylabel(r'$\log_{10}(\mathbb{P}[J(t) \notin \mathcal{I}^*])$')
# axes.set_ylabel(r'$\log_{10}(\Pr[J_t \neq I^*])$')
axes.grid(False)

fig_basename = get_pemab_fig_basename(setting, arm_dist, n_arms, noise_var)
save_fig_path = f'results/{model}/fig/{n_arms}_{arm_dist}/poe2_{fig_basename}.pdf'
# todo: save for log plot
# save_fig_path = 'results/fig/{}_na_{}_no_{}_log.pdf'.format(setting, n_action, n_outcome)
print(f'saving to {save_fig_path}')
# plt.savefig(save_fig_path,
#             bbox_inches='tight',
#             # pad_inches=0.05,
#             pad_inches=0.2,
#             )
plt_save(save_fig_path)
plt.close()


'''
horizon - track metric
'''
fig = plt.figure()
axes = fig.add_subplot(111)
for saved_name, linestyle in zip(saved_path_list, linestyle_list):
    algo_name, setting, _, _, _ = split_save_path(saved_name)
    setting = int(setting)

    # if 'OptNN' not in algo_name:
    #     continue
    if algo_name != 'OptNN-H':
        continue
    # if algo_name != 'OptNN-H-Lazy':  # todo: for now (for quite large horizon)
    #     continue

    if os.path.exists(saved_name):
        print(f'Loading from {saved_name}')
        results_df = pd.read_csv(saved_name)
    else:
        print('!!! the file {} does not exist'.format(saved_name))
        continue

    results_df = results_df[results_df.times <= horizon]

    saved_time_array = results_df.loc[:, 'times'].unique()
    num_save_step_per_sim = saved_time_array.shape[0]  # e.g., 1000
    # num_dataplot_per_sim = 20  # todo: decide for clear plotting
    num_dataplot_per_sim = num_save_step_per_sim // 2  # todo: decide for clear plotting
    ###
    '''
    plot time for x-axis, track-metric for y-axis
    '''
    track_df = results_df.loc[:, ['times', 'track_metric']]
    mean_track_df = track_df.groupby('times').mean()
    std_track_df = track_df.groupby('times').std()
    mean_track_df.plot(# x='times',
        y='track_metric',
        ax=axes,
        # title='cumulative regret',
        label=change_name_for_plot(algo_name) + ' (average)',
        # colormap='Accent',
        grid=True, legend=True,
        alpha=0.8,
        linestyle='dotted'
        # linestyle = linestyle
    )
    max_track_df = track_df.groupby('times').max()
    max_track_df.plot(# x='times',
        y='track_metric',
        ax=axes,
        label=change_name_for_plot(algo_name) + ' (worst)',
        grid=True, legend=True,
        alpha=0.8,
        linestyle='dashed'
    )
    tsav = np.linspace(0, num_save_step_per_sim-1, num_dataplot_per_sim, dtype=int)
    mean = np.array(mean_track_df).flatten()[tsav]
    dev = np.array(std_track_df).flatten()[tsav]
    plt.fill_between(saved_time_array[tsav], mean - dev, mean + dev,
                     # facecolor=mycolor,
                     alpha=0.125)
    '''
    plot differently by the prediction was correct or not
    '''
    rich_metricr_df = results_df.loc[results_df['times'] == horizon - 1, ['times', 'sim_nums', 'error', 'track_metric']]
    error_series = results_df.loc[results_df['times'] == horizon - 1, 'error']
    num_sims = args.num_sims
    fail_sim_num_array = np.arange(1, 1+num_sims)[(error_series == 1)]  # sim_nums starts is 1-indexed
    fail_track_df = track_df[results_df['sim_nums'].isin(fail_sim_num_array)]
    # success_sim_num_array = np.arange(1, 1+num_sims)[(error_series == 0)]
    # success_track_df = track_df[results_df['sim_nums'].isin(success_sim_num_array)]
    fail_mean_track_df = fail_track_df.groupby('times').mean()
    fail_std_track_df = fail_track_df.groupby('times').std()
    fail_mean_track_df.plot(# x='times',
        y='track_metric',
        ax=axes,
        label=change_name_for_plot(algo_name) + ' (average in fail)',
        grid=True, legend=True,
        alpha=1.0,
        linestyle='solid'
    )

    # TODO
    if len(fail_mean_track_df) == 0:
        continue

    fail_mean = np.array(fail_mean_track_df).flatten()[tsav]
    fail_dev = np.array(fail_std_track_df).flatten()[tsav]
    # plt.fill_between(tsav + 1, mean - dev, mean + dev,
    plt.fill_between(saved_time_array[tsav], fail_mean - fail_dev, fail_mean + fail_dev,
                     # facecolor=mycolor,
                     alpha=0.125)

    '''
    Compare
      1. Q(T) in the failed trial
      2. Q_inf := argmin_Q ( sum_i pi_i(Q) KL(Q_i || P_i) ) for given instance P
         <- this is obtaiend in the above computation
    '''
    # 1.
    emp_dist_df = results_df.loc[:, ['times', 'emp_dist']]
    fail_emp_dist_df = emp_dist_df[results_df['sim_nums'].isin(fail_sim_num_array)]
    fail_emp_dist_at_final_df = fail_emp_dist_df[fail_emp_dist_df['times'] == horizon - 1]  # Q(T) for each simulation

    # 2.
    print(f'{Q_inf=}')
    # import ipdb; ipdb.set_trace()

    # final. compute value of sum_k pi_k(Q) KL(Q_i||P_i)
    # 0. get setting of P
    means = gen_pe_setting(n_arms, setting)
    # true_means = torch.FloatTensor(means[None, None, :])
    true_means = torch.FloatTensor(means)

    # 1. load policy model
    policy_model = load_trained_policy_model(comp='H')
    policy_model.eval()

    policy_loss_inst = PolicyLoss(n_arms, arm_dist, policy_model, H_batch_tensor)  # todo: this H_batch_tensor should be fixed when doing OptNN-OptH

    # computation of sum_k pi_k(Q) KL(Q_k||P_k) for 1
    tmp_list = []
    for final_emp_dist_str in fail_emp_dist_at_final_df['emp_dist']:
        final_emp_dist = [float(q) for q in final_emp_dist_str.strip('[]').split(',')]
        final_emp_dist = torch.FloatTensor(final_emp_dist)
        neg_exponent_no_H = policy_loss_inst.compute_negative_exponent(true_means[None, :], final_emp_dist[None, :], no_H=True)
        neg_exponent_no_H = neg_exponent_no_H.to('cpu').detach().numpy().copy().squeeze()
        # print(f'{neg_exponent_no_H=}')
        tmp_list.append(neg_exponent_no_H)
    tmp_array = np.array(tmp_list)
    print(f'{np.mean(tmp_array)=}, {np.min(tmp_array)=}')

    # computation of sum_k pi_k(Q) KL(Q_k||P_k) for 2
    neg_exponent_no_H_of_Qinf = policy_loss_inst.compute_negative_exponent(true_means[None, :], Q_inf[None, :], no_H=True)
    print(f'{neg_exponent_no_H_of_Qinf=}')

plt.xlim(left=0.0)
plt.ylim(bottom=0.0)
plt.legend()
axes.set_xlabel(r'round $t$')
axes.set_ylabel(r'$ \max_{i\in[K]} | r_i(\mathbf{Q}(t)) - Q_i(t) |}$')
axes.grid(False)

fig_basename = get_pemab_fig_basename(setting, arm_dist, n_arms, noise_var)
save_fig_path = f'results/{model}/fig/{n_arms}_{arm_dist}/track_{fig_basename}.pdf'
print(f'saving to {save_fig_path}')
plt_save(save_fig_path)
plt.close()
