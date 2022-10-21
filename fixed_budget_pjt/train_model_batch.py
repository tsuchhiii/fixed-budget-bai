import os
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import fixed_budget_pjt.config as config
from fixed_budget_pjt.config import ENV, MODEL, PJT_PATH
from model import PolicyNet
from utils import seed_everything, load_model, load_optimizer
from dataset import BatchTrueMeansDataset

from utils import H_batch_tensor, H_tensor, kl_func # kl_bernoulli_bernoulli, kl_normal_normal
from loss import PolicyLoss


# state_int2str = {0: 'pretrain', 1: 'train-complexity', 2: 'train-policy'}
def step2state(state: int):
    if state == 0:
        return 'pretrain'
    elif state % 2 == 1:
        return 'train-complexity'
    elif state % 2 == 0:
        return 'train-policy'
    else:
        raise NotImplementedError


# tmp
def compute_total_param_norm(policy_model):
    total_norm = 0.
    for p in policy_model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_total_grad_norm(policy_model, normalized=True):
    """ default: Normalized
    if normalized is True, then return || grad. of PolicyLoss(param) ||_F / || param ||_F
    else, return || grad. of PolicyLoss(param) ||_F
    """
    total_norm = 0.
    for p in policy_model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if normalized:
        # todo: should be refactored
        return total_norm / compute_total_param_norm(policy_model)
    else:
        return total_norm


def plot_total_norm_history(total_norm_list, epoch, save_name, accurate_total_norm_list=None):
    nrm = np.convolve(total_norm_list, np.ones(100) / 100, mode='same')
    plt.plot(np.arange(len(total_norm_list)), nrm, label="grad of params")
    plt.legend()
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.savefig(save_name)
    plt.clf()

    if accurate_total_norm_list is not None:
        plt.plot((len(total_norm_list) / (epoch+1)) * np.arange(1 + epoch), accurate_total_norm_list,
                 label="acc. grad of params")
        plt.legend()
        plt.xlim(left=0.0)
        plt.ylim(bottom=0.0)
        plt.savefig('acc_' + save_name)
        plt.clf()


def train_policy(policy_model, complexity_model, optimizer, dataloaders_dict, bandit_info, num_epochs, scheduler, policy_loss_inst, step):
    n_arms, arm_dist = bandit_info['n_arms'], bandit_info['arm_dist']
    best_loss = 0.0
    total_norm_list = []   # tmp
    accurate_total_norm_list = []   # tmp, accurate means we use many data for estimation

    state_str = step2state(step)

    # for accurate grad computation, (accurate = many data points for emp data)
    # tmp_true_means_dataset = BatchTrueMeansDataset(n_arms=n_arms, arm_dist=arm_dist, n_data=1, n_means_for_loss=10000)
    tmp_true_means_dataset = BatchTrueMeansDataset(n_arms=n_arms, arm_dist=arm_dist, n_data=1, n_means_for_loss=100)
    used_true_means = torch.FloatTensor(tmp_true_means_dataset)

    if state_str != 'pretrain':
        complexity_model.eval()

    for epoch in range(num_epochs):
        # TODO: START
        #  ONLY training data is changed for each time
        """
        START
        todo: make training data randomly for each epoch
        """
        true_means_dataset = BatchTrueMeansDataset(n_arms=n_arms, arm_dist=arm_dist,
                                                   # n_data=100000,
                                                   n_data=MODEL.NUM_SAMPLED_TRUE_MEANS,
                                                   # n_means_for_loss=20,
                                                   n_means_for_loss=MODEL.NUM_TRUE_MEANS_FOR_LOSS,
                                                   )
        val_ratio = 0.2
        train_size = int((1. - val_ratio) * len(true_means_dataset))
        val_size = len(true_means_dataset) - train_size
        train_true_means_dataset, val_true_means_dataset = torch.utils.data.random_split(true_means_dataset,
                                                                                         [train_size, val_size])

        batch_size = MODEL.BATCH_SIZE
        train_true_means_loader = DataLoader(
            train_true_means_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=MODEL.NUM_WORKER,
            pin_memory=True
        )
        # val_true_means_loader = DataLoader(
        #     val_true_means_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=MODEL.NUM_WORKER,
        #     pin_memory=True
        # )
        # dataloaders_dict = {'train': train_true_means_loader, 'val': val_true_means_loader}
        dataloaders_dict['train'] = train_true_means_loader
        """
        END OF TRAINING DATA
        """
        # TODO: END

        policy_model.to(ENV.DEVICE)
        if state_str != 'pretrain':
            complexity_model.to(ENV.DEVICE)

        for phase in ['train', 'val']:
            epoch_loss = 0.0

            # todo: acc. grad computation
            # if phase == 'train':
            if phase == 'val':  # todo: evaluation time evaluation for accurate norm tracking
                policy_model.eval()
                true_means_minmin_list, emp_means_minmin_list = \
                    policy_loss_inst.compute_critical_point(used_true_means, n_sampled_emp_means=1000000)
                    # policy_loss_inst.compute_critical_point(used_true_means, n_sampled_emp_means=10000)
                # policy_model.eval()
                loss = policy_loss_inst.compute_policy_loss(true_means_minmin_list, emp_means_minmin_list)
                policy_model.zero_grad()
                loss.backward()
                # accurate_total_norm = compute_total_grad_norm(policy_model)
                accurate_total_norm = compute_total_grad_norm(policy_model, normalized=False)
                accurate_total_norm_list.append(accurate_total_norm)
                print(f'{epoch=}, (normalized) {accurate_total_norm=}')

            true_means_loader = dataloaders_dict[phase]
            for true_item in tqdm(true_means_loader, leave=True):
                batch_true_means = true_item.to(ENV.DEVICE).float()
                policy_model.eval()  # for fast computation

                true_means_minmin_list, emp_means_minmin_list = \
                    policy_loss_inst.compute_critical_point(batch_true_means, n_sampled_emp_means=MODEL.NUM_EMP_MEANS)

                # (remark) batch here means n_sampled_true_means = 6
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        policy_model.train()

                    loss = policy_loss_inst.compute_policy_loss(true_means_minmin_list, emp_means_minmin_list)
                    """
                    loss = torch.tensor(0.).to(ENV.DEVICE)
                    for idx in range(len(true_means_min_list)):
                        loss += compute_policy_loss(
                            true_means_min_list[idx][None, :], emp_means_minmin_list[idx][None, :],
                            policy_model, complexity_model)[0][0]
                    """

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # todo: for now, compute gradient of norm
                        # total_norm = compute_total_grad_norm(policy_model)
                        total_norm = compute_total_grad_norm(policy_model, normalized=False)
                        # print(f'Gradient of parameters norm is {total_norm}')
                        total_norm_list.append(total_norm)

                    epoch_loss += loss.item()

            data_size = len(true_means_loader.dataset)  # todo: old
            # data_size = len(true_means_loader)  # todo: this should be correct....
            epoch_loss = epoch_loss / data_size  # todo: fix when batch changing

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.2f}')

            # if phase == 'train':
            if phase == 'val':  # TODO: valid time plotting
                save_name = f'grad_of_params_{n_arms}_{arm_dist}_epoch-{epoch}.png'
                plot_total_norm_history(total_norm_list, epoch, save_name, accurate_total_norm_list)

        if epoch_loss < best_loss:
            os.makedirs(MODEL.MODEL_SAVE_DIR_PATH, exist_ok=True)
            save_stem_name = f'{step}_{epoch:02}_valloss{epoch_loss:.2f}'
            # -- original jit -- <- not suited for parallel computation
            # traced = torch.jit.trace(policy_model.cpu(), torch.rand(1, n_arms))
            # traced.save(MODEL.MODEL_SAVE_DIR_PATH / f'policy_{save_stem_name}.pth')
            # -- for parallel --
            torch.save(policy_model.state_dict(), MODEL.MODEL_SAVE_DIR_PATH / f'policy-2_{save_stem_name}.pth')
            # -- optimizer --
            # torch.save(optimizer.state_dict(), MODEL.MODEL_SAVE_DIR_PATH / f'policy-opt_{save_stem_name}.pth')
            best_loss = epoch_loss

        scheduler.step()

    return policy_model


def get_args():
    parser = argparse.ArgumentParser(description='fixed-budget project')

    parser.add_argument('--n_arms', '-na', required=False, type=int, default=3,
                        help='# of arms')
    parser.add_argument('--arm_dist', '-ad', required=False, type=str, default='Ber',
                        choices=config.SETTING_CHOICES,
                        help='arm distribution')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args)

    seed = 42
    seed_everything(seed)

    n_arms = args.n_arms
    arm_dist = args.arm_dist

    MODEL.setup_model_dir_path(n_arms, arm_dist)
    MODEL.setup_num_sampled_true_means(n_arms)

    bandit_info = {
        'n_arms': n_arms,
        'arm_dist': arm_dist,
    }

    policy_model = PolicyNet(n_arms=n_arms)

    '''
    dataset
    '''
    true_means_dataset = BatchTrueMeansDataset(n_arms=n_arms, arm_dist=arm_dist,
                                               n_data=MODEL.NUM_SAMPLED_TRUE_MEANS,
                                               n_means_for_loss=MODEL.NUM_TRUE_MEANS_FOR_LOSS,
                                               )
    val_ratio = 0.2
    train_size = int((1. - val_ratio) * len(true_means_dataset))
    val_size = len(true_means_dataset) - train_size
    train_true_means_dataset, val_true_means_dataset = torch.utils.data.random_split(true_means_dataset, [train_size, val_size])

    batch_size = MODEL.BATCH_SIZE
    train_true_means_loader = DataLoader(
        train_true_means_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=MODEL.NUM_WORKER,
        pin_memory=True
    )
    val_true_means_loader = DataLoader(
        val_true_means_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=MODEL.NUM_WORKER,
        pin_memory=True
    )
    dataloaders_dict = {'train': train_true_means_loader, 'val': val_true_means_loader}

    '''
    train policy based on complexity measure to H1
    '''
    # todo for now
    policy_loss_inst = PolicyLoss(n_arms, arm_dist, policy_model, H_batch_tensor)
    # TODO: better to setup optimizer for each step or reuse?
    policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3, weight_decay=1e-7)  # this best now
    scheduler = torch.optim.lr_scheduler.MultiStepLR(policy_optimizer, milestones=[25, 40], gamma=0.1)
    policy_model = \
        train_policy(policy_model=policy_model, complexity_model=H_batch_tensor,  # H_tensor,
                     optimizer=policy_optimizer,
                     dataloaders_dict=dataloaders_dict, bandit_info=bandit_info, num_epochs=MODEL.NUM_EPOCHS,
                     scheduler=scheduler,
                     policy_loss_inst=policy_loss_inst,
                     step=0)

    # remark. minimal code is remained

if __name__ == '__main__':
    main()
