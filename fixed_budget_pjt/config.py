""" Configuration file for the fixed-budget project.
"""
import pathlib
import torch
import os


# _LOCAL_PJT_DIR = '/Users/tairatsuchiya/research/codes/bandit/fixed_budget_pjt'
_LOCAL_PJT_DIR_PRE = '/Users/tairatsuchiya/research/codes/fixed-budget-bai'
_LOCAL_PJT_DIR = _LOCAL_PJT_DIR_PRE + '/fixed_budget_pjt'
# PJT_DIR = '/opt/project/'   # local docker
_SERVER_PJT_DIR = '/home/tsuchiya/codes/bandit/fixed-budget-bai'

_cwd_str = os.getcwd()
print(_cwd_str)
if _cwd_str in _LOCAL_PJT_DIR:
    _LOCAL = True
elif _cwd_str in _SERVER_PJT_DIR:
    _LOCAL = False
else:
    print('setup path in fixed_budget_pjt/config.py correctly')
    raise NotImplementedError

if _LOCAL:
    print('THIS IS DEBUG MODE (expected to be run in local env)!\n' * 5)

if _LOCAL:
    PJT_DIR = _LOCAL_PJT_DIR
else:
    PJT_DIR = _SERVER_PJT_DIR

PJT_PATH = pathlib.Path(PJT_DIR)


class ENV:
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DEBUG = _LOCAL


# policy model
class MODEL:
    # NUM_EPOCHS = 50
    # NUM_EPOCHS = 30
    NUM_EPOCHS = 1000
    NUM_WORKER = 0 if _LOCAL else 1
    BATCH_SIZE = 32  # mini batch size
    COMP_BATCH_SIZE = 1024

    # config of data loader
    NUM_SAMPLED_TRUE_MEANS = None
    COMP_NUM_SAMPLED_TRUE_MEANS = None
    NUM_TRUE_MEANS_FOR_LOSS = None

    NUM_EMP_MEANS = None  # of sampled empirical means for each true means
    COMP_NUM_EMP_MEANS = None  # # of sampled empirical means for each true means

    MODEL_SAVE_DIR_PATH = None
    PRE_CHECKPOINT_PATH = PJT_PATH / 'checkpoint'

    '''
    for alternate learning
    '''
    NUM_SAMPLED_TRUE_MEANS_ALT = None
    NUM_TRUE_MEANS_FOR_LOSS_ALT = None

    @classmethod
    def setup_model_dir_path(cls, n_arms, setting):
        # now = datetime.datetime.now()
        _CHECKPOINT_PATH = PJT_PATH / 'checkpoint' / f'{n_arms}_{setting}'
        os.makedirs(_CHECKPOINT_PATH, exist_ok=True)
        tmp = len([p for p in _CHECKPOINT_PATH.iterdir() if p.is_dir()])
        cls.MODEL_SAVE_DIR_PATH = _CHECKPOINT_PATH / str(tmp)
        print('==> model save dir path is ', cls.MODEL_SAVE_DIR_PATH)

    @classmethod
    def setup_num_sampled_true_means(cls, n_arms):
        cls.NUM_SAMPLED_TRUE_MEANS = n_arms * 10000  # todo: for using different dataset
        cls.COMP_NUM_SAMPLED_TRUE_MEANS = n_arms * 10**6  # like pretraining?
        cls.NUM_TRUE_MEANS_FOR_LOSS = n_arms * 5   # true menas for each computation of inf
        cls.NUM_EMP_MEANS = 90   # emp_means_per_arm virtually
        cls.COMP_NUM_EMP_MEANS = 900   # emp_means_per_arm virtually

        # for alternate learning
        cls.NUM_SAMPLED_TRUE_MEANS_ALT = cls.NUM_SAMPLED_TRUE_MEANS // 2   # half of pretrain
        cls.NUM_TRUE_MEANS_FOR_LOSS_ALT = cls.NUM_TRUE_MEANS_FOR_LOSS * 2

    '''
    setting for pretrain complexity net
    '''
    PRETRAIN_COMPLEXITY_SAVE_DIR_PATH = PJT_PATH / 'checkpoint' / 'pretrain-complexity'
    PRETRAIN_COMPLEXITY_NUM_EPOCHS = 100


SETTING_CHOICES = [
    'Ber',  # only bernoulli for now
    'Normal',
]

ARM_DIST_CHOICES = [
    'Ber',  # only bernoulli for now
    'Normal',
]


