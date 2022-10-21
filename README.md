## Fixed-Budget Best Arm Identification
Public code for a paper
["Minimax Optimal Algorithms for Fixed-Budget Best Arm Identification"](https://arxiv.org/abs/2206.04646)
in NeurIPS2022.

This repository includes codes for $R^{go}$-tracking.
All algorithms included are the followings:
```
- Uniform
- Successive Rejection (Audibert et al., 2010)
- $R^{go}$-tracking (Ours)
```

### Environment
The following provides three ways for constructing the environment.

(1). If you use `pyenv`, just run `setup.sh` (it uses venv and automatically install the required packages) 
To setup the environment, use `Python 3.7` or a more up-to-date version.
```
bash setup.sh
source venv/bin/activate
```

(2) If you are using `conda` environment:
``` 
conda create -n bandit python=3.8
conda activate bandit
pip install -r requirements.txt
mkdir -p results/pe-mab/fig
```

Then, run
```bash
pip install -r requirements.txt 
```

(3) Otherwise, install the following packages with `Python 3.n (n >= 7)` environment.
```
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- torch (only required for training $r$ from scratch)
```

For (2) and (3), for generating the required directory for replicating the results, need to run the following:

```
mkdir -p results/pe-mab/fig
mkdir -p fixed_budget_pjt/checkpoint/3_Ber
```

Finally, setup the following configurations:
```
** Write the project path in _LOCAL_PJT_DIR_PRE in fixed_budget_pjt/config.py
e.g.,
_LOCAL_PJT_DIR_PRE = '~/research/codes/bandit/fixed-budget-bai'
```


### How to Run (easy ver: use pretrained policy)
0. setup the environment
```
Setup the `PJT_PATH` in `fixed_budget_pjt/config.py`.
```

1. Run algorithm with setting 1, P = [0.5, 0.45, 0.3], see mab_setting.py for other settings of P
```
# For OptNN-Ho, the trained policy is uploaded in fixed_budget_pjt/checkpoint/3_Ber/0 and the path to pretrained model is specified in test_pe_mab.py

# small experiments (just one trial)
$ python pe_algorithms/mab/test_pe_mab.py -na 3 -ad Ber -s 1 -ns 1000 -ho 2000 -a OptNN-H
$ python pe_algorithms/mab/test_pe_mab.py -na 3 -ad Ber -s 1 -ns 1000 -ho 2000 -a Uniform
$ python pe_algorithms/mab/test_pe_mab.py -na 3 -ad Ber -s 1 -ns 1000 -ho 2000 -a SR
```

2. plot experimental results (for setting 1)
```
$ python plot_pe.py -na 3 -ad Ber -s 1 -ho 2000 -ns 1000 
```

Obtained results by running the above small experiments:
[(pdf of prob. of error)](readme_fig/poe2_na_3_setting_1_dist_Ber_nv_1.pdf)
[(pdf of tracking error)](readme_fig/track_na_3_setting_1_dist_Ber_nv_1.pdf)

### How to Run (train policy and then run trained policy)
0. setup the environment
```
Setup the `PJT_PATH` in `fixed_budget_pjt/config.py`.
```


1. train policy models $r_\theta$
```
$ python fixed_budget_pjt/train_model_batch.py -na 3 -ad Ber
```


2. setup the model path in pe_algorithms/test_pe_mab.py, see codes around Line 131 and write like
```
"""
if n_arms == 3 and args.arm_dist == 'Ber':
    model_path = Path(PJT_PATH / 'checkpoint' / f'{n_arms}_{arm_dist}' /
                      '1' / 'policy_0_33_valloss-0.98.pth' <=== REWRITE HERE DEPENDING ON THE TRAINING RESULTS
                      )
"""
You can find the trained file names in fixed_budget_pjt/checkout/3_Ber
```


3. run algorithm with setting 1, P = [0.5, 0.45, 0.3], see mab_setting.py for other settings of P
```
$ python pe_algorithms/mab/test_pe_mab.py -na 3 -ad Ber -s 1 -ns 1000 -ho 2000 -a OptNN-H
$ python pe_algorithms/mab/test_pe_mab.py -na 3 -ad Ber -s 1 -ns 1000 -ho 2000 -a Uniform
$ python pe_algorithms/mab/test_pe_mab.py -na 3 -ad Ber -s 1 -ns 1000 -ho 2000 -a SR
```

4. plot experimental results (for setting 1)
```
$ python plot_pe.py -na 3 -ad Ber -s 1 -ho 2000 -ns 1000 
```

### Contact
If you have any problems or questions when running codes, please send an email to the authors.

