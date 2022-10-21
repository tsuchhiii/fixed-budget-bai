#!/usr/bin/env bash

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# for linux server
# pip install numpy pandas matplotlib seaborn tqdm
# pip install torch torchvision torchaudio

# Python
export PYTHONIOENCODING=utf8
export PYTHONENCODING=utf8
# export PYTHONPATH=.
export PYTHONPATH="."

# make empty folders
mkdir -p results/pe-mab/fig

# setup for fixed-budget project
mkdir -p fixed_budget_pjt/checkpoint/3_Ber


