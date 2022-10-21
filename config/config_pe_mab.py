""" Configuration file for pure exploration in multi-armed bandits
"""

'''
Algorithm choices
'''
ALGO_CHOICES = [
    'Uniform',
    'SR',  # successive rejects
    'OptNN-H',  # optimal learningg based on trained neural network based on H (used for fixed conf setting)
    # 'OptNN-H-Lazy',  # lazy version of OptNN-H
    # 'OptNN-OptH',  # optimal learning based on trained neural network after H is trained
    # 'UCB1PE',  # UCB1 for pure exploration
    # 'UGapEb',  # V. Gabillon, M. Ghavamzadeh, A. Lazaric NeurIPS2012.
]

'''
Setting choices
'''
ARM_DIST_CHOICES = [
    'Ber',
    'Normal',
]

SETTING_CHOICES = [i for i in range(1, 7)]


