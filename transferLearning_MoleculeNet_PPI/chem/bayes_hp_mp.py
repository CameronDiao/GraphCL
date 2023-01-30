from finetune_motif import main

import pickle

import yaml
import numpy as np
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

import random
import torch

dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
n_heads = [2, 4, 10]

device = torch.device('cuda')
params = {}

for d in dropout:
    for n in n_heads:
        params['lr'] = 0.001
        params['threshold'] = 10
        params['enc_dropout'] = 0.4
        params['tfm_dropout'] = 0.4
        params['dec_dropout'] = 0.4
        params['enc_ln'] = False
        params['tfm_ln'] = True
        params['conc_ln'] = False
        
        params['vocab'] = 'mgssl'
        params['init'] = 'zeros'
        params['num_heads'] = int(n)

        res = []
        print('sider mask: {}'.format(params))

        for __ in range(2):
            val_acc = main(**params)
            res.append(-val_acc)

        ret = sum(res) / len(res)
        print(ret)
