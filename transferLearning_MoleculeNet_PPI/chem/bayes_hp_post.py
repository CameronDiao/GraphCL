from hp_finetune_post import FineTune
from dataset.dataset_test import MolTestDatasetWrapper

import pickle

import yaml
import numpy as np
from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.early_stop import no_progress_loss

import random
import torch

use_ln = [False, True]
activations = ['relu', 'softplus']

hp_space = {'threshold': hp.quniform('threshold', 0, 100, 10), 
            'enc_dropout': hp.quniform('enc_dropout', 0.0, 0.5, 0.1),
            'tfm_dropout': hp.quniform('tfm_dropout', 0.0, 0.5, 0.1),
            'dec_dropout': hp.quniform('dec_dropout', 0.0, 0.5, 0.1),
            'enc_ln': hp.choice('enc_ln', use_ln), 
            'tfm_ln': hp.choice('tfm_ln', use_ln),
            'conc_ln': hp.choice('conc_ln', use_ln),
            'n_heads': hp.quniform('n_heads', 2, 4, 2),
            'ortho_weight': hp.quniform('ortho_weight', 0, 5e-5, 5e-6)}

config = yaml.load(open("hp_config.yaml", "r"), Loader=yaml.FullLoader)


def objective(params):
    config['threshold'] = int(params['threshold'])
    config['model']['enc_dropout'] = float(params['enc_dropout'])
    config['model']['tfm_dropout'] = float(params['tfm_dropout'])
    config['model']['dec_dropout'] = float(params['dec_dropout'])
    config['model']['enc_ln'] = use_ln[int(params['enc_ln'])]
    config['model']['tfm_ln'] = use_ln[int(params['tfm_ln'])]
    config['model']['conc_ln'] = use_ln[int(params['conc_ln'])]
    config['model']['n_heads'] = int(params['n_heads'])
    config['ortho_weight'] = float(params['ortho_weight'])

    print(config)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/bace/bace.csv'
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions", "Endocrine disorders",
            "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders", "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]

    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/freesolv/freesolv.csv'
        target_list = ["expt"]

    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]

    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0",
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    elif config["task_name"] == 'qm9':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = 'data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    res = []

    for __ in range(2):
        for target in target_list:
            torch.cuda.empty_cache()
            config['dataset']['target'] = target
            dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

            fine_tune = FineTune(dataset, config)
            fine_tune.train()

            if config['dataset']['task'] == 'classification':
                res.append(-fine_tune.roc_auc)
            if config['dataset']['task'] == 'regression':
                if config['task_name'] in ['qm7', 'qm8', 'qm9']:
                    res.append(fine_tune.mae)
                else:
                    res.append(fine_tune.rmse)
    ret = sum(res) / len(res)

    print(ret)
    return ret

trials = Trials()

save_file_loc = './ckpt/trials' + config['task_name'] + '.pkl'

print(save_file_loc)

best = fmin(objective, hp_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=1000) 

print(best)
space_eval(hp_space, best)


pickle.dump(trials, open(save_file_loc, 'wb'))
