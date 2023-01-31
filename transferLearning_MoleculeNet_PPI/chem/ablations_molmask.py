import argparse

from loader import MoleculeDataset, MolCliqueDatasetWrapper
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model_mp_mask import GNN, GNN_mask, GNN_mp_mask
from clique import get_mol, brics_decomp, tree_decomp, get_clique_mol, get_smiles
import vocab
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil
import random
from collections import defaultdict
from copy import deepcopy

from tensorboardX import SummaryWriter

from rdkit import RDLogger     
RDLogger.DisableLog('rdApp.*')  

criterion = nn.CrossEntropyLoss()

def _ortho_constraint(device, prompt):
    return torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))

def _gen_clique_to_mol(clique_list, mol_to_clique):
    clique_to_mol = defaultdict(list)
    for mol in mol_to_clique:
        for clique in mol_to_clique[mol]:
            clique_to_mol[clique_list.index(clique)].append(mol)
    return clique_to_mol

def _get_training_molecules(train_loader):
    train_mol = []
    for data in train_loader:
        for d in data.to_data_list():
            train_mol.append(d.mol_index.item())
    return train_mol

def _extract_cliques(device, batch, mol_to_clique, clique_list):
    mol_idx = []
    clique_idx = []
    for i, d in enumerate(batch.to_data_list()):
        for clique in mol_to_clique[d.mol_index.item()].keys():
            mol_idx.append(i)
            clique_idx.append(clique_list.index(clique))

    mol_idx = torch.tensor(mol_idx).to(device)
    clique_idx = torch.tensor(clique_idx).to(device)

    return mol_idx, clique_idx

def train(args, kwargs, target, model_list, loader, optimizer, clique_list, mol_to_clique, device):
    model, atom_prompts = model_list

    model.train()
    atom_prompts.train()

    #for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        mol_idx, clique_idx = _extract_cliques(device, batch, mol_to_clique, clique_list)
        
        graph_rep = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mol_idx, clique_idx)

        if len(batch.y.shape) > 1:
            y = batch.y[:, target].flatten().to(torch.long)
        else:
            y = batch.y[target].flatten().to(torch.long)

        prompt_indices = [random.randint(0, kwargs['num_clusters'] - 1), random.randint(kwargs['num_clusters'], 2 * kwargs['num_clusters'] - 1)]
        prompts = atom_prompts(torch.tensor(prompt_indices, device=device))
        prompts = F.normalize(prompts, dim=-1)

        pred = torch.mm(graph_rep, prompts.T)
        loss = criterion(pred.double(), y)
        loss += float(kwargs['ortho_weight']) * _ortho_constraint(device, atom_prompts.weight)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

def eval(args, kwargs, target, model_list, loader, clique_list, mol_to_clique, device):
    model, atom_prompts = model_list

    model.eval()
    atom_prompts.train()

    y_true = []
    y_scores = []

    #for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        mol_idx, clique_idx = _extract_cliques(device, batch, mol_to_clique, clique_list)

        with torch.no_grad():
            graph_rep = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mol_idx, clique_idx)

            if len(batch.y.shape) > 1:
                y = batch.y[:, target].flatten().to(torch.long)
            else:
                y = batch.y[target].flatten().to(torch.long)

            prompts = atom_prompts.weight
            prompts = F.normalize(prompts, dim=1)
            prompts_0 = torch.index_select(prompts, 0, torch.tensor([i for i in range(kwargs['num_clusters'])], device=device))
            prompts_0 = prompts_0.mean(dim=0, keepdim=True)
            prompts_0 = F.normalize(prompts_0, dim=1)
            prompts_1 = torch.index_select(prompts, 0, torch.tensor([kwargs['num_clusters'] + i for i in range(kwargs['num_clusters'])], device=device))
            prompts_1 = prompts_1.mean(dim=0, keepdim=True)
            prompts_1 = F.normalize(prompts_1, dim=1)
            prompts = torch.cat((prompts_0, prompts_1), dim=0)

            pred = torch.mm(graph_rep, prompts.T)
            pred = F.softmax(pred, dim=-1)

            if device == 'cpu':
                y_true.extend(y.numpy())
                y_scores.extend(pred.detach().numpy())
            else:
                y_true.extend(y.cpu().numpy())
                y_scores.extend(pred.cpu().detach().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    return roc_auc_score(y_true, y_scores[:, 1])


def main(**kwargs):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_model_file', type=str, default = 'new_model_gin/masking.pth', help='filename to read the gnn model (if there is any)')
    parser.add_argument('--proj_head_file', type=str, default = 'new_model_gin/masking_atom_head.pth', help='filename to read the projection head weights')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    elif args.dataset == "hiv":
        num_tasks = 1
        target_list = ["HIV_active"]
    elif args.dataset == "muv":
        num_tasks = 17
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]
    elif args.dataset == "bace":
        num_tasks = 1
        target_list = ["Class"]
    elif args.dataset == "bbbp":
        num_tasks = 1
        target_list = ["p_np"]
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
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
    elif args.dataset == "clintox":
        num_tasks = 2
        target_list = ['CT_TOX', 'FDA_APPROVED']
    else:
        raise ValueError("Invalid dataset name.")

    print(args, kwargs)

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    def gen_cliques(smiles_data):
        mol_to_clique = {}
        clique_set = set()
        for i, m in enumerate(smiles_data):
            mol_to_clique[i] = {}
            mol = vocab.get_mol(m)
            cliques, edges = vocab.tree_decomp(mol)
            for c in cliques:
                cmol = vocab.get_clique_mol(mol, c)
                cs = vocab.get_smiles(cmol)
                clique_set.add(cs)
                if cs not in mol_to_clique[i]:
                    mol_to_clique[i][cs] = 1
                else:
                    mol_to_clique[i][cs] += 1
        return list(clique_set), mol_to_clique

    def filter_cliques(threshold, train_loader, clique_list, mol_to_clique, clique_to_mol):
        train_mol = _get_training_molecules(train_loader)

        fil_clique_list = []
        for i, d in enumerate(clique_list):
            if sum(mol in train_mol for mol in clique_to_mol[i]) <= threshold:
                fil_clique_list.append(d)

        tmol_to_clique = deepcopy(mol_to_clique)
        for mol in mol_to_clique:
            for clique in mol_to_clique[mol].keys():
                if clique in fil_clique_list:
                    del tmol_to_clique[mol][clique]

        mol_to_clique = deepcopy(tmol_to_clique)
        emp_mol = []
        for mol in tmol_to_clique:
            if len(tmol_to_clique[mol]) == 0:
                mol_to_clique[mol]['EMP'] = 1
                emp_mol.append(mol)

        clique_list = list(set(clique_list) - set(fil_clique_list))
        return emp_mol, clique_list, mol_to_clique

    #print(dataset)

    total_val_acc = []
    for target in range(len(target_list)):
        if args.split == "scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, task_idx=target, null_value=np.nan, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
            # print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, task_idx=target, null_value=np.nan, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed = args.seed)
            # print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, task_idx=target, null_value=np.nan, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
            # print("random scaffold")
        else:
            raise ValueError("Invalid split option.")

        if train_dataset is None:
            print("Task omitted!")
            total_val_acc.append(1.0)
            continue

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        full_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        clique_list, mol_to_clique = gen_cliques(smiles_list)
        clique_to_mol = _gen_clique_to_mol(clique_list, mol_to_clique)
        emp_mol, clique_list, mol_to_clique = filter_cliques(kwargs['threshold'], train_loader, clique_list, mol_to_clique, clique_to_mol)
        num_motifs = len(clique_list) + 1

        clique_dataset = MolCliqueDatasetWrapper(clique_list, num_motifs, args.num_workers)
        clique_loader = clique_dataset.get_data_loaders()

        #set up model
        gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
        model = GNN_mask(gnn)
        atom_prompts = torch.nn.Embedding(2 * kwargs['num_clusters'], 119).to(device)
        # atom_prompts = torch.nn.Linear(119, 2, bias=False).to(device)
        if not (args.gnn_model_file == "" or args.proj_head_file == ""):
            model.from_pretrained(args.gnn_model_file, args.proj_head_file)
            #model.load_state_dict(torch.load(args.gnn_model_file))
        else:
            raise ValueError("No pretrained file provided for GNN or projection head.")

        model.to(device)

        with torch.no_grad():
            motif_feats = []
            for c in clique_loader:
                c = c.to(device)
                emb = model.forward_cl(c.x, c.edge_index, c.edge_attr, c.batch)
                #embs = model(c)
                #emb = global_mean_pool(embs, c.batch)
                motif_feats.append(emb)

            motif_feats = torch.cat(motif_feats)

            clique_list.append("EMP")

            dummy_motif = torch.zeros((1, motif_feats.shape[1])).to(device)
            motif_feats = torch.cat((motif_feats, dummy_motif), dim=0)

            atom_list_0 = [0 for i in range(119)]
            atom_list_1 = [0 for i in range(119)]

            for batch in train_loader:
                for data in batch.to_data_list():
                    atom_types = batch.x.clone()
                    atom_types = atom_types[:, 0][atom_types[:, 0] != 120]
                    a, c = torch.unique(atom_types.to(torch.long), sorted=True, return_counts=True)
                    a = a.tolist()
                    c = c.tolist()
                    if len(batch.y.shape) > 1:
                        y = data.y[:, target].flatten().to(torch.long)
                    else:
                        y = data.y[target].flatten().to(torch.long)
                    if y.item() == 0:
                        for atom_idx in range(len(a)):
                            atom_list_0[a[atom_idx]] += float(c[atom_idx])
                    if y.item() == 1:
                        for atom_idx in range(len(a)):
                            atom_list_1[a[atom_idx]] += float(c[atom_idx])

            atom_list_0 = torch.tensor(atom_list_0)
            atom_list_1 = torch.tensor(atom_list_1)

            atom_feats = torch.vstack((atom_list_0, atom_list_1)).to(device)
            atom_feats = F.normalize(atom_feats, dim=-1)
            offset = torch.zeros(atom_feats.shape).to(device)
            torch.nn.init.constant_(offset, 1e-8)
            atom_feats += offset

            cluster_indices = [0 for i in range(kwargs['num_clusters'])]
            cluster_indices.extend([1 for i in range(kwargs['num_clusters'])])

            atom_feats = torch.index_select(atom_feats, 0, torch.tensor(cluster_indices, device=device))

        model = GNN_mp_mask(num_motifs, args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, 
                enc_dropout=kwargs['enc_dropout'], tfm_dropout=kwargs['tfm_dropout'], dec_dropout=kwargs['dec_dropout'],
                enc_ln=kwargs['enc_ln'], tfm_ln=kwargs['tfm_ln'], conc_ln=kwargs['conc_ln'], num_heads=kwargs['num_heads'], graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
        if not (args.gnn_model_file == "" or args.proj_head_file == ""):
            model.from_pretrained(args.gnn_model_file, args.proj_head_file)
            model.init_clique_emb(motif_feats)
            with torch.no_grad():
                atom_prompts.weight.data = torch.nn.Parameter(atom_feats)
        else:
            raise ValueError("No pretrained file provided for GNN or projection head.")

        model_list = [model, atom_prompts]

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        best_val_acc = -1
        ass_test_acc = -1
        avg_val_acc = []
        
        for epoch in range(1, args.epochs+1):
            #print("====epoch " + str(epoch))
            
            train(args, kwargs, target, model_list, train_loader, optimizer,  clique_list, mol_to_clique, device)

            #print("====Evaluation")
            if args.eval_train:
                train_acc = eval(args, kwargs, target, model_list, train_loader, clique_list, mol_to_clique, device)
            else:
            #    print("omit the training accuracy computation")
                train_acc = 0
            val_acc = eval(args, kwargs, target, model_list, val_loader, clique_list, mol_to_clique, device)
            test_acc = eval(args, kwargs, target, model_list, test_loader, clique_list, mol_to_clique, device)

            #print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

            avg_val_acc.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ass_test_acc = test_acc

            #print("")

        avg_val_acc = sum(avg_val_acc) / len(avg_val_acc)
        total_val_acc.append(avg_val_acc)
        print(ass_test_acc)
    return sum(total_val_acc) / len(total_val_acc)

if __name__ == "__main__":
    for _ in range(10):
        main(num_clusters=12, ortho_weight=4e-5, threshold=10, lr=0.001, enc_dropout=0.4, tfm_dropout=0.4, dec_dropout=0.4, enc_ln=False, tfm_ln=True, conc_ln=False, num_heads=1)
