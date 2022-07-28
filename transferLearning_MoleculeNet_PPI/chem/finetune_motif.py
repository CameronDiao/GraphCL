import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from model_motif import GNN_M, GNN_M_graphpred
import clique
import vocab
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil
from collections import defaultdict
from copy import deepcopy

from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def extract_cliques(device, batch, mol_to_clique, clique_list):
    mol_idx = []
    clique_idx = []
    for i, d in enumerate(batch.to_data_list()):
        for clique in mol_to_clique[d.mol_index.item()].keys():
            mol_idx.append(i)
            clique_idx.append(clique_list.index(clique))

    mol_idx = torch.tensor(mol_idx).to(device)
    clique_idx = torch.tensor(clique_idx).to(device)

    #motif_samples = motif_embed(clique_idx).to(self.device)

    #return mol_idx, motif_samples
    return mol_idx, clique_idx

def train(args, model, device, loader, optimizer, extract_cliques, clique_list, mol_to_clique):
    model.train()

    #for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        mol_idx, clique_idx = extract_cliques(device, batch, mol_to_clique, clique_list)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mol_idx, clique_idx)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, clique_list, mol_to_clique):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        mol_idx, clique_idx = extract_cliques(device, batch, mol_to_clique, clique_list)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, mol_idx, clique_idx)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    #if len(roc_list) < y_true.shape[1]:
    #    print("Some target is missing!")
    #    print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

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


def _ortho_constraint(device, prompt):
    return torch.norm(torch.mm(prompt, prompt.T) - torch.eye(prompt.shape[0]).to(device))


def main(**kwargs):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
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
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
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
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

 
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    def gen_cliques(smiles_data):
        mol_to_clique = {}
        clique_set = set()
        if kwargs['vocab'] == 'mgssl':
            for i, m in enumerate(smiles_data):
                mol_to_clique[i] = {}
                mol = clique.get_mol(m)
                cliques, edges = clique.brics_decomp(mol)
                if len(edges) <= 1:
                    cliques, edges = clique.tree_decomp(mol)
                for c in cliques:
                    cmol = clique.get_clique_mol(mol, c)
                    cs = clique.get_smiles(cmol)
                    clique_set.add(cs)
                    if cs not in mol_to_clique[i]:
                        mol_to_clique[i][cs] = 1
                    else:
                        mol_to_clique[i][cs] += 1
        elif kwargs['vocab'] == 'junction':
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

    def _filter_cliques(threshold, train_loader, clique_list, mol_to_clique, clique_to_mol):
        train_mol = _get_training_molecules(train_loader)

        fil_clique_list = []
        for i, d in enumerate(clique_list):
            if sum(mol in train_mol for mol in clique_to_mol[i]) <= threshold:
                fil_clique_list.append(d)

        tmol_to_clique = deepcopy(mol_to_clique)
        for mol in mol_to_clique:
            for clique in mol_to_clique[mol].keys():
                if clique in fil_clique_list:
                    if kwargs['init'] == 'uniform':
                        tmol_to_clique[mol]['EMP'] = 1
                    del tmol_to_clique[mol][clique]

        mol_to_clique = deepcopy(tmol_to_clique)
        emp_mol = []
        for mol in tmol_to_clique:
            if kwargs['init'] == 'uniform':
                if all('EMP' in clique for clique in tmol_to_clique[mol].keys()):
                    emp_mol.append(mol)
            elif kwargs['init'] == 'zeros':
                if len(tmol_to_clique[mol]) == 0:
                    mol_to_clique[mol]['EMP'] = 1
                    emp_mol.append(mol)

        clique_list = list(set(clique_list) - set(fil_clique_list))
        return emp_mol, clique_list, mol_to_clique

    clique_list, mol_to_clique = gen_cliques(smiles_list)
    clique_to_mol = _gen_clique_to_mol(clique_list, mol_to_clique)
    emp_mol, clique_list, mol_to_clique = filter_cliques(kwargs['threshold'], num_tasks, train_loader, clique_list, mol_to_clique, clique_to_mol)
    num_motifs = len(clique_list) + 1
    #num_motifs = len(clique_list)
    print("Finished generating motif vocabulary")

    clique_dataset = MolCliqueDatasetWrapper(clique_list, args.batch_size, args.num_workers)
    clique_loader = clique_dataset.get_data_loaders()

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    with torch.no_grad():

        motif_feats = []
        for c in clique_loader:
            c = c.to(self.device)
            __, emb = model(c)
            motif_feats.append(emb)

        motif_feats = torch.cat(motif_feats)

        clique_list.append("EMP")

        label_feats = []
        labels = []
        for d in train_loader:
            d = d.to(self.device)
            feat_emb, out_emb = model(d)
            label_feats.append(out_emb)
            labels.append(d.y)

        label_feats = torch.cat(label_feats)
        labels = torch.cat(labels)

        linit0 = torch.mean(label_feats[torch.nonzero(labels == 0)[:, 0]], dim=0)
        linit1 = torch.mean(label_feats[torch.nonzero(labels == 1)[:, 0]], dim=0)

        label_feats = torch.vstack((linit0, linit1)).to(self.device)

        dummy_motif = torch.zeros((1, motif_feats.shape[1])).to(self.device)
        if kwargs['init'] == 'uniform':
            nn.init.xavier_uniform_(dummy_motif)
        motif_feats = torch.cat((motif_feats, dummy_motif), dim=0)

    model = GNN_M_graphpred(num_motifs, args.num_layer, args.emb_dim, num_tasks, JK = args.JK,
            drop_ratio = args.dropout_ratio, enc_dropout=kwargs['enc_dropout'], tfm_dropout=kwargs['tfm_dropout'], dec_dropout=kwargs['dec_dropout'],
            enc_ln=kwargs['enc_ln'], tfm_ln=kwargs['tfm_ln'], conc_ln=kwargs['conc_ln'], num_heads=kwargs['num_heads'], graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    model.init_clique_emb(motif_feats)

    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        
    layer_list = []
    for name, param in model.named_parameters():
        if 'clique' in name or 'motif' in name or 'conc' in name or 'graph_pred' in name:
            layer_list.append(name)

    pred_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
    model_param_group.append({"params": pred_params, "lr": kwargs['lr']})
    #model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    #print(optimizer)

    #train_acc_list = []
    #val_acc_list = []
    #test_acc_list = []

    best_val_acc = -1
    ass_test_acc = -1
    avg_val_acc = []
    
    for epoch in range(1, args.epochs+1):
        #print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer, extract_cliques, clique_list, mol_to_clique)

        #print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, clique_list, mol_to_clique)
        else:
            #print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader, clique_list, mol_to_clique)
        test_acc = eval(args, model, device, test_loader, clique_list, mol_to_clique)

        avg_val_acc.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ass_test_acc = test_acc

        #print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        #val_acc_list.append(val_acc)
        #test_acc_list.append(test_acc)
        #train_acc_list.append(train_acc)

        #print("")

    avg_val_acc = sum(avg_val_acc) / len(avg_val_acc)
    print("val: %f, test: %f" %(avg_val_acc, ass_test_acc))

    #with open('result.log', 'a+') as f:
    #    f.write(args.dataset + ' ' + str(args.runseed) + ' ' + str(np.array(test_acc_list)[-1]))
    #    f.write('\n')

    return avg_val_acc, ass_test_acc

if __name__ == "__main__":
    main(threshold=10, lr=0.001, enc_dropout=0.3, tfm_dropout=0., dec_dropout=0., enc_ln=True, tfm_ln=False, conc_ln=False, num_heads=4)
