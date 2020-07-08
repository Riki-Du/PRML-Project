#%%
import os
import sys
from rdkit import Chem
from rdkit import RDPaths
import numpy as np
import csv
import torch
from functools import partial
import dgl
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    print('use GPU')
    device='cuda'
else:
    print('use CPU')
    device='cpu'
 
from dgllife.model import GCNPredictor,MPNNPredictor,AttentiveFPPredictor
from dgllife.utils import smiles_to_complete_graph,mol_to_complete_graph,smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer,WeaveAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer,WeaveEdgeFeaturizer,BaseBondFeaturizer
from torch import nn
 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss



# get a path
def GetPath(file):
    path = sys.path[0]
    path = os.path.normpath(path)
    return os.path.join(path, file)

# open a csv
def OpenCSV(file):
    file = os.path.normpath(file)
    file = GetPath(file)
    f = open(file)
    f_csv = csv.reader(f)
    return f_csv

# get train / test path everytime
def train_test_path(num=10):
    train_path = ""
    test_path = ""
    dev_path = ""
    if num == 10:
        train_path = "data/train.csv"
        test_path = "data/test.csv"
    else:
    # walk through fold 0-9
        train_path = "data/train_cv/fold_" + str(num) + "/train.csv"
        test_path = "data/train_cv/fold_" + str(num) + "/test.csv"
        dev_path = "data/train_cv/fold_" + str(num) + "/dev.csv"

    return train_path, test_path , dev_path

def read_from_rdkit(num, choice):
    train_path, test_path, dev_path = train_test_path(num)
    path = train_path
    if choice == 1:
        path = test_path
    elif choice == 2:
        path = dev_path
    f_csv = OpenCSV(path)
    SMILES_list = []
    if num == 10:
        # id,smiles,activity
        if choice == 0:
            SMILES_list = [(row[1],row[2]) for row in f_csv]
        else:
            SMILES_list = [(row[1]) for row in f_csv]
    else:
        # smiles,activity
        SMILES_list = [(row[0],row[1]) for row in f_csv]
    # SMILES = SMILES_list[1]
    # print("Test for rdkit!")
    # print(SMILES)
    SMILES_list = SMILES_list[1:]
    sm = [row[0] for row in SMILES_list]
    labels = [row[1] for row in SMILES_list]
    return sm, labels

def data_read(num): 
    print("Loading dataset ",num)
    trainmols, train_y = read_from_rdkit(num,0)
    testmols, test_y = read_from_rdkit(num,1)
    train_g = [smiles_to_bigraph(m, add_self_loop=False, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer) for m in trainmols]
    train_y = np.array(train_y, dtype=np.int64)
    print("Training set ",len(train_g))
    
    test_g = [smiles_to_bigraph(m, add_self_loop=False, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer) for m in testmols]
    test_y = np.array(test_y, dtype=np.int64)
    print("Test set",len(test_g))
    print("Data loaded.")

    return train_g, train_y, test_g, test_y

def collate(sample):
    graphs, labels = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)


atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()

def featurize_edges(mol, add_self_loop=False):
    feats = []
    num_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())
    distance_matrix = Chem.GetDistanceMatrix(mol)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j or add_self_loop:
                feats.append(float(distance_matrix[i, j]))
    return {'e': torch.tensor(feats).reshape(-1, 1).float()}

edge_featurizer = partial(featurize_edges, add_self_loop=False)
# check feature size
n_feats = atom_featurizer.feat_size()
e_feats = bond_featurizer.feat_size()
print(n_feats, e_feats)
# bond_featurizer = BaseBondFeaturizer({'e': lambda bond: [0 for _ in range(10)]})



m = 'Oc1c(I)cc(Cl)c2cccnc12'
m = 'CCO'
mol = Chem.MolFromSmiles(m)
num_bonds = mol.GetNumBonds()
# print(mol,num_bonds)
smiles_to_bigraph(m, add_self_loop=False, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer)
print(bond_featurizer(mol)['e'].shape)
# 二分类任务
ncls = 2


#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc


def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s


def train_one(num, model, loss_fn, optimizer, show=False,show_acc=False,EDGE=False):
    train_g, train_y, test_g, test_y = data_read(num)
    # print(train_g[0].ndata,train_g[0].edata)
    train_data = list(zip(train_g, train_y))
    train_loader = DataLoader(train_data, batch_size=128,shuffle=True,collate_fn=collate, drop_last=False)
    test_data = list(zip(test_g, test_y))
    test_loader = DataLoader(test_data, batch_size=128,shuffle=True,collate_fn=collate, drop_last=False)
    model.train()

    # 训练过程
    print("Training")
    epoch_losses = []
    epoch_roc_accuracies = []
    epoch_prc_accuracies = []
    test_losses = []
    test_roc_accuracies = []
    test_prc_accuracies = []
    e = 201
    for epoch in range(1,e):
        # train
        model.train()
        epoch_loss = 0
        test_loss = 0
        epoch_roc_auc = 0
        test_roc_auc = 0
        epoch_prc_auc = 0
        test_prc_auc = 0

        t = 0
        epoch_tot_pos_ps = []
        for i, (bg, labels) in enumerate(train_loader):
            labels = labels.to(device)
            atom_feats = bg.ndata.pop('h').to(device)
            atom_feats, labels = atom_feats.to(device), labels.to(device)
            if EDGE:
                edge_feats = bg.edata.pop('e').to(device)
                edge_feats = edge_feats.to(device)
            if EDGE:
                pred = model(bg, atom_feats, edge_feats)
            else:
                pred = model(bg, atom_feats)
            # print(pred)

            # 损失函数回传
            # pred = pred.reshape(-1)
            # print(pred.shape,labels.shape)
            # loss = loss_fn(pred, labels)
            loss = loss_fn(pred, labels.float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

            # 准确率
            pred_cls = pred.detach().to('cpu').numpy()
            pred_cls = softmax(pred_cls)
            # print(pred_cls)
            true_label = labels.to('cpu').numpy()
            # print(true_label)
            # print(pred_y)
            tot_pos_ps = [pred_cls[i] if true_label[i]==1 else 1-pred_cls[i] for i in range(len(true_label))]
            # tot_pos_ps = [pred_cls[i][true_label[i]] for i in range(len(true_label))]
            epoch_tot_pos_ps.extend(tot_pos_ps)
            t = i

        epoch_loss /= (t + 1)
        roc_auc = roc_auc_score(train_y, epoch_tot_pos_ps)
        p,r,thr = precision_recall_curve(train_y, epoch_tot_pos_ps)
        prc_auc = auc(r,p)

        epoch_roc_auc = roc_auc
        epoch_prc_auc = prc_auc

        # evaluate
        model.eval()
        test_tot_pos_ps = []
        t = 0
        for i, (bg, labels) in enumerate(test_loader):
            with torch.no_grad():
                labels = labels.to(device)
                atom_feats = bg.ndata.pop('h').to(device)
                atom_feats, labels = atom_feats.to(device), labels.to(device)
                if EDGE:
                    edge_feats = bg.edata.pop('e').to(device)
                    # print(edge_feats.shape)
                    edge_feats = edge_feats.to(device)
                if EDGE:
                    pred = model(bg, atom_feats, edge_feats)
                else:
                    pred = model(bg, atom_feats)
                # print(pred)

                # 损失函数回传
                loss = loss_fn(pred, labels)
                # loss = loss_fn(pred, labels.float()).mean()
                test_loss += loss.item()

                # 准确率
                pred_cls = pred.to('cpu').numpy()
                pred_cls = softmax(pred_cls)
                # print(pred_cls)
                true_label = labels.to('cpu').numpy()
                tot_pos_ps = [pred_cls[i] if true_label[i]==1 else 1-pred_cls[i] for i in range(len(true_label))]
                # tot_pos_ps = [pred_cls[i][true_label[i]] for i in range(len(true_label))]
                test_tot_pos_ps.extend(tot_pos_ps)
                t = i

        test_loss /= (t + 1)
        roc_auc = roc_auc_score(test_y, test_tot_pos_ps)
        p,r,thr = precision_recall_curve(test_y, test_tot_pos_ps)
        prc_auc = auc(r,p)

        test_roc_auc = roc_auc
        test_prc_auc = prc_auc

        if epoch % 20 == 0 and show_acc:
            print(f"epoch: {epoch}")
            print(f"Training loss: {epoch_loss:.3f}, roc_auc: {epoch_roc_auc:.3f},  prc_auc: {epoch_prc_auc:.3f}")
            print(f"Test loss: {test_loss:.3f}, roc_auc: {test_roc_auc:.3f},  prc_auc: {test_prc_auc:.3f}")
        epoch_roc_accuracies.append(epoch_roc_auc)
        epoch_prc_accuracies.append(epoch_prc_auc)
        epoch_losses.append(epoch_loss)
        test_roc_accuracies.append(test_roc_auc)
        test_prc_accuracies.append(test_prc_auc)
        test_losses.append(test_loss)

    # 显示出来
    if show:
        plt.subplot(211)
        # plt.style.use('ggplot')
        plt.plot([i for i in range(1, e)], epoch_losses, c='b', alpha=0.6, label='train_loss')
        plt.legend()
        plt.plot([i for i in range(1, e)], test_losses, c='r', alpha=0.6, label='test_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.subplot(212)
        # plt.style.use('ggplot')
        plt.plot([i for i in range(1, e)], epoch_roc_accuracies, c='b', alpha=0.6, label='training_roc_auc')
        plt.legend()
        plt.plot([i for i in range(1, e)], test_roc_accuracies, c='r', alpha=0.6, label='test_roc_auc')
        plt.legend()
        plt.plot([i for i in range(1, e)], epoch_prc_accuracies, c='g', alpha=0.6, label='traininig_prc_auc')
        plt.legend()
        plt.plot([i for i in range(1, e)], test_prc_accuracies, c='c', alpha=0.6, label='test_prc_auc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.show()

    train_roc_auc = epoch_roc_accuracies[-1]
    test_roc_auc = test_roc_accuracies[-1]
    train_prc_auc = epoch_prc_accuracies[-1]
    test_prc_auc = test_prc_accuracies[-1]
    print("Dataset ",num)
    print(f"Train set, roc_auc: {train_roc_auc:.3f},  prc_auc: {train_prc_auc:.3f}")
    print(f"Test set, roc_auc: {test_roc_auc:.3f},  prc_auc: {test_prc_auc:.3f}")
    return train_roc_auc,test_roc_auc,train_prc_auc,test_prc_auc
    


#%%
# define GCN NET with 2 GCN layers
model = GCNPredictor(in_feats=n_feats,
                    hidden_feats=[60,20],
                    n_tasks=ncls,
                    classifier_hidden_feats=10,
                    dropout=[0.5,0.5],)
model = model.to(device)
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1E-2)

train_roc_auc,test_roc_auc,train_prc_auc,test_prc_auc = train_one(0, model, loss_fn, optimizer,show_acc=True)

#%%
# attentive FP
model = AttentiveFPPredictor(node_feat_size=n_feats,
                            edge_feat_size=e_feats,
                            num_layers=2,
                            num_timesteps=2,
                            n_tasks=1,
                            dropout=0.2)
model = model.to(device)
# loss_fn = nn.MSELoss(reduction='none')
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (-2.5), weight_decay=10 ** (-5.0),)

train_roc_auc,test_roc_auc,train_prc_auc,test_prc_auc = train_one(0, model, loss_fn, optimizer,show_acc=True,EDGE=True)

#%%
model = MPNNPredictor(
                 node_in_feats = n_feats,
                 edge_in_feats = e_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=2,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3)
model = model.to(device)
# loss_fn = nn.MSELoss(reduction='none')
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=10 ** (-2.5), weight_decay=10 ** (-5.0),)

train_roc_auc,test_roc_auc,train_prc_auc,test_prc_auc = train_one(0, model, loss_fn, optimizer,show_acc=True,EDGE=True)

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


model.eval()
pred_y = []
test, y = collate(test_data)
atom_feats = test.ndata.pop('h').to(device)
pred_y = model(test, atom_feats)

pred_y = pred_y.detach().to('cpu').numpy()
pred_y = softmax(pred_y)
tot_pos_ps = [pred_y[i][true_label[i]] for i in range(len(true_label))]
roc_auc = roc_auc_score(true_label,tot_pos_ps)
p,r,thr = precision_recall_curve(true_label,tot_pos_ps)
prc_auc = auc(r,p)
print(test_y)
print(np.sum(test_y))
print(accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))

# %%
