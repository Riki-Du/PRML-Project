#%%
import os
import sys
from rdkit import Chem
from rdkit import RDPaths
import numpy as np
import csv
import torch
import dgl
if torch.cuda.is_available():
    print('use GPU')
    device='cuda'
else:
    print('use CPU')
    device='cpu'
 
from dgllife.model import GCNPredictor
# from dgl.data.chem.utils import mol_to_graph
# from dgl.data.chem.utils import mol_to_complete_graph
from dgllife.utils import smiles_to_complete_graph
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer
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
    trainmols, train_y = read_from_rdkit(num,0)
    testmols, test_y = read_from_rdkit(num,1)
    train_g = [smiles_to_complete_graph(m, add_self_loop=False, node_featurizer=atom_featurizer) for m in trainmols]
    train_y = np.array(train_y, dtype=np.int64)
    print("Training set ",len(train_g))
    
    test_g = [smiles_to_complete_graph(m, add_self_loop=False, node_featurizer=atom_featurizer) for m in testmols]
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
# check feature size
n_feats = atom_featurizer.feat_size('h')
print(n_feats)
# 二分类任务
ncls = 2

#%%
# define GCN NET with 2 GCN layers
gcn_net = GCNPredictor(in_feats=n_feats,
                    hidden_feats=[60,20],
                    n_tasks=ncls,
                    classifier_hidden_feats=10,
                    dropout=[0.5,0.5],)
gcn_net = gcn_net.to(device)
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.Adam(gcn_net.parameters(), lr=1E-3)

train_g, train_y, test_g, test_y = data_read(0)
train_data = list(zip(train_g, train_y))
# import numpy as np
# np.set_printoptions(threshold=np.inf)
# print(train_y)
train_loader = DataLoader(train_data, batch_size=128,shuffle=True,collate_fn=collate, drop_last=False)
test_data = list(zip(test_g, test_y))
test_loader = DataLoader(test_data, batch_size=128,shuffle=True,collate_fn=collate, drop_last=False)

gcn_net.train()

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
epoch_losses = []
epoch_accuracies = []
test_losses = []
test_accuracies = []
e = 201

def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s

for epoch in range(1,e):
    # train
    gcn_net.train()
    epoch_loss = 0
    test_loss = 0
    epoch_acc = 0
    test_acc = 0
    t = 0

    epoch_tot_pos_ps = []
    for i, (bg, labels) in enumerate(train_loader):
        labels = labels.to(device)
        atom_feats = bg.ndata.pop('h').to(device)
        atom_feats, labels = atom_feats.to(device), labels.to(device)
        pred = gcn_net(bg, atom_feats)
        # print(pred)

        # 损失函数回传
        loss = loss_fn(pred, labels)
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
        tot_pos_ps = [pred_cls[i][true_label[i]] for i in range(len(true_label))]
        epoch_tot_pos_ps.extend(tot_pos_ps)

    epoch_loss /= (t + 1)
    roc_auc = roc_auc_score(train_y, epoch_tot_pos_ps)
    p,r,thr = precision_recall_curve(train_y, epoch_tot_pos_ps)
    prc_auc = auc(r,p)
    epoch_acc = prc_auc

    # evaluate
    gcn_net.eval()
    pred_y = []
    test, y = collate(test_data)
    atom_feats = test.ndata.pop('h').to(device)
    pred_y = gcn_net(test, atom_feats)

    # test loss
    test_loss = loss_fn(pred_y, y.to(device))
    true_label = y
    pred_y = pred_y.detach().to('cpu').numpy()
    pred_y = softmax(pred_y)
    tot_pos_ps = [pred_y[i][true_label[i]] for i in range(len(true_label))]
    roc_auc = roc_auc_score(true_label,tot_pos_ps)
    p,r,thr = precision_recall_curve(true_label,tot_pos_ps)
    prc_auc = auc(r,p)
    # print(prc_auc)
    test_acc = prc_auc

    # print(pred_y)
    # print(np.sum(pred_y))
    # print(test_y)
    # print(np.sum(test_y))
    # print(accuracy_score(test_y, pred_y))
    # test_acc = accuracy_score(test_y, pred_y)


    if epoch % 20 == 0:
        print(f"epoch: {epoch}")
        print(f"Training loss: {epoch_loss:.3f}, acc: {epoch_acc:.3f}")
        print(f"Test loss: {test_loss:.3f}, acc: {test_acc:.3f}")
    epoch_accuracies.append(epoch_acc)
    epoch_losses.append(epoch_loss)
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)

#%%

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot([i for i in range(1, e)], epoch_losses, c='b', alpha=0.6, label='loss')
plt.legend()
plt.plot([i for i in range(1, e)], epoch_accuracies, c='r', alpha=0.6, label='acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()
plt.clf()

plt.plot([i for i in range(1, 201)], test_losses, c='b', alpha=0.6, label='loss')
plt.legend()
plt.plot([i for i in range(1, 201)], test_accuracies, c='r', alpha=0.6, label='acc')
plt.legend()
plt.xlabel('test')
plt.ylabel('loss/acc')
plt.show()

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


gcn_net.eval()
pred_y = []
test, y = collate(test_data)
atom_feats = test.ndata.pop('h').to(device)
pred_y = gcn_net(test, atom_feats)

pred_y = pred_y.argmax(-1).detach().to('cpu').numpy()
print(pred_y)
print(np.sum(pred_y))
print(test_y)
print(np.sum(test_y))
print(accuracy_score(test_y, pred_y))
print(classification_report(test_y, pred_y))

# %%
