#%%

from collections import namedtuple
import dgl
from dgl import DGLGraph
import dgl.function as fn
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
 
from rdkit import Chem
from rdkit.Chem import RDConfig
 
import os
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
 
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
MAX_ATOMNUM =60
BOND_FDIM = 5 
MAX_NB = 10


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        print("None!")
        return None
    Chem.Kekulize(mol)
    return mol

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
    + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
    + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
    + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
    + [atom.GetIsAromatic()]))


def mol2dgl_single(mols):
    """
    inputs
      mols: a list of molecules
    outputs
      cand_graphs: a list of dgl graphs
    """
    cand_graphs = []
    for mol in mols:
        n_atoms = mol.GetNumAtoms()
        g = DGLGraph() 
        node_feats = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        node_feats.append(atom_features(atom))
        g.add_nodes(n_atoms)
        bond_src = []
        bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        g.add_edges(bond_src, bond_dst)

        g.ndata['h'] = torch.Tensor([a.tolist() for a in node_feats])
        cand_graphs.append(g)
    return cand_graphs

 
# Note that during graph decoding they don't predict stereochemistry-related
# characteristics (i.e. Chiral Atoms, E-Z, Cis-Trans).  Instead, they decode
# the 2-D graph first, then enumerate all possible 3-D forms and find the
# one with highest score.
'''
def atom_features(atom):
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()]))
'''
def atom_features(atom):
    return (onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()])
 
def bond_features(bond):
    bt = bond.GetBondType()
    return (torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]))
 
def mol2dgl_single(mols):
    cand_graphs = []
    n_nodes = 0
    n_edges = 0
    bond_x = []
 
    for mol in mols:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()
        g = DGLGraph()        
        nodeF = []
        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            nodeF.append(atom_features(atom))
        g.add_nodes(n_atoms)
 
        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            features = bond_features(bond)
 
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)
        g.add_edges(bond_src, bond_dst)
        g.ndata['h'] = torch.Tensor(nodeF)
        cand_graphs.append(g)
    return cand_graphs

msg = fn.copy_src(src="h", out="m")
print("Initialization done.")

# 定义 GCN 模型
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
    
    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}
    
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
    
    def forward(self, g, feature):
        g.ndata['h'] = feature
        # g.update_all(msg,reduce)
        # collect features from source nodes and aggregate them in destination nodes
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
        # multiply source node features with edge weights and aggregate them in destination nodes
        # g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.max('m', 'h_max'))
        g.apply_nodes(func=self.apply_mod)
        h =  g.ndata.pop('h')
        # print(h.shape)
        return h
    
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([GCN(in_dim, hidden_dim, F.relu),
                                    GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)
    def forward(self, g):
        h = g.ndata['h']
        # print(h.shape)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        # print(h.shape)
        hg = dgl.mean_nodes(g, 'h')
        # print(hg.shape)
        return self.classify(hg)
print("GCN model done.")



#%%
from read_rdkit import read_from_rdkit

def data_reader():
    trainset = read_from_rdkit(num=10)
    # print(trainset[0:10])
    mols = []
    for (sm, label) in trainset:
        mol = get_mol(sm)
        # print(mol)
        if mol is not None:
            mols.append((mol, label))
    mol = [row[0] for row in mols]
    trainset = mol2dgl_single(mol)
    trainset = [(trainset[i],torch.tensor([int(mols[i][1])])) for i in range(len(mols))]

    # look like this
    # (DGLGraph(num_nodes=9, num_edges=16,
    #      ndata_schemes={'h': Scheme(shape=(35,), dtype=torch.float32)}
    #      edata_schemes={}), (<rdkit.Chem.rdchem.Mol object at 0x00000150EEE57490>, '0'))
    
    return trainset



# print(trainset[0])
data_loader = data_reader()
print("Loading trainset!",len(data_loader))
print(type(data_loader[0][0]))

#%%
dropout=0.5
gpu=-1
lr=0.01
n_epochs=200
n_hidden=16
n_layers=2
weight_decay = 5E-4
self_loop=True



# (self, in_dim, hidden_dim, n_classes)
model = Classifier(35, 128, 2)
# model = model.cuda()
# data_loader = torch.Tensor(data_loader).cuda()
# 采用交叉熵损失函数和 Adam 优化器
loss_fcn = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
print("initialize model")
print(model)


#%%


epoch_losses = []
for epoch in range(200):
    epoch_loss = 0
    for i, (bg, label) in enumerate(data_loader):
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.set_n_initializer(dgl.init.zero_initializer)        
        pred = model(bg)
        loss = loss_fcn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (i + 1)
    if (epoch+1) % 20 == 0:
        
        print('Epoch {}, loss {:.4f}'.format(epoch+1, epoch_loss))
    epoch_losses.append(epoch_loss)

plt.plot(epoch_losses, c='b')

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

accuracy_score(test_y, pred_y)


#%%
print(classification_report(test_y, pred_y))


#%%
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import rdMolDescriptors
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

def calc_dragon_type_desc(mol):
 return rdMolDescriptors.CalcAUTOCORR3D(mol) + rdMolDescriptors.CalcMORSE(mol) + \
        rdMolDescriptors.CalcRDF(mol) + rdMolDescriptors.CalcWHIM(mol)
train_X = normalize([calc_dragon_type_desc(m) for m in train_mols2])
test_X = normalize([calc_dragon_type_desc(m) for m in test_mols2])

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_X, train_y)