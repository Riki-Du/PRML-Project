import os
import sys
import csv
import dgl
import numpy as np
import random
import torch
from functools import partial
from dgllife.utils import smiles_to_complete_graph,smiles_to_bigraph
from dgllife.utils import CanonicalAtomFeaturizer,WeaveAtomFeaturizer,atom_type_one_hot
from dgllife.utils import CanonicalBondFeaturizer,WeaveEdgeFeaturizer,BaseAtomFeaturizer
from sklearn.preprocessing import LabelEncoder    #用于Label编码
from sklearn.preprocessing import OneHotEncoder     #用于one-hot编码
from rdkit import Chem
from rdkit import RDPaths

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

def featurize_edges(mol, add_self_loop=False):
    feats = []
    num_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())
    distance_matrix = Chem.GetDistanceMatrix(mol)
    bonds = mol.GetBonds()
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats.append(float(distance_matrix[i, j]))
        feats.append(float(distance_matrix[i, j]))
    return {'distance': torch.tensor(feats).reshape(-1, 1).float()}

def featurize_nodes(mol, add_self_loop=False):
    feats = []
    num_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())
    for at in atoms:
        feats.append(at.GetAtomicNum())
    return {'node_type': torch.tensor(feats).long()}

node_featurizer = partial(featurize_nodes, add_self_loop=False)
edge_featurizer = partial(featurize_edges, add_self_loop=False)

def load_data(num): 
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()
    # atom_featurizer = WeaveAtomFeaturizer()
    # bond_featurizer = WeaveEdgeFeaturizer()
    # atom_featurizer = node_featurizer
    # bond_featurizer = edge_featurizer
    
    # m = 'CCO'
    # mol = Chem.MolFromSmiles(m)
    # num_bonds = mol.GetNumBonds()
    # print(mol,num_bonds)
    # smiles_to_bigraph(m, add_self_loop=False, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer)
    # print(bond_featurizer(mol)['distance'].shape)
    # print(atom_featurizer(mol)['node_type'].shape)
    trainmols, train_y = read_from_rdkit(num,0)
    testmols, test_y = read_from_rdkit(num,2)
    train_g = [smiles_to_bigraph(m, add_self_loop=False, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer) for m in trainmols]
    
    train_y = np.array(train_y, dtype=np.int64)
    train_y = OneHotEncoder(sparse=False).fit(train_y.reshape(-1,1)).transform(train_y.reshape(-1,1))
    train_y = np.array(train_y, dtype=np.float32)
    print("Training set ",len(train_g))
    
    test_g = [smiles_to_bigraph(m, add_self_loop=False, node_featurizer=atom_featurizer,edge_featurizer=bond_featurizer) for m in testmols]
    test_y = np.array(test_y, dtype=np.int64)
    test_y = OneHotEncoder(sparse=False).fit(test_y.reshape(-1,1)).transform(test_y.reshape(-1,1))
    test_y = np.array(test_y, dtype=np.float32)
    print("Test set",len(test_g))
    print("Data loaded.")

    return train_g, train_y, test_g, test_y


def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def collate_molgraphs(data):
    graphs, labels = map(list,zip(*data))

    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)

    return batched_graph, torch.tensor(labels)

def load_model(args):
    if args['model'] == 'SchNet':
        from dgllife.model import SchNetPredictor
        model = SchNetPredictor(node_feats=args['node_feats'],
                                hidden_feats=args['hidden_feats'],
                                classifier_hidden_feats=args['classifier_hidden_feats'],
                                n_tasks=args['n_tasks'])

    if args['model'] == 'MGCN':
        from dgllife.model import MGCNPredictor
        model = MGCNPredictor(feats=args['feats'],
                              n_layers=args['n_layers'],
                              classifier_hidden_feats=args['classifier_hidden_feats'],
                              n_tasks=args['n_tasks'])

    if args['model'] == 'MPNN':
        from dgllife.model import MPNNPredictor
        model = MPNNPredictor(node_in_feats=args['node_in_feats'],
                              edge_in_feats=args['edge_in_feats'],
                              node_out_feats=args['node_out_feats'],
                              edge_hidden_feats=args['edge_hidden_feats'],
                              n_tasks=args['n_tasks'])

    return model
