import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
from dgl import DGLGraph

# ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se',
#  'Zn', 'H', 'Cu', 'Mn', 'unknown']
# ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1 # 23 + degree, charge, is_aromatic = 39

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
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


smiles = ['OCCS(=O)(=O)c1no[n+]([O-])c1c2ccccc2', 'Cl.CCCC1(C)CC(=O)N(CCCCN2CCN(CC2)c3nsc4ccccc34)C(=O)C1']
mols = []
for sm in smiles:
    mol = get_mol(sm)
    mols.append(mol)
graphs = mol2dgl_single(mols)

graphs[0].adjacency_matrix().to_dense()

for a in graphs[1].adjacency_matrix().to_dense():
    print(a)