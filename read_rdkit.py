import csv
import sys
import os
from rdkit import Chem
from rdkit.Chem import Draw

# get a path
def GetPath(file):
    path = sys.path[0]
    path = os.path.normpath(path)
    return os.path.join(path, file)

SMILES1 = "O"
SMILES2 = "CCO"
SMILES3 = "O=C=O"
SMILES4 = "C#N"
SMILES5 = "C1CCCCC1"
SMILES6 = "CC"
SMILES7 = "C=C"
SMILES8 = "C#C"
SMILES9 = "CC(=O)OCC"
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES1), GetPath("水.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES2), GetPath("乙醇.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES3), GetPath("二氧化碳.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES4), GetPath("氰化氢.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES5), GetPath("环已烷.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES6), GetPath("乙烷.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES7), GetPath("乙烯.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES8), GetPath("乙炔.jpg"))
Draw.MolToImageFile(Chem.MolFromSmiles(SMILES9), GetPath("乙酸乙酯.jpg"))




