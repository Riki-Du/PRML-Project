import csv
import sys
import os
import pysmiles

def GetPath(file):
    path = sys.path[0]
    path = os.path.normpath(path)
    return os.path.join(path, file)

def OpenCSV(file):
    file = os.path.normpath(file)
    file = GetPath(file)
    f = open(file)
    f_csv = csv.reader(f)
    return f_csv

f_csv = OpenCSV("data/train.csv")
SMILES_list = [row[1] for row in f_csv]
SMILES = SMILES_list[1]
m = pysmiles.read_smiles(SMILES)
print(m.nodes(data = 'element'))