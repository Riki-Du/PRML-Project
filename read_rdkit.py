import csv
import sys
import os
from rdkit import Chem
from rdkit.Chem import Draw

data_train_path = "data/train.csv"
data_test_path = "data/test.csv"


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

# to read data
def read_data_smiles(num=10):
    train_path, test_path, dev_path = train_test_path(num)
    train_csv = OpenCSV(train_path)


def read_from_rdkit(num=10):
    train_path, test_path, dev_path = train_test_path(num)
    f_csv = OpenCSV(train_path)
    if num == 10:
        # id,smiles,activity
        SMILES_list = [(row[1],row[2]) for row in f_csv]
    else:
        # smiles,activity
        SMILES_list = [(row[0],row[1]) for row in f_csv]
    SMILES = SMILES_list[1]
    print("Test for rdkit!")
    print(SMILES)
    return SMILES_list[1:]

    # Draw.MolToImageFile(m, GetPath("mol.jpg"))