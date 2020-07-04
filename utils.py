# This is for some tools for models

import csv
import sys
import os
import pysmiles
from rdkit import Chem
from rdkit.Chem import Draw


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
def read_data_smiles(num=10, choice=0):
    # choice = 0,1,2
    train_path, test_path, dev_path = train_test_path(num)
    path = train_path
    if choice == 1:
        path = test_path
    elif choice == 2:
        path = dev_path

    f_csv = OpenCSV(path)
    dataset = [row[1] for row in f_csv]
    labels = [row[2] for row in f_csv]

    return dataset, labels



dataset, labels = read_data_smiles()

temp = dataset[1]
m = Chem.MolFromSmiles(temp)
Draw.MolToImageFile(m, GetPath("mol.jpg"))