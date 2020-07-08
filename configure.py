# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

MPNN_Alchemy = {
    'random_seed': 0,
    'batch_size': 64,
    'node_in_feats': 74,
    'node_out_feats': 64,
    'edge_in_feats': 12,
    'edge_hidden_feats': 64,
    'n_tasks': 2,
    'lr': 0.001,
    'patience': 50,
    'metric_name': 'roc_auc_score',
    'mode': 'higher',
    'weight_decay': 0
}

SchNet_Alchemy = {
    'random_seed': 0,
    'batch_size': 64,
    'node_feats': 64,
    'hidden_feats': [64, 64, 64],
    'classifier_hidden_feats': 64,
    'n_tasks': 2,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'roc_auc_score',
    'mode': 'higher',
    'weight_decay': 0
}

MGCN_Alchemy = {
    'random_seed': 0,
    'batch_size': 64,
    'feats': 39,
    'n_layers': 3,
    'classifier_hidden_feats': 64,
    'n_tasks': 2,
    'lr': 0.0001,
    'patience': 50,
    'metric_name': 'roc_auc_score',
    'mode': 'higher',
    'weight_decay': 0
}

experiment_configures = {
    'MPNN_Alchemy': MPNN_Alchemy,
    'SchNet_Alchemy': SchNet_Alchemy,
    'MGCN_Alchemy': MGCN_Alchemy
}
def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
