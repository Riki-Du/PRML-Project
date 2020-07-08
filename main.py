import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dgllife.data import TencentAlchemyDataset
from dgllife.utils import EarlyStopping, Meter

from utils import set_random_seed, collate_molgraphs, load_model, load_data

training_score = []
validation_score = []

def regress(args, model, bg):
    if args['model'] == 'MPNN':
        h = bg.ndata.pop('h')
        e = bg.edata.pop('e')
        h, e = h.to(args['device']), e.to(args['device'])
        return model(bg, h, e)
    elif args['model'] in ['SchNet', 'MGCN']:
        node_types = bg.ndata.pop('node_type')
        edge_distances = bg.edata.pop('distance')
        node_types, edge_distances = node_types.to(args['device']), \
                                     edge_distances.to(args['device'])
        return model(bg, node_types, edge_distances)

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        bg, labels = batch_data
        labels = labels.to(args['device'])
        prediction = regress(args, model, bg)
        # print(prediction.dtype,labels.dtype)
        loss = (loss_criterion(prediction, labels)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    total_score = np.mean(train_meter.compute_metric(args['metric_name']))
    training_score.append(total_score)
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], total_score))

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, labels = batch_data
            labels = labels.to(args['device'])
            prediction = regress(args, model, bg)
            eval_meter.update(prediction, labels)
        total_score = np.mean(eval_meter.compute_metric(args['metric_name']))
    return total_score

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    train_g, train_y, val_g, val_y = load_data(0)

    train_set = list(zip(train_g, train_y))
    val_set = list(zip(val_g, val_y))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)

    model = load_model(args)
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    stopper = EarlyStopping(mode=args['mode'], patience=args['patience'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        validation_score.append(val_score)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric_name'], val_score,
            args['metric_name'], stopper.best_score))

        if early_stop:
            break

    plt.subplot(212)
    # plt.style.use('ggplot')
    plt.plot(training_score, c='b', alpha=0.6, label='training_roc_auc')
    plt.legend()
    plt.plot(validation_score, c='r', alpha=0.6, label='validation_roc_auc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.show()

if __name__ == "__main__":
    import argparse

    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Alchemy for Quantum Chemistry')
    parser.add_argument('-m', '--model', type=str, choices=['MPNN', 'SchNet', 'MGCN'],
                        help='Model to use')
    parser.add_argument('-n', '--num-epochs', type=int, default=20,
                        help='Maximum number of epochs for training')
    args = parser.parse_args().__dict__
    #print(args)
    args['dataset'] = 'Alchemy'
    args['exp'] = '_'.join([args['model'], args['dataset']])
    #print(args)
    args.update(get_exp_configure(args['exp']))
    #print(args)

    main(args)
