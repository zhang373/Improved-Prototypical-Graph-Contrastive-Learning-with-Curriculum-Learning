import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import Encoder_Core,GNN_pregraph, encoder_support
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

from ray import tune

# raytuned arguments' best results:
# todo: 这个dataset_num_features不太好动，最后手来吧。。。。
dataset_num_features = 128
#Change3_num_atom_type = 128 #tune.choice([32, 64, 128, 200, 256]),  # 自定义采样
#Change4_num_chirality_tag = 3 #tune.choice([2, 3, 6])

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer,Encoder_support):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        #pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        pred= model.forward(batch.x, batch.edge_index, batch.batch, Encoder_support)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader, Encoder_support):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            #pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model.forward(batch.x, batch.edge_index, batch.batch, Encoder_support)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def Train(config):
    # import fixed configs
    train_loader, val_loader, test_loader = config['train_loader'], config['val_loader'],config['test_loader']
    args, model, device, optimizer, fname = \
        config['args'], config['model'], config['device'],config['optimizer'], config['fname']
    val_acc_list, test_acc_list, train_acc_list = config['val_acc_list'], config['test_acc_list'], config['train_acc_list']

    # import variable configs
    Change4_num_chirality_tag, Change3_num_atom_type, dataset_num_features=config['Change4_num_chirality_tag'], \
                                                                           config['Change3_num_atom_type'], config['dataset_num_features']
    writer = SummaryWriter(fname)
    Encoder_support = encoder_support(Change4_num_chirality_tag, Change3_num_atom_type, dataset_num_features)
    Encoder_support.to(device)
    for epoch in range(1, args.epochs + 1):
        #print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer, Encoder_support)

        #print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, Encoder_support)
        else:
            #print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader, Encoder_support)
        test_acc = eval(args, model, device, test_loader, Encoder_support)

        print("train: %f val: %f test: %f" % (train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        #print("")

    if not args.filename == "":
        writer.close()
    #tune.report(test_acc=max(test_acc_list), val_acc=max(val_acc_list), train_acc=max(train_acc_list))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    #parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument("--model_file", default='model_weights.pth', type=str,
                        help="the name of the file storing the pretrained encoder' parameter.value")
    parser.add_argument('--filename', type=str, default = 'result.log', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = 'cuda'#torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    #print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        #print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        #print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        #print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    ##print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    #model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    model = GNN_pregraph(dataset_num_features, args.hidden_dim, args.num_gc_layers, device,num_tasks,graph_pooling="sum")
    model.from_pretrained(args.model_file)
    model.to(device)


    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.classifier.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    #print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            #print("removed the existing file.")
        #writer = SummaryWriter(fname)
    #https: // blog.csdn.net / m0_38052500 / article / details / 122136111
    config = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "args": args,
        "model": model,
        "device": 'cuda',
        "optimizer": optimizer,
        "fname":fname,
        "val_acc_list": val_acc_list,
        "test_acc_list": test_acc_list,
        "train_acc_list": train_acc_list,
        "dataset_num_features": dataset_num_features,#tune.choice([128, 96, 72, 168]),
        "Change3_num_atom_type": 128,#tune.choice([32, 64, 128, 200, 256]),  # 自定义采样
        "Change4_num_chirality_tag":3,#tune.choice([2, 3, 6])
    }
    Train(config)
    #result = tune.run(  # 执行训练过程，执行到这里就会根据config自动调参了
    #    Train,  # 要训练的模型
    #    resources_per_trial={"gpu": 1, "cpu": 7},  # 指定训练资源
    #    config=config,
    #    num_samples=20,  # 迭代的次数
    #)
    #print("======================== Result =========================")
    #print(result.results_df)

if __name__ == "__main__":
    main()
