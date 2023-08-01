from loader import MoleculeDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
import os
import shutil
from tensorboardX import SummaryWriter
import argparse
import torch.nn.parallel
import torch.optim
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import numpy as np
import torch.optim as optim
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from ray import tune

criterion = nn.BCEWithLogitsLoss(reduction = "none")

class Classifier (torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        #model.embedding_dim, Change2_hiddenDim_classifier, num_task
        super(Classifier, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        # #print(x, x.shape, self.hidden)
        x = self.hidden(x)
        x = F.relu(x)      # activation function for hidden layer
        x = self.out(x)
        return x

class GNN_pregraph(nn.Module):
    """
    The Parameter here is copied form trained SPGCL key model: Encoder, so u need to pre-train the SPGCL_main first!
    """
    def __init__(self,num_features, dim, num_gc_layers, device, graph_pooling="sum"):
        # t_odo: 需要加一个SVC进来，下游任务sklearn做不了，我要用SVC出loss
        super(GNN_pregraph,self).__init__()
        self.num_feature,self.dim, self.num_gc_layers, self.device, self.graph_pooling = num_features, dim, num_gc_layers, device, graph_pooling
        self.gnn=Encoder_Core(num_features, dim, num_gc_layers, device)
        if self.num_gc_layers < 2:
            raise ValueError("num_gc_layers must be bigger than 1")
        self.embedding_dim = mi_units = dim * num_gc_layers     # dim 就是内边的 hidden dim

    def from_pretrained(self, model_file):
        weight_dict = torch.load(model_file)
        # print("weight_dict: ", weight_dict.keys())
        # print("model.gnn: ",model.gnn)
        encoder_weight_dict = {}
        for key, value in weight_dict.items():
            if "encoder" in key:
                new_key = key.replace("encoder.", "")  # 去掉前面的conv1
                encoder_weight_dict[new_key] = value
        # encoder_weight_dict= Or
        # model.gnn.load_state_dict(encoder_weight_dict)  # 就可以进行加载
        self.gnn.load_state_dict(encoder_weight_dict)

    def forward(self, *argv):
        loader=argv[0]
        #print("fine-model.forward, embedding is done")
        representTu,label = self.gnn.forward(loader)
        return representTu, label
        #return self.classifier(representTu), label

class Encoder_Core(torch.nn.Module):
    """
    This is the core of the model, The other class "Encoder_pred" is the interface of the finetune model
    """
    def __init__(self, num_features, dim, num_gc_layers, device):
        super(Encoder_Core, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers
        self.device = device
        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(self.num_gc_layers * dim, self.num_gc_layers * dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.num_gc_layers * dim, self.num_gc_layers * dim))

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)      # GINConv是PyG方法
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)

    def in_forward(self, x, edge_index, batch):
        # process each data in loader(for data)
        device=self.device
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):     # 这个地方的convs和bns都是需要copy以下的
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
            # if i == 2:
            # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        # x = F.dropout(x, p=0.5, training=self.training)
        y = self.proj_head(x)
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        # y = F.dropout(y, p=0.5, training=self.training)
        return y, x #第一个有用，后边是不加head的，不应该加，后边那个

    def forward(self, *argv):
        # 实现了把一堆东西data里边的信息合并起来
        device = self.device
        loader=argv[0]
        tag=True
        #with torch.no_grad():
        for data in loader:
            try:
                data = data[0]
            except TypeError:
                pass
            data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            if x is None:
                x = torch.ones((batch.shape[0], 1)).to(device)
            # x, _ = self.forward(x, edge_index, batch)
            x.to(device)
            edge_index.to(device)
            x, emb = self.in_forward(x, edge_index, batch)     # emb就是embbeding的东西
            # ret.append(x.cpu().numpy())
            x.to(device)
            emb.to(device)
            if tag:
                ret, y, tag = emb, data.y, False
            else:
                ret = torch.cat((ret,emb))#.append(emb.cpu().detach().numpy())
                y = torch.cat((y,data.y))#.append(data.y.cpu().detach().numpy())
            ##print(ret.shape)
        ##print(ret,type(ret[0]))
        #ret = np.concatenate(ret, 0)
        #y = np.concatenate(y, 0)
        return ret ,y#torch.from_numpy(ret), torch.from_numpy(y)   # 第1个有用， 第二个是干嘛的我有点没看懂。。。。

def get_loss(model,classifier, dataloader_eval, search=True):
    # t_odo： 需要把numpy方法变成tensor方法，看起来工作量不小
    #labels = preprocessing.LabelEncoder().fit_transform(labels.cpu())
    ##print(type(embeddings), type(labels))
    #x, y = np.array(embeddings.cpu().detach().numpy()), labels
    emb, labels = model.forward(dataloader_eval)
    # labels = preprocessing.LabelEncoder().fit_transform(labels.cpu())  # 这个就变成numpy了
    ##print(labels, labels.shape)
    x = classifier.forward(emb)
    ##print(x, x.shape)
    criterion = nn.CrossEntropyLoss().cuda()
    loss_out = criterion(x, labels)
    # loss = svc_classify_loss(x, y, search) 放弃了，这逼sklearn的fit用不了tensor，无语。。。。妈的花了一天
    return loss_out

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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

def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

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

def Set_augs():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
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
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = 'result.log', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    return parser.parse_args()

def Set_num_task(args):
    if args.dataset == "tox21":
        return 12
    elif args.dataset == "hiv":
        return 1
    elif args.dataset == "pcba":
        return 128
    elif args.dataset == "muv":
        return 17
    elif args.dataset == "bace":
        return 1
    elif args.dataset == "bbbp":
        return 1
    elif args.dataset == "toxcast":
        return 617
    elif args.dataset == "sider":
        return 27
    elif args.dataset == "clintox":
        return 2
    else:
        raise ValueError("Invalid dataset name.")

def Set_dataset(args):
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    return train_loader, val_loader, test_loader

def Set_Optimizer(args, model, classifier, Change1_lr):
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    # We did not make pooling
    #if args.graph_pooling == "attention":
    #    model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": classifier.parameters()}) # , "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=Change1_lr, weight_decay=args.decay)   # todo: 这个地方可以有一个分开的参数：sccle
    #print("optimizer parameters: ", optimizer)
    return optimizer

def main():
    # Set arguments
    args = Set_augs()

    # Seed and device
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    num_tasks = Set_num_task(args)

    #set up dataset
    train_loader, val_loader, test_loader = Set_dataset(args)

    #set up model todo：fit my model
    # My gnn should have: num_features, dim, num_gc_layers, device, graph_pooling="sum"
    model = GNN_pregraph(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    model.to(device)

    # set up optimizer todo: fit my model
    optimizer = Set_Optimizer(args,model)


    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        # not going to be used in current task _ wsZHANG
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
