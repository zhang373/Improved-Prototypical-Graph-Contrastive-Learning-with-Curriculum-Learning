import argparse
from aug import TUDataset_aug as TUDataset
import matplotlib
# from evaluate_embedding import evaluate_embedding
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import os.path as osp
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

matplotlib.use('agg')
using_my_simple_model = True    # simplify the model or not, Not used
logs = 1

def set_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--DS', dest='DS',default="NCI1", help='Dataset')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # spgcl里边是32，但是代码里边修改成了256，可以改一改调一下，32挺好
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epoch', type=int, default=2,
                        help='number of epoch to train (default: 100)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
                        help='Learning rate.')
    #decay这个spgcl没用上
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='weight decay (default: 0)')
    # 这个保持和spgcl一致
    parser.add_argument('--dropout_rate', dest='dropout_rate', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--seed', type=int, default=0)

    # 这个要检查一下模型，对一下拷贝的内容
    #
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
                        help='')
    #
    parser.add_argument("--nmb_prototypes", default=10, type=int,
                        help="number of prototypes")
    #
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                        help='Number of graph convolution layers before each pooling')

    #
    parser.add_argument('--info_ent_threshold', dest='info_ent_threshold', type=float, default=4.54553,
                        help='info_ent_threshold')  # 取值范围为[0,log_2(cluster_num)]，因为cluster_num在运行之前没法知道，因此可以暂时设置成[0,1]之间
    parser.add_argument('--pos_div_threshold', dest='pos_div_threshold', type=float, default=0.742017,
                        help='pos_div_threshold')  # 选用JS散度时，取值范围为[0,1]
    parser.add_argument('--neg_div_threshold', dest='neg_div_threshold', type=float, default=0.113974,
                        help='neg_div_threshold')  # 选用JS散度时，取值范围为[0,1]
    parser.add_argument('--reliab_pacing_type', type=str, default='logar')
    parser.add_argument('--pos_reliab_pacing_type', type=str, default='logar')
    parser.add_argument('--neg_reliab_pacing_type', type=str, default='logar')
    # args.lamda1， args.lamda2
    parser.add_argument('--lamda1', dest='lamda1', type=float, default=0.844702,
                        help='lamda1')  # 取值范围设置成[0,1]
    parser.add_argument('--lamda2', dest='lamda2', type=float, default=0.3,
                        help='lamda2')  # 取值范围设置成[0,1]
    #处理dataset的，应该不用修改，处理方法是一样的
    parser.add_argument('--aug', type=str, default='subgraph')
    parser.add_argument('--stro_aug', type=str, default='stro_dnodes')
    parser.add_argument('--weak_aug2', type=str, default=None)


    # 下边俩存结果的东西，不太想用它，一会看
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)


    #sinkhorn算法的
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--world_size", default=1, type=int, help="""
                            number of processes: it is set automatically and
                            should not be passed as argument""")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    return parser.parse_args()

def get_data(args):
    DS = args.DS
    aug = args.aug
    stro_aug = args.stro_aug
    #print(args)
    #print(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    #print(path)
    dataset = TUDataset(path, name=DS, aug=aug, stro_aug=stro_aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none',stro_aug='none').shuffle()
    print(len(dataset))
    print(len(dataset_eval))
    print(dataset.get_num_feature())
    print(dataset_eval.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)
    # init_emb, y = model.encoder.get_embeddings(dataloader_eval)
    return dataset_num_features, dataloader, dataloader_eval


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
            #print(ret.shape)
        #print(ret,type(ret[0]))
        #ret = np.concatenate(ret, 0)
        #y = np.concatenate(y, 0)
        return ret,y#torch.from_numpy(ret), torch.from_numpy(y)   # 第1个有用， 第二个是干嘛的我有点没看懂。。。。

    def get_embeddings_v(self, loader):
        # todo: 这个东西完全没用到，最后可以删掉的
        device = self.device
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                    break

        return x_g, ret, y

class GNN_pregraph(nn.Module):
    """
    The Parameter here is copied form trained SPGCL key model: Encoder, so u need to pre-train the SPGCL_main first!
    """
    def __init__(self,num_features, dim, num_gc_layers, device, num_task, graph_pooling="sum"):
        super(GNN_pregraph,self).__init__()
        self.num_feature,self.dim, self.num_gc_layers, self.device, self.graph_pooling = num_features, dim, num_gc_layers, device, graph_pooling
        self.gnn=Encoder_Core(num_features, dim, num_gc_layers, device)
        if self.num_gc_layers < 2:
            raise ValueError("num_gc_layers must be bigger than 1")
        self.embedding_dim = mi_units = dim * num_gc_layers     # dim 就是内边的 hidden dim
        self.classifier = nn.Linear(self.embedding_dim, num_task, bias=False)

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        loader=argv[0]
        print("fine1")
        representTu,label = self.gnn.forward(loader)
        return representTu, label
        #return self.classifier(representTu), label

    def loss(self, embeddings, labels, search=True):
        # todo： 需要把numpy方法变成tensor方法，看起来工作量不小
        #labels = preprocessing.LabelEncoder().fit_transform(labels.cpu())
        #print(type(embeddings), type(labels))
        #x, y = np.array(embeddings.cpu().detach().numpy()), labels
        x, y = embeddings, labels
        loss = svc_classify_loss(x, y, search)
        return loss

def svc_classify_loss(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # https://hg95.github.io/sklearn-notes/model_selection/
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        #print(kf.split(x,y))
        #print(train_index, test_index)
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)    # 训练了一份
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))       # 计算了一下acc的数据

    return np.mean(accuracies_val), np.mean(accuracies)

def Set_num_task(args):
    # 这个不知道是干嘛的, 这个没用到，他的下游任务变成SVC了，不是简单分类，要配合SVC，不用自己写classifier，但是要调
    return 20

def Set_Optimizer(model,args):
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    # We did not make pooling
    #if args.graph_pooling == "attention":
    #    model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.classifier.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)
    return optimizer


# 以下是train的部分，下游任务是svc_classify
def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # https://hg95.github.io/sklearn-notes/model_selection/
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):
        #print(kf.split(x,y))
        #print(train_index, test_index)
        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)    # 训练了一份
        print("classifier.predict(x_test):", classifier.predict(x_test))
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))       # 计算了一下acc的数据

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies_val), np.mean(accuracies)

def evaluate_embedding(embeddings, labels, search=True):

    print(labels)
    labels = preprocessing.LabelEncoder().fit_transform(labels.cpu())
    print(labels)
    print(type(embeddings),type(labels))
    x, y = np.array(embeddings.cpu().detach().numpy()), labels

    acc = 0
    acc_val = 0

    '''
    _acc_val, _acc = logistic_classify(x, y)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    _acc_val, _acc = svc_classify(x,y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    """
    _acc_val, _acc = linearsvc_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    """
    '''
    _acc_val, _acc = randomforest_classify(x, y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc
    '''

    print(acc_val, acc)

    return acc_val, acc

def eval(epoch,model,dataloader_eval,accuracies):
    #if epoch % args.log_interval == 0:
    model.eval()

    #emb, y = model.encoder.get_embeddings(dataloader_eval)
    emb, y = model.forward(dataloader_eval)
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    #print('Epoch {}, Loss {}, acc {}'.format(epoch, loss_all / len(dataloader), acc))
    print('Epoch {}, acc {}'.format(epoch, acc))

def Get_loss():
    return

def main():
    # baseline
    args = set_args()
    num_task = Set_num_task(args)
    torch.manual_seed(args.seed)    # default 0,可输入量
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")  # default 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # data
    dataset_num_features, dataloader, dataloader_eval = get_data(args)      # dataloader是增强之后的数据容器， dataloader_eval是原始的
    print("Data:",args.DS," is OK")

    # model
    model= GNN_pregraph(dataset_num_features, args.hidden_dim, args.num_gc_layers, device, num_task, graph_pooling="sum" )
    #model= Encoder_Core(dataset_num_features,args.hidden_dim,args.num_gc_layers,device)
    # todo: 下游任务在 evaluate_embedding里边，没看明白, 是否可以只进行一个简单分类呢？ no, 需要适配svc
    model.to(device)
    #print(model)
    #representTu, label = model.forward(dataloader)
    #print(representTu)

    # optimizer
    optimizer=Set_Optimizer(model,args)

    accuracies = {'val': [], 'test': []}
    for epoch in range(1, args.epoch):
        if epoch % logs == 0:
            eval(epoch, model,dataloader_eval,accuracies)   #这个dataloader应该换一下哈



if __name__ == "__main__":
    main()