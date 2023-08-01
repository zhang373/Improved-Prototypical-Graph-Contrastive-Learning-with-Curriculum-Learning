import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import json

# from core.encoders import *
from sklearn.cluster import DBSCAN
# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import json

from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from src.utils import (
    AverageMeter,
)
from tsne import *

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out

class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, nmb_prototypes=0, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim))

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(self.embedding_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(self.embedding_dim, nmb_prototypes, bias=False)


        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.args = arg_parse()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)
        # y = F.dropout(y, p=args.dropout_rate, training=self.training)
        # y = self.proj_head(y)

        # if self.l2norm:
        # y = F.normalize(y, dim=1)

        if self.prototypes is not None:
            return y, self.prototypes(y)
        else:
            return y

    def loss_cal(self, z, novel_prot_assign):
        estimator = 'easy'
        temperature = 0.2
        print(z.shape[0])
        batch_size=z.shape[0]//2     #batch_size相当于N'
        
        z= F.normalize(z, dim=1)
        
        #初始化pos_mask
        pos_mask = torch.ones(batch_size).bool().to(device)
        #修正pos_mask
        for i in range(batch_size):
            if i+batch_size>2*batch_size-1:         #边界
                break
            else:
                if novel_prot_assign[i+batch_size]!=novel_prot_assign[i]:
                    pos_mask[i] = False
      
        #二次修正pos_mask  （暂且空着）
   
        #计算neg_mask
        neg_mask = torch.zeros(2 * batch_size, 2 * batch_size).bool().to(device)
        for i, row in enumerate(neg_mask):
            for j, col in enumerate(row):
                if i!=j and novel_prot_assign[i]!=novel_prot_assign[j]:
                    neg_mask[i][j],neg_mask[j][i]=True,True
        #修正neg_mask（暂且空着）
      


        #两两样本差异度矩阵sample_dist计算
        sample_dist = 1 - torch.mm(z, z.t().contiguous())
       
        
        #逐行计算每行的平均值和标准差
        mu=torch.mean(sample_dist,dim=1)
        std = torch.std(sample_dist,dim=1)
        
        #计算负样本的权重矩阵reweight
        reweight = torch.exp(torch.pow(sample_dist - mu,2)/(2 * torch.pow(std,2))).to(device)
        
        reweight= reweight * neg_mask
        reweight_normalize = torch.zeros(reweight.size()[0])        #初始化一个reweight_normalize
        #计算第i行的归一化系数，并对第i行进行归一化
        for i in range(reweight.size()[0]):
            reweight_normalize[i] = torch.sum(neg_mask[i])/torch.sum(reweight[i])
         
        #计算两两样本的相似度
        sim_matrix  = torch.exp(torch.mm(z, z.t().contiguous()) / temperature)
        #对相似度矩阵进行加权
        sim_matrix = (sim_matrix * reweight).masked_select(neg_mask)
        
        
        # 计算正样本对相似度
       
        pos_sim = torch.exp(torch.sum(z[:batch_size] * z[-batch_size:],dim=-1)/temperature)

        pos_sim = pos_sim * pos_mask
        
           
        #计算对比loss
        loss = pos_sim.sum(dim=-1) / (pos_sim.sum(dim=-1) + sim_matrix.sum(dim=-1))
        loss = -torch.log(loss).mean()
    
        
        return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
#计算样本的可信度
def calculate_reliability(p1,p2,prot_assign_1,prot_assign_2):
    row=p1.size()[0]
    col=p1.size()[1]                 #p1,p2都是[N,K]的tensor
    p_1=torch.zeros(row,col)
    p_2=torch.zeros(row,col)
    for i in range(row):
        for j in range(col):
            p_1[i][j]=p1[i][j]*torch.log(p1[i][j])
    for i in range(row):
        for j in range(col):
            p_2[i][j]=p2[i][j]*torch.log(p2[i][j])                      #信息熵计算p(x)logp(x)
    reliab1 = -torch.sum(p_1,dim=1).reshape(row, 1).to(device)             #reliab1和reliab2都是[N,1]的tensor
    reliab2 = -torch.sum(p_2, dim=1).reshape(row, 1).to(device)
   
    #初步假设信息熵阈值为2.17
    ceita=2.17
    for i in range(row):
        if reliab1[i] < ceita:
            reliab1[i] = 1                            #小于阈值的代表可信，大于的代表不可信
        else:
            reliab1[i] = 0
        if reliab2[i] < ceita:
            reliab2[i] = 1
        else:
            reliab2[i] = 0
    reliab1_mask,reliab2_mask=reliab1.bool(),reliab2.bool()         #转为True or False
    reliab_mask = reliab1_mask & reliab2_mask              #当且仅当来自同一个锚图的两个样本都是可信的
    p1 = p1.masked_select(reliab_mask)
    p2 = p2.masked_select(reliab_mask)
    prot_assign_1 = prot_assign_1.reshape(row,1).masked_select(reliab_mask)
    prot_assign_2 = prot_assign_2.reshape(row,1).masked_select(reliab_mask)
  
    return p1,p2,prot_assign_1,prot_assign_2,reliab_mask

@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = 1
    vis_interval = 1
    batch_size = 128
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    args.nmb_prototypes = 30
    print("args.nmb_prototypes:{}".format(args.nmb_prototypes))

    dataset = TUDataset(path, name=DS, aug=args.aug,
                        stro_aug=args.stro_aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none',
                             stro_aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, nmb_prototypes=args.nmb_prototypes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('epochs: {}'.format(epochs))
    print('================')

    model.eval()
    init_emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    # build the queue
    queue = None
    args.queue_length -= args.queue_length % (batch_size)

    for epoch in range(1, epochs+1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        loss_all = 0
        model.train()
        use_the_queue = True
        end = time.time()

        # optionally starts a queue
        # if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
        queue = torch.zeros(
            len(args.crops_for_assign),
            args.queue_length,
            args.hidden_dim * args.num_gc_layers,
            ).cuda()

        global_emb, global_output, global_prot, global_y, global_plabel = [], [], [], [], []

        for it, data in enumerate(dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            data, data_aug, data_stro_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)

            bs = data.y.size(0)

            # update learning rate
            iteration = epoch * len(dataloader) + it
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = lr_schedule[iteration]

            if it == 0:
                global_prot.append(model.prototypes.weight)

            # normalize the prototypes
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                print("w:{}".format(w))
                model.prototypes.weight.copy_(w)
            proto_dist = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
            #proto_dist = F.normalize(proto_dist,dim=1)
            # ============ forward passes ... ============
            # feature, scores
            embedding, prot_scores1 = model(data.x, data.edge_index, data.batch, data.num_graphs)
            
            global_emb.append(embedding)
            global_output.append(prot_scores1)
            global_y.append(data.y)

            # print(model.prototypes.weight.size())

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == \
                    'ppr_aug' or args.aug == 'random2' or args.aug == 'random3' \
                    or args.aug == 'random4' or args.aug == 'dedge_nodes':
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
 
            _embedding, prot_scores2 = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

            embedding = torch.cat((embedding, _embedding))
            z=embedding
            
            prot_scores = torch.cat((prot_scores1, prot_scores2))
            # embedding = embedding.detach()
            # ============ clustering consistency loss ... ============
            loss = 0
            prot_assign, p = [], []
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    _prot_scores = prot_scores[bs * crop_id: bs * (crop_id + 1)].detach()
                    # time to use the queue
                    if queue is not None:
                      if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        # print("queue is not None")
                        use_the_queue = True
                        _prot_scores= torch.cat((torch.mm(
                            queue[i],
                            model.prototypes.weight.t()
                        ),  _prot_scores))
                        # print("queue[i]:{}".format(queue[i]))
                        # print("model.prototypes.weight.t():{}".format(model.prototypes.weight.t()))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    # print(queue.size(), embedding.size(), bs)
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                    

                    # get assignments
                    q = distributed_sinkhorn(_prot_scores)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = prot_scores[bs * v: bs * (v + 1)] / args.temperature
                   
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    p.append(F.softmax(x,dim=1))        #第一次将p2添加到p中，第二次将p1添加到p中
            print("subloss:",subloss)
            p[0],p[1]=p[1],p[0]        #将p1放置在p[0]位置上，p2放置在p[1]位置上
            prot_assign.append(torch.argmax(p[0],dim=1))  #prot_assign[0]为prot_assign_1
            prot_assign.append(torch.argmax(p[1],dim=1))  #prot_assign[1]为prot_assign_2
           
            p[0],p[1],prot_assign[0],prot_assign[1],reliab_mask=calculate_reliability(p[0],p[1],prot_assign[0],prot_assign[1])   #可信样本选择
            
           
            D=z.shape[1]
            
            z=z.masked_select(torch.cat([reliab_mask,reliab_mask],dim=0)).reshape(-1,D)
          
              
            #采用DBSCAN对model.prototypes进行聚类得到novel_prots_assign,DBSCAN的参数暂时随便设置 model.prototypes.weight.data.clone().cpu().numpy()
            prot_old2novel=DBSCAN(eps=2,min_samples=2).fit_predict(model.prototypes.weight.data.clone().cpu().numpy())
            print(prot_old2novel)
            prot_old2novel_dict=dict()
            for i in range(len(prot_old2novel)):
                prot_old2novel_dict[i]=prot_old2novel[i]
            import pdb
            pdb.set_trace()
            prot_assign_1_novel=[prot_old2novel_dict[int(i)] for i in prot_assign[0]]
            
            prot_assign_2_novel=[prot_old2novel_dict[int(i)] for i in prot_assign[1]]
            prot_assign_1_novel=torch.tensor(prot_assign_1_novel)
            prot_assign_2_novel=torch.tensor(prot_assign_2_novel)
            novel_prot_assign = torch.cat((prot_assign_1_novel,prot_assign_2_novel))
            
            

            contrast_loss = model.loss_cal(z, novel_prot_assign)
        
            loss += contrast_loss + 6 * subloss / (np.sum(args.nmb_crops) - 1)


            loss_all += loss.item() #* data.num_graphs
            loss.backward()
            import pdb 
            pdb.set_trace() 

            optimizer.step()
            # ============ misc ... ============
            losses.update(loss.item(), data.y.size(0))
            batch_time.update(time.time() - end)
            end = time.time()



        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)

        print('Epoch {}, Loss {}, acc {}'.format(epoch, loss_all / len(dataloader),acc))
        accuracies['test'].append(acc)

    print('[info] AVG acc:{}'.format(sum(accuracies['test'])/epochs))
    print('[info] MAX acc:{}'.format(max(accuracies['test'])))
    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')
