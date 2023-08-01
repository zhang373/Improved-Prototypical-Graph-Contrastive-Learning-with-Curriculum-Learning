import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
from splitters import scaffold_split, random_split, random_scaffold_split
import torch.nn.functional as F
import json
import pandas as pd
from numpy import exp
# from core.encoders import *
from sklearn.cluster import DBSCAN
# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
from loader import MoleculeDataset_aug
import json
import scipy.stats
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
import math
from arguments import arg_parse
from copy import deepcopy
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


    def forward(self, x, edge_index, batch, num_graphs, Encoder_support):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch, Encoder_support)
        # y = F.dropout(y, p=args.dropout_rate, training=self.training)
        # y = self.proj_head(y)

        # if self.l2norm:
        # y = F.normalize(y, dim=1)
        m = torch.nn.ReLU()
        if self.prototypes is not None:
            return y, m(self.prototypes(y))
            # return y, self.prototypes(y)
        else:
            return y
    

    def loss_cal(self, z, reliab_neg_mask,reliab_pos_mask,reliab_pos_idx):
        estimator = 'easy'
        temperature = 0.2
        batch_size=z.shape[0]//2     #batch_size相当于N
        reliab_neg_mask = reliab_neg_mask.to(device)
        reliab_pos_mask = reliab_pos_mask.to(device)
        z= F.normalize(z, dim=1)
        #--1)基于所有样本对的距离评估
        #两两样本差异度矩阵sample_dist计算
        sample_dist = 1 - torch.mm(z, z.t().contiguous())
        #逐行计算每行的平均值和标准差
        mu=torch.mean(sample_dist,dim=1)
        std = torch.std(sample_dist,dim=1)
        #计算负样本的权重矩阵reweight
        reweight = torch.exp(-torch.pow(sample_dist - mu,2)/(2 * torch.pow(std,2))).to(device)
        reweight= reweight * reliab_neg_mask
        #--2)基于可信负样本对评估
        #sample_dist = 1 - torch.mm(z, z.t().contiguous())
        #mu = torch.sum(sample_dist*reliab_neg_mask,dim=1) / (torch.sum(reliab_neg_mask,dim=1)+ exp(-10))
        #reshape_mu = mu.reshape(-1,1)
       # temp_res = torch.pow(sample_dist.sub(reshape_mu),2)*reliab_neg_mask
       # std = torch.sqrt(temp_res / (torch.sum(reliab_neg_mask,dim=1)+ exp(-10)))
       # reweight = torch.exp(-torch.pow(sample_dist - mu,2)/(2 * torch.pow(std,2)+ exp(-10))).to(device)
       # reweight= reweight * reliab_neg_mask
       
        reweight_normalize = torch.sum(reliab_neg_mask,dim=1) / (torch.sum(reweight,dim=1) + exp(-10))
        reweight = reweight * reweight_normalize.reshape(reweight.shape[0],1)
        #计算两两样本的相似度
        sim_matrix  = torch.exp(torch.mm(z, z.t().contiguous()) / temperature).to(device)
        #对相似度矩阵进行加权
        sim_matrix = (sim_matrix * reweight)*(reliab_neg_mask)
      
        # 计算正样本对相似度
        pos_sim = torch.exp(torch.sum(z[:batch_size] * z[-batch_size:],dim=-1)/temperature).to(device)
        pos_sim = pos_sim * reliab_pos_mask
        pos_sim = torch.cat((pos_sim,pos_sim),dim=0)
     
        #计算对比loss
        # print(f"pos_sim:{pos_sim}")
        # print(f"sim_matrix.sum(dim=-1): {sim_matrix.sum(dim=-1)}")
        # if reliab_pos_idx.shape[0] == 0 :
        #     return 0
        # else:
        #     print(f"shape: {(pos_sim /( pos_sim + sim_matrix.sum(dim=-1) + exp(-10)))[reliab_pos_idx].shape}; value: {(pos_sim /( pos_sim + sim_matrix.sum(dim=-1) + exp(-10)))[reliab_pos_idx]}")
        #     loss = -(torch.log((pos_sim /( pos_sim + sim_matrix.sum(dim=-1) + exp(-10)))[reliab_pos_idx])).mean()
            
        #     return loss

        loss = -(torch.log((pos_sim /( pos_sim + sim_matrix.sum(dim=-1) + exp(-10)))[reliab_pos_idx])).mean()
        
        return loss



import random
def setup_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
#计算信息熵函数
def get_info_ent(p):
    info_ent_p = torch.zeros(p.shape[0],1)
    for i in range(info_ent_p.shape[0]):
        info_ent_p[i] = - torch.sum(p[i]*torch.log(p[i]),dim=-1)
    return info_ent_p 
#计算JS散度函数
def get_div(p,q):
    
    p=p.cpu().detach().numpy()
    q=q.cpu().detach().numpy()
    
    M = (p+q) / 2
    JS_div = 0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)
    return JS_div

def score_agg(prot_2_cluster_dict,prot_score,cluster_num):
    cluster_score = torch.zeros(prot_score.shape[0],cluster_num)  
    for i in range(prot_score.shape[0]):
        for prot in range(prot_score.shape[1]):
            cluster = prot_2_cluster_dict[prot]
            cluster_score[i][cluster] = max(cluster_score[i][cluster],prot_score[i][prot])
    return cluster_score
            
def reliab_pacing(avg,threshold,pacing_type,sample_num,t,T):
    relative_cap = min(1,threshold/avg) #relative_cap = threshold / (threshold+avg)
    # print(f"avg:{avg}, relative_cap:{relative_cap}")
    #relative_cap = 1
    if pacing_type == "logar":
        reliab_num = (1+0.1*math.log(relative_cap*(t/T)+exp(-10))) * sample_num
    elif pacing_type == "poly1":
        reliab_num = (relative_cap*(t/T)) * sample_num
    elif pacing_type == "poly2":
        reliab_num = (relative_cap*(t/T))**2 * sample_num
    elif pacing_type == "poly3":
        reliab_num = (relative_cap*(t/T))**3 * sample_num
    # print(reliab_num)
    return int(reliab_num)

def reliab_neg_pacing(neg_cand_div_avg,neg_div_threshold,pacing_type,sample_num,t,T):
    relative_cap = min(1,neg_cand_div_avg/neg_div_threshold)
    # print(f"neg_cand_div_avg:{neg_cand_div_avg}, relative_cap:{relative_cap}")
    #relative_cap = 1
    if pacing_type == "logar":
        reliab_num = (1+0.1*math.log(relative_cap*(t/T)+exp(-10))) * sample_num
    elif pacing_type == "poly1":
        reliab_num = (relative_cap*(t/T)) * sample_num
    elif pacing_type == "poly2":
        reliab_num = (relative_cap*(t/T))**2 * sample_num
    elif pacing_type == "poly3":
        reliab_num = (relative_cap*(t/T))**3 * sample_num
    # print(reliab_num)
    return int(reliab_num)

def get_reliab_mask(info_ent1,info_ent2,reliab_num):
    info_ent = torch.cat((info_ent1,info_ent2))
    reliab_idx = torch.argsort(info_ent,dim=0)[:reliab_num] #选取信息熵最低的reliab_num个样本作为可信样本
    sample_num = len(info_ent)
    reliab_mask = torch.zeros(sample_num).bool()
    for i in range(len(reliab_idx)):
        reliab_mask[reliab_idx[i]] = True
    #当且仅当两个都为可信时
    #reliab_mask1,reliab_mask2 = reliab_mask[:sample_num/2],reliab_mask[sample_num/2:]
    #reliab_mask = reliab_mask1 & reliab_mask2
    #reliab_mask = torch.cat((reliab_mask,reliab_mask))
    return reliab_mask,reliab_idx

def get_reliab_pos_mask(pos_div,reliab_pos_num):
    reliab_pos_idx = torch.argsort(pos_div)[:reliab_pos_num]
    sample_num = len(pos_div)
    reliab_pos_mask = torch.zeros(sample_num).bool()
    for i in range(len(reliab_pos_idx)):
        reliab_pos_mask[reliab_pos_idx[i]]=True         #将可信样本置为True
    return reliab_pos_mask,reliab_pos_idx

def get_reliab_neg_mask(neg_div,reliab_neg_num):
    sample_num = neg_div.shape[0]
    reliab_neg_mask = torch.zeros(sample_num,sample_num).bool()
    neg_div = neg_div.reshape(-1)
    reliab_neg_idx = torch.argsort(neg_div,descending=True)[:reliab_neg_num]
    reliab_neg_row_idx = []
    reliab_neg_col_idx = [] 
    for idx in reliab_neg_idx:
        i = (idx / sample_num).type(torch.long)
        j = (idx % sample_num).type(torch.long)
        reliab_neg_mask[i][j] = True
        reliab_neg_row_idx.append(i)
        reliab_neg_col_idx.append(j)
    return reliab_neg_mask,reliab_neg_row_idx,reliab_neg_col_idx
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

# #***************
# def get_cluster_num(prot_2_cluster):
#     # prot_2_cluster是DBSCAN.fit_predict()的返回值：
#     # 1）没有noisy sample时，返回值为0~(cluster_num - 1)之间的连续整数； 
#     # 2）有noisy sample时，返回值有-1~(unnoisy_cluster_num - 1)，这种情况下cluster_num = unnoisy_cluster_num + 1 + (-1 出现的总次数)
#     if -1 not in prot_2_cluster:
#         # print("-1 not in")
#         cluster_num = max(prot_2_cluster)+1
#     else:
#         # print("-1 in")
#         unnoisy_cluster_num = max(prot_2_cluster)+1
#         noisy_cluster_num = sum(i==-1 for i in prot_2_cluster)
#         cluster_num = unnoisy_cluster_num + noisy_cluster_num
#      return cluster_num

#***************
def denoise_cluster_idx(prot_2_cluster):
    # prot_2_cluster是DBSCAN.fit_predict()的返回值：
    # 1）没有noisy sample时，返回值为0~(cluster_num - 1)之间的连续整数； 
    # 2）有noisy sample时，返回值有-1~(unnoisy_cluster_num - 1)，这种情况下cluster_num = unnoisy_cluster_num + 1 + (-1 出现的总次数)
    if -1 not in prot_2_cluster:
        return prot_2_cluster
    else:
        unnoisy_cluster_num = max(prot_2_cluster)+1
        noisy_cluster_idx = unnoisy_cluster_num
        noisy_cluster_num = sum(i==-1 for i in prot_2_cluster)
        prot_num = len(prot_2_cluster)
        # cluster_num = unnoisy_cluster_num + noisy_cluster_num
        for i in range(prot_num):
            if prot_2_cluster[i] == -1:
                prot_2_cluster[i] = noisy_cluster_idx
                noisy_cluster_idx += 1
                noisy_cluster_num -= 1
                if noisy_cluster_num==0: # prot_2_cluster中的-1已经被遍历完成
                    break
        return prot_2_cluster


if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)
    device = torch.device('cuda:1') # if torch.cuda.is_available() else 'cpu')

    accuracies = {'val':[], 'test':[]}
    epochs = args.epochs
    log_interval = 1
    vis_interval = 1
    batch_size = 128
    # batch_size = 512
    lr = args.lr
    # def forward(self, x, edge_index, batch, num_graphs):
    DS = args.DS
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    args.nmb_prototypes = 10
    prototype_num = args.nmb_prototypes
    print("args.nmb_prototypes:{}".format(args.nmb_prototypes))

    #dataset = TUDataset(path, name=DS, aug=args.aug, stro_aug=args.stro_aug).shuffle()
    #dataset_eval = TUDataset(path, name=DS, aug='none',stro_aug='none').shuffle()
    dataset = MoleculeDataset_aug("dataset/" + DS, dataset=DS)
    #dataset_eval = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    smiles_list = pd.read_csv('dataset/' + DS + '/processed/smiles.csv', header=None)[0].tolist()
    #train_dataset, valid_dataset, test_dataset = scaffold_split(dataset,smiles_list, null_value=0, frac_train=0, frac_valid=0,frac_test=1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    dataloader = DataLoader(dataset1, batch_size=batch_size, num_workers = args.num_workers, shuffle=False)
    dataloader_eval = DataLoader(dataset2, batch_size=batch_size, num_workers = args.num_workers, shuffle=False)
    #dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = args.num_workers)
    try:
        for it, data in enumerate(dataloader):
            data = data
            dataset_num_features = data.x.size()[1]
            break
    except:
        print("wrong")
        dataset_num_features = 1
    Change4_num_chirality_tag, Change3_num_atom_type = 3, 128
    Encoder_support = encoder_support(Change4_num_chirality_tag, Change3_num_atom_type, dataset_num_features)
    Encoder_support.to(device)
    model = simclr(args.hidden_dim, args.num_gc_layers, nmb_prototypes=args.nmb_prototypes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # todo: deal with y
    #emb, y = model.encoder.get_embeddings(dataloader_test, device, Encoder_support)
    #acc_val, acc = evaluate_embedding(emb, y)
    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('epochs: {}'.format(epochs))
    print('info_ent_threshold: {}'.format(args.info_ent_threshold))
    print('pos_div_threshold: {}'.format(args.pos_div_threshold))
    print('neg_div_threshold: {}'.format(args.neg_div_threshold))
    print('================')

    model.eval()
    # init_emb, y = model.encoder.get_embeddings(dataloader_eval, device, Encoder_support)
    # print(emb.shape, y.shape)
    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """

    for epoch in range(1, epochs+1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        loss_all = 0
        model.train()
        use_the_queue = True
        end = time.time()

        for it, data in enumerate(zip(dataloader,dataloader_eval)):
            # measure data loading time
            data_time.update(time.time() - end)
            data, data_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)
            bs = batch_size
            # update learning rate
            iteration = epoch * len(dataloader) + it
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = lr_schedule[iteration]

            # normalize the prototypes
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                # print("w:{}".format(w))
                model.prototypes.weight.copy_(w)
            proto_dist = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
            #print(f"proto_dist:{proto_dist}")
            # ============ forward passes ... ============
            # feature, scores
            embedding, prot_scores1 = model(data.x, data.edge_index, data.batch, data.num_graphs,Encoder_support)

            # print(f"embedding: {embedding}")
            # print(f"model.prototypes: {model.prototypes}")

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

            _embedding, prot_scores2 = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs, Encoder_support)

            embedding = torch.cat((embedding, _embedding))
            z=embedding
            temp = 0.2
            prot_scores = torch.cat((prot_scores1, prot_scores2))
            #根据打分归一化得到预测概率p1和p2；为各图选择概率最大的prototypes，得到prot_assign
            p1 = F.softmax(prot_scores1 / temp,dim=1)
            prot_assign1 = torch.argmax(p1,dim=1)
            p2 = F.softmax(prot_scores2 / temp,dim=1)
            prot_assign2 = torch.argmax(p2,dim=1)
            #embedding = embedding.detach()

            loss = 0
            #prot_assign, p = [], []
            #根据Sinkhorn计算q1和q2
            with torch.no_grad():
                q1 = distributed_sinkhorn(prot_scores1)[-bs:]
                q2 = distributed_sinkhorn(prot_scores2)[-bs:]

            #采用DBSCAN对prototypes进行聚类，得到各样本的聚类assignment，即cluster_assign
            #prot_2_cluster=DBSCAN(eps=0.8,min_samples=1).fit_predict(model.prototypes.weight.data.clone().cpu().numpy())
            # print(f"prot_2_cluster:{prot_2_cluster}")
            # DBSCAN.fit_predict()的返回值：
            # 1）没有noisy sample时，返回值为0~(cluster_num - 1)之间的连续整数； 
            # 2）有noisy sample时，返回值有-1~(unnoisy_cluster_num - 1)，这种情况下cluster_num = unnoisy_cluster_num + 1 + (-1 出现的总次数)


            #将prot_2_cluster中的-1逐个替换成从(unnoisy_cluster_num + 1)开始连续整数,并且计算cluster_num #***************
            #prot_2_cluster = denoise_cluster_idx(prot_2_cluster)

            #cluster_num = prototype_num
            # print(f"cluster_num:{cluster_num}")

            #计算对应的字典
            #prot_2_cluster_dict=dict()
            #for i in range(len(prot_2_cluster)):
            #prot_2_cluster_dict[i]=prot_2_cluster[i]
            # prot_2_cluster_dict = dict(sorted(prot_2_cluster_dict.items(),key=lambda x:x[1])) #*************此处无意义，可以删掉

            # 得到cluster_score
            #cluster_score1 = score_agg(prot_2_cluster_dict,prot_scores1,cluster_num)
            #cluster_score2 = score_agg(prot_2_cluster_dict,prot_scores2,cluster_num)
            #cluster_score = torch.cat((cluster_score1,cluster_score2))
            # print(f"cluster_score: {cluster_score}")

            #cluster_p1 = F.softmax(cluster_score1 / temp,dim=1)
            #cluster_p2 = F.softmax(cluster_score2 / temp,dim=1)
            #cluster_p = torch.cat((cluster_p1,cluster_p2))
            # print(f"cluster_p: {cluster_p}")
            #cluster_assign1 = torch.argmax(cluster_p1,dim=1)
            #cluster_assign2 = torch.argmax(cluster_p2,dim=1)
            # cluster_assign1 = get_cluster_assign(prot_2_cluster_dict,prot_assign1)
            # cluster_assign2 = get_cluster_assign(prot_2_cluster_dict,prot_assign2)
            prot_assign = torch.cat((prot_assign1,prot_assign2))
            # print(f"cluster_assign: {cluster_assign}")
            #根据Sinkhorn计算cluster_q1和cluster_q2
            #with torch.no_grad():
            #cluster_q1 = distributed_sinkhorn(cluster_score1)
            #cluster_q2 = distributed_sinkhorn(cluster_score2)

            # ============ clustering consistency loss ... ============
            L_cluster_consistency = 0
            L_cluster_consistency -= torch.mean(torch.sum(q1 * F.log_softmax(prot_scores2 / temp,dim=1),dim=1)) +torch.mean(torch.sum(q2 * F.log_softmax(prot_scores1 / temp,dim=1),dim=1))
            #L_cluster_consistency -= torch.mean(torch.sum(q1 * F.log_softmax(prot_score2 / temp,dim=1),dim=1)) +torch.mean(torch.sum(q2 * F.log_softmax(prot_score1 / temp,dim=1),dim=1))
            L_cluster_consistency = (0.5*L_cluster_consistency).to(device)

            #根据prototypes/聚类概率,计算各样本的信息熵并判断各样本是否可信
            info_ent1 = get_info_ent(p1) #info_ent1 = get_info_ent(p1)
            info_ent2 = get_info_ent(p2) #info_ent2 = get_info_ent(p2)
            info_ent = torch.cat((info_ent1,info_ent2))

            info_ent_avg = torch.mean(info_ent)
            print(f"info_ent_avg:{info_ent_avg}; relative_cap:{args.info_ent_threshold / info_ent_avg}")
            sample_num = 2*bs
            reliab_num = reliab_pacing(info_ent_avg,args.info_ent_threshold,args.reliab_pacing_type,sample_num,epoch,epochs)
            reliab_num = int(reliab_num)

            reliab_mask,reliab_idx = get_reliab_mask(info_ent1,info_ent2,reliab_num)

            #选定[待选正样本对]pos_cand_mask;进一步根据正样本对的双向KL散度orJS散度确定reliab_pos_mask
            graph_num = sample_num // 2
            #来自同一个锚图的两个视图均可信时，二者构成的正样本对才是可信的

            pos_cand_mask = reliab_mask[:graph_num] & reliab_mask[graph_num:]

            #来自同一个锚图的两个视图能聚到同一类时，二者构成的正样本对才是可信的

            assign_mask = [prot_assign1[i]==prot_assign2[i] for i in range(graph_num)] #assign_mask=[prot_assign1[i]==prot_assign2[i] for i in range(graph_num)]
            assign_mask = torch.tensor(assign_mask)
            pos_cand_mask = pos_cand_mask & assign_mask

            pos_cand_num = 0                   #torch1.4版本没有count_nonzero函数
            for i in pos_cand_mask:
                if i!=False:
                    pos_cand_num+=1

            #计算待选正样本对聚类概率的差异度，非待选正样本对的位置置为inf
            pos_div = [get_div(p1[i],p2[i]) if pos_cand_mask[i]==True else 1000 for i in range(graph_num)]
            # print(f"pos_div:{pos_div}")
            pos_div = torch.tensor(pos_div)
            pos_cand_div_avg = torch.sum(pos_div * pos_cand_mask) / pos_cand_num
            print(f"pos_cand_div_avg:{pos_cand_div_avg}; pos_relative_cap:{args.pos_div_threshold / pos_cand_div_avg}")
            reliab_pos_num = reliab_pacing(pos_cand_div_avg,args.pos_div_threshold,args.pos_reliab_pacing_type,pos_cand_num,epoch,epochs)

            reliab_pos_mask,reliab_pos_idx = get_reliab_pos_mask(pos_div,reliab_pos_num)
            reliab_idx = reliab_idx.reshape(-1)
            #选定[待选负样本对]neg_cand_mask;进一步根据负样本对的双向KL散度orJS散度确定reliab_neg_mask
            neg_cand_mask = torch.zeros(sample_num,sample_num).bool()
            #计算待选负样本对聚类概率的差异度，非待选负样本对的位置置为0
            neg_div = torch.zeros(sample_num,sample_num)
            #当样本i拥有正样本对时，才为其选择负样本
            for i in reliab_pos_idx:
                #当样本j是可信样本时，才有资格被选为负样本
                for j in reliab_idx:
                    not_self = (i!=j)
                    not_same_graph = i!=(j+graph_num) #(i!=(j+graph_num) and j!=(i+graph_num))###################
                    i_cluster_idx,j_cluster_idx = prot_assign[i],prot_assign[j] #样本i,j所属类别的索引
                    not_same_cluster = (i_cluster_idx != j_cluster_idx)    #not_same_cluster = prot_assign[i]!=prot_assign[j]
                    if not_self and not_same_graph and not_same_cluster:

                        neg_cand_mask[i][j] = True
                        #只关心负样本对两者所属类别上的概率差异度

                        score_i = torch.stack((prot_scores[i][i_cluster_idx],prot_scores[i][j_cluster_idx]))
                        score_j = torch.stack((prot_scores[j][i_cluster_idx],prot_scores[j][j_cluster_idx]))
                        p_i = F.softmax(score_i / temp,dim=0)
                        p_j = F.softmax(score_j / temp,dim=0)
                        neg_div[i][j] = get_div(p_i,p_j)

            neg_cand_num = exp(-10)
            for i in range(neg_cand_mask.shape[0]):
                for j in range(neg_cand_mask.shape[1]):
                    if neg_cand_mask[i][j]!=False:
                        neg_cand_num+=1
            # print(f"neg_cand_num: {neg_cand_num}")
            neg_cand_div_avg =  torch.sum(neg_div) / neg_cand_num
            print(f"neg_cand_div_avg:{neg_cand_div_avg}; neg_relative_cap:{neg_cand_div_avg / args.neg_div_threshold}")
            # print(f"neg_cad_div_neg: {neg_cand_div_avg}")

            reliab_neg_num = reliab_neg_pacing(neg_cand_div_avg,args.neg_div_threshold,args.neg_reliab_pacing_type,neg_cand_num,epoch,epochs)
            reliab_neg_mask,reliab_neg_row_idx,reliab_neg_col_idx = get_reliab_neg_mask(neg_div,reliab_neg_num)
            #计算对比损失
            # todo:查一下z
            L_contrastive = model.loss_cal(z,reliab_neg_mask,reliab_pos_mask,reliab_pos_idx)


            # import pdb
            # pdb.set_trace()
            loss += L_contrastive + args.lamda1 * L_cluster_consistency
            # print(f"loss:{loss}; L_contrastive:{L_contrastive}; L_cluster_consistency: {L_cluster_consistency}")
            # import pdb
            # pdb.set_trace()
            loss_all += loss.item() #* data.num_graphs
            loss.backward()

            optimizer.step()

            # ============ misc ... ============
            losses.update(loss.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
            # import pdb
            # pdb.set_trace()



        #if epoch % log_interval == 0:

            #model.eval()

            #emb, y = model.encoder.get_embeddings(dataloader_eval,device,Encoder_support)
            # todo: deal with y
            #acc_val, acc = evaluate_embedding(emb, y)
            #accuracies['val'].append(acc_val)
            #accuracies['test'].append(acc)
        #print('Epoch {}, Loss {}, acc {}'.format(epoch, loss_all / len(dataloader),acc))

    torch.save(model.state_dict(), 'model_weights.pth')
    max_acc = max(accuracies['test'])


    print('[info] MAX acc:{}'.format(max_acc))
    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')
