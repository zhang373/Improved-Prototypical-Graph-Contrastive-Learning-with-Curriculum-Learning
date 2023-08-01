import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Sequential, Linear, ReLU
from model import GINConv

class encoder_support(torch.nn.Module):
    # 这个是为了适配输入的数据做的embedding，参考chem/model/GNN,因为model送不进去，所以单列一个model
    def __init__(self,num_chirality_tag,num_atom_type, num_features):
        super(encoder_support, self).__init__()
        self.x_embedding1 = torch.nn.Embedding(num_atom_type, num_features)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, num_features)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
    def forward(self,x):
        old_x = x
        x = self.x_embedding1(old_x[:,0]) + self.x_embedding2(old_x[:,1])
        return x

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

    def forward(self, x, edge_index, batch, Encoder_support):
        # process each data in loader(for data)
        device=self.device
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        x = Encoder_support.forward(x)
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

    def loader_forward(self, *argv):
        # 实现了把一堆东西data里边的信息合并起来
        device = self.device
        loader=argv[0]
        Encoder_support = argv[1]
        tag=True
        #with torch.no_grad():
        for data in loader:
            try:
                data = data[0]
            except TypeError:
                pass
            data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            print("data.y is this :",data.y)
            if x is None:
                x = torch.ones((batch.shape[0], 1)).to(device)
            # x, _ = self.forward(x, edge_index, batch)
            x.to(device)
            x = Encoder_support.forward(x)
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
            print("-------->", ret.shape,y.shape )
            #print(ret.shape)
        #print(ret,type(ret[0]))
        #ret = np.concatenate(ret, 0)
        #y = np.concatenate(y, 0)
        return ret ,y#torch.from_numpy(ret), torch.from_numpy(y)   # 第1个有用， 第二个是干嘛的我有点没看懂。。。。

class GNN_pregraph(torch.nn.Module):
    """
    The Parameter here is copied form trained SPGCL key model: Encoder, so u need to pre-train the SPGCL_main first!
    """
    def __init__(self,num_features, dim, num_gc_layers, device, graph_pooling="sum"):
        # t_odo: 需要加一个SVC进来，下游任务sklearn做不了，我要用SVC出loss
        super(GNN_pregraph,self).__init__()
        self.num_feature,self.dim, self.num_gc_layers, self.device, self.graph_pooling = num_features, dim, num_gc_layers, device, graph_pooling
        self.gnn = Encoder_Core(num_features, dim, num_gc_layers, device)
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

    def forward(self, x, edge_index, batch, Encoder_support):
        print("fine-model.forward, embedding is done")
        representTu,label = self.gnn.forward(x, edge_index, batch, Encoder_support)
        return representTu
        #return self.classifier(representTu), label

