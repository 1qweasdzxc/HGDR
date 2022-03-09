'''
@Date : 2021/11/5
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
import torch
from torch_geometric.nn import GATConv
from torch.nn import functional as F
from torch.nn import Parameter
import torch.nn as nn

from torch_geometric.nn.inits import glorot, zeros
import numpy as np
import random as rd

def cos_sim(vector_a, vector_b):
    """
    :param vector_a: vector a 
    :param vector_b: vector b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return round(cos,4)

class PEABaseChannel(torch.nn.Module):
    def reset_parameters(self):
        for module in self.gnn_layers:
            module.reset_parameters()

    def forward(self, x, edge_index_list):
        assert len(edge_index_list) == self.num_steps

        for step_idx in range(self.num_steps - 1):
            x = F.relu(self.gnn_layers[step_idx](x, edge_index_list[step_idx]))
        x = self.gnn_layers[-1](x, edge_index_list[-1])
        return x

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp_i1, z_mp_i2):
        z_proj_mp_i1= self.proj(z_mp_i1)
        z_proj_mp_i2 = self.proj(z_mp_i2)
        
        matrix_mpi12mpi2 = self.sim(z_proj_mp_i1, z_proj_mp_i2)
        matrix_mpi22mpi1 = matrix_mpi12mpi2.t()
        matrix_mpi12mpi2 = matrix_mpi12mpi2 / (torch.sum(matrix_mpi12mpi2, dim=1).view(-1, 1) + 1e-8)
        ssl_mp = -torch.log(matrix_mpi12mpi2).mean()
        return ssl_mp

class GraphRecsysModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GraphRecsysModel, self).__init__()

        self._init(**kwargs)

        self.reset_parameters()
        self.contrast = Contrast(kwargs['repr_dim'], kwargs['tau'])

    def _init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self, pos_neg_pair_t):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()

    def loss(self,pos_neg_pair_t,**kwargs):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 1])
        neg_pred = self.predict(pos_neg_pair_t[:, 0], pos_neg_pair_t[:, 2])
        cf_loss = -(pos_pred - neg_pred).sigmoid().log().sum()
        # print("cf_loss",cf_loss)
        if self.ssl and self.training:
            mp = rd.choice([2,3])
            #print("mp",mp)
            self.drop = round(rd.uniform(0.1, 0.3),2)
            #print("self.meta_path_edge_index_list[mp][1]",self.meta_path_edge_index_list[mp][1])
            developers_mp1 = self.meta_path_edge_index_list[mp][1][1].cpu().numpy()
            feature_entitys_mp1 = self.meta_path_edge_index_list[mp][1][0].cpu().numpy()
            developers_mp2 = developers_mp1
            feature_entitys_mp2 = feature_entitys_mp1
            batch_developer = rd.choices(list(set(pos_neg_pair_t[:,2].cpu().numpy())),k=100)
            for d in batch_developer:
              try:
                feature_entity_mp1 = feature_entitys_mp1[np.where(developers_mp1==d)]
                feature_entity_mp2 = feature_entitys_mp2[np.where(developers_mp2==d)]
                #print("feature_entity_mp1",feature_entity_mp1)
                sim_i = []
                seed_i = rd.choice(feature_entity_mp1)
                for i in feature_entity_mp1:
                  try:
                    if mp == 3:
                      sim_i.append(cos_sim(self.issues_embedding[seed_i],self.issues_embedding[i]))
                    elif mp == 2:
                      sim_i.append(cos_sim(self.source_code_embedding[seed_i],self.source_code_embedding[i]))
                  except:
                    sim_i.append(0.0)
                sorted_i = sorted(enumerate(sim_i), key=lambda x:x[1], reverse=True)
                li = (int)(len(feature_entity_mp1)/2)
                sorted_imp1 = [i[0] for i in sorted_i[0:li]]
                sorted_imp2 = [i[0] for i in sorted_i[li:]]
                index_d1 = np.where(developers_mp1!=d)
                index_d2 = np.where(developers_mp2!=d)
                developers_mp1 = developers_mp1[index_d1]
                feature_entitys_mp1 = feature_entitys_mp1[index_d1]
                developers_mp2 = developers_mp2[index_d2]
                feature_entitys_mp2 = feature_entitys_mp2[index_d2]
                developers_mp1_mask = rd.sample(sorted_imp1, (int)((1-self.feat_drop)*len(sorted_imp1)))
                developers_mp1 = np.append(developers_mp1,[d]*len(developers_mp1_mask))
                # print(1-self.feat_drop)
                feature_entitys_mp1 = np.append(feature_entitys_mp1, feature_entity_mp1[developers_mp1_mask])
                developers_mp2_mask = rd.sample(sorted_imp1, (int)((1-self.feat_drop)*len(sorted_imp2)))
                developers_mp2 = np.append(developers_mp2,[d]*len(developers_mp2_mask))
                feature_entitys_mp2 = np.append(feature_entitys_mp2,feature_entity_mp2[developers_mp2_mask])
              #   print("feature_entitys_mp1",feature_entitys_mp1)
              #   print()
              except:
                continue
            
            # print(torch.from_numpy(np.array([developers_mp1,issues_mp1])).long().cuda())
            self.meta_path_edge_index_list[mp][0].data = torch.from_numpy(np.array([developers_mp1,feature_entitys_mp1])).long().cuda()
            self.meta_path_edge_index_list[mp][1].data = torch.flip(torch.from_numpy(np.array([developers_mp1,feature_entitys_mp1])).long().cuda(), dims=[0])
            
            self.cached_repr1 = self.forward()
            #print("self.cached_repr1",self.cached_repr1)
            self.drop = round(rd.uniform(0.1, 0.3),2)
            # print("self.drop",self.drop)
            # print(np.array([developers_mp2,issues_mp2]))
            self.meta_path_edge_index_list[mp][0].data = torch.from_numpy(np.array([developers_mp2,feature_entitys_mp2])).long().cuda()
            self.meta_path_edge_index_list[mp][1].data = torch.flip(torch.from_numpy(np.array([developers_mp2,feature_entitys_mp2])).long().cuda(), dims=[0])
            self.cached_repr2 = self.forward()
            #print("self.cached_repr2",self.cached_repr2)
            #print("self.cached_repr1.shape",self.cached_repr1.shape)
            # print("list(set(pos_neg_pair_t[:, 2]))",list(set(pos_neg_pair_t[:, 2]))[0:100])
            # print("self.cached_repr1[list(set(pos_neg_pair_t[:, 2]))]",self.cached_repr1[list(set(pos_neg_pair_t[:, 2]))[0:100]].shape)
            # print("self.cached_repr1[list(set(pos_neg_pair_t[:, 2]))]",self.cached_repr1[list(set(pos_neg_pair_t[:, 2]))[0:100]])
            ssl_loss = self.contrast(self.cached_repr1[list(set(pos_neg_pair_t[:, 2]))], self.cached_repr2[list(set(pos_neg_pair_t[:, 2]))])
            loss = cf_loss + self.ssl_coff * ssl_loss
            #print("ssl_loss",ssl_loss)
        else:
            loss = cf_loss

        return loss

    def update_graph_input(self, dataset):
        raise NotImplementedError

    def predict(self, unids, inids):
        raise NotImplementedError

    def eval(self, metapath_idx=None):
        super(GraphRecsysModel, self).eval()
        if self.__class__.__name__ not in ['KGATRecsysModel', 'KGCNRecsysModel']:
          with torch.no_grad():
              self.cached_repr = self.forward()


class PEABaseRecsysModel(GraphRecsysModel):
    def __init__(self, **kwargs):
        super(PEABaseRecsysModel, self).__init__(**kwargs)

    def _init(self, **kwargs):
        self.ssl = kwargs['ssl']
        self.ssl_coff = kwargs['ssl_coff']
        self.side_info = kwargs['side_info']
        self.word_dims = kwargs['side_info_vector_size']
        self.e2nid_dict = kwargs['dataset']['e2nid_dict']
        self.issues_embedding = kwargs['dataset']['issues_embedding']
        self.commit_embedding = kwargs['dataset']['commit_embedding']
        self.source_code_embedding = kwargs['dataset']['source_code_embedding']
        self.emb_dim = kwargs['emb_dim']
        self.drop = kwargs['dropout']
        self.feat_drop = kwargs['feat_drop']
        self.meta_path_steps = kwargs['meta_path_steps']
        self.if_use_features = kwargs['if_use_features']
        self.channel_aggr = kwargs['channel_aggr']

        # Create node embedding
        if not self.if_use_features and not self.side_info:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], self.emb_dim))
        # side information enhancement
        elif not self.if_use_features and self.side_info:
          if not kwargs['emb_cat']:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], self.emb_dim))
            for iid in kwargs['dataset']['unique_iids']:
              try:
                self.x.data[self.e2nid_dict['iid'][iid],:] = torch.from_numpy(self.issues_embedding[iid])
              except:
                continue
            for did in kwargs['dataset']['unique_dids']:
                try:
                    if did in self.commit_embedding.keys():
                        self.x.data[self.e2nid_dict['did'][did],:] = torch.from_numpy(self.issues_embedding[did])
                except:
                    continue
        else:
            self.x = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], self.emb_dim))
            for iid in kwargs['dataset']['unique_iids']:
              self.x.data[self.e2nid_dict['iid'][iid],:] = torch.from_numpy(self.issues_embedding[iid])
            for did in kwargs['dataset']['unique_dids']:
              try:
                  if did in self.commit_embedding.keys():
                      self.x.data[self.e2nid_dict['did'][did],:] = torch.from_numpy(self.issues_embedding[did])
              except:
                  continue
            self.x.data.requires_grad = False #词向量，因此是不可学习
            self.x1 = Parameter(torch.Tensor(kwargs['dataset']['num_nodes'], self.emb_dim))
            self.x = Parameter(torch.cat((self.x,self.x1),dim=1))

        # Create graphs
        meta_path_edge_index_list = self.update_graph_input(kwargs['dataset'])
        assert len(meta_path_edge_index_list) == len(kwargs['meta_path_steps'])
        self.meta_path_edge_index_list = meta_path_edge_index_list

        # Create channels
        self.pea_channels = torch.nn.ModuleList()
        for num_steps in kwargs['meta_path_steps']:
            kwargs_cpy = kwargs.copy()
            kwargs_cpy['num_steps'] = num_steps
            self.pea_channels.append(kwargs_cpy['channel_class'](**kwargs_cpy))

        if self.channel_aggr == 'att':
            self.att = Parameter(torch.Tensor(1, len(kwargs['meta_path_steps']), kwargs['repr_dim']))

        if self.channel_aggr == 'cat':
            self.fc1 = torch.nn.Linear(2 * len(kwargs['meta_path_steps']) * kwargs['repr_dim'], kwargs['repr_dim'])
        else:
            self.fc1 = torch.nn.Linear(2 * kwargs['repr_dim'], kwargs['repr_dim'])
        self.fc2 = torch.nn.Linear(kwargs['repr_dim'], 1)

    def reset_parameters(self):
        if not self.if_use_features:
            glorot(self.x)
        for module in self.pea_channels:
            module.reset_parameters()
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)
        if self.channel_aggr == 'att':
            glorot(self.att)

    def forward(self, metapath_idx=None):
        if self.drop > 0:
            self.dropout = nn.Dropout(self.drop)
        else:
            self.dropout = lambda x: x
        x = self.x
        x = [self.dropout(module(x, self.meta_path_edge_index_list[idx])).unsqueeze(1) for idx, module in enumerate(self.pea_channels)]
        if metapath_idx is not None:
            x[metapath_idx] = torch.zeros_like(x[metapath_idx])
        x = torch.cat(x, dim=1)
        if self.channel_aggr == 'concat':
            x = x.view(x.shape[0], -1)
        elif self.channel_aggr == 'mean':
            x = x.mean(dim=1)
        elif self.channel_aggr == 'att':
            atts = F.softmax(torch.sum(x * self.att, dim=-1), dim=-1).unsqueeze(-1)
            x = torch.sum(x * atts, dim=1)
        else:
            raise NotImplemented('Other aggr methods not implemeted!')
        return x

    def predict(self, unids, inids):
        u_repr = self.cached_repr[unids.type(torch.int64),:]
        i_repr = self.cached_repr[inids.type(torch.int64),:]
        x = torch.cat([u_repr, i_repr], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PEAGATChannel(PEABaseChannel):
    def __init__(self, **kwargs):
        super(PEAGATChannel, self).__init__()
        self.num_steps = kwargs['num_steps']
        self.num_nodes = kwargs['num_nodes']
        self.dropout = kwargs['dropout']

        self.gnn_layers = torch.nn.ModuleList()
        if kwargs['num_steps'] == 1:
            self.gnn_layers.append(GATConv(kwargs['emb_dim'], kwargs['repr_dim'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
        else:
            self.gnn_layers.append(GATConv(kwargs['emb_dim'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            for i in range(kwargs['num_steps'] - 2):
                self.gnn_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['hidden_size'], heads=kwargs['num_heads'], dropout=kwargs['dropout']))
            self.gnn_layers.append(GATConv(kwargs['hidden_size'] * kwargs['num_heads'], kwargs['repr_dim'], heads=1, dropout=kwargs['dropout']))

        self.reset_parameters()


class PEAGATRecsysModel(PEABaseRecsysModel):
    def __init__(self, **kwargs):
        kwargs['channel_class'] = PEAGATChannel
        super(PEAGATRecsysModel, self).__init__(**kwargs)