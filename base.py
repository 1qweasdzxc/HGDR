'''
@Date : 2021/8/19
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
import torch
from torch.nn import functional as F
from torch.nn import Parameter

class GraphRecsysModel(torch.nn.Module):
    def __init__(self,**kwargs):
        super(GraphRecsysModel,self).__init__()
        self._init(**kwargs)
        self.reset_parameters()

    def _init(self,**kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def loss(self,pos_neg_pair_t):
        if self.training:
            self.cached_repr = self.forward()
        pos_pred = self.predict(pos_neg_pair_t[:,0],pos_neg_pair_t[:,1])
        neg_pred = self.predict(pos_neg_pair_t[:,0],pos_neg_pair_t[:,2])
        cf_loss = -(pos_pred-neg_pred).sigmoid().log().sum()

        # if self.entity_aware and self.training:
        #     pos_item_entity,neg_item_entity = pos_neg_pair_t[:,3],pos_neg_pair_t[:,4]
        #     pos_user_entity, neg_user_entity = pos_neg_pair_t[:, 6], pos_neg_pair_t[:, 7]
        #     item_entity_mask, user_entity_mask = pos_neg_pair_t[:, 5], pos_neg_pair_t[:, 8]
        loss = cf_loss
