'''
@Date : 2021/9/10
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
import numpy as np
from itertools import product
NUM_RECS_RANGE = 20

#命中率
def hit(hit_vec_np):
    HRatK = []
    for num_recs in range(5,NUM_RECS_RANGE+1):
        if np.sum(hit_vec_np[:num_recs]) > 0:
            HRatK.append(1)
        else:
            HRatK.append(0)
    return HRatK


def ndcg(hit_vec_np):
    NDCGatK = []
    for num_recs in range(5,NUM_RECS_RANGE + 1):
        hit_vec_atK_np = np.array(hit_vec_np[:num_recs],dtype=np.int)
        hit_vec_atK_np = hit_vec_atK_np.reshape(1,-1)
        NDCGatK.append(np.sum(hit_vec_atK_np)/(np.log2(np.argmax(hit_vec_atK_np)+2)))
    return NDCGatK

def auc(preds_pos, preds_neg):
    product_comp = [1 if p_pred > n_pred else 0 for p_pred, n_pred in product(preds_pos, preds_neg)] #product多个可迭代对象笛卡尔积
    return np.mean(product_comp)