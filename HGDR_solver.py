'''
@Date : 2021/11/5
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
import argparse
import torch
import os
import sys

from models import PEAGATRecsysModel
from utils import get_folder_path, update_pea_graph_input
from solver import  BaseSolver

MODEL_TYPE = 'Graph'
LOSS_TYPE = 'BPR'
MODEL = 'HGDR'
GRAPH_TYPE = 'hete'

parser = argparse.ArgumentParser()
# Dataset params
parser.add_argument('--dataset', type=str, default='Flutter', help='')		#Tensorflow
parser.add_argument('--if_use_features', type=str, default='false', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='random', help='') # unseen/random
parser.add_argument('--entity_aware', type=str, default='false', help='')
parser.add_argument('--side_info',type=str, default='true',help='') #commit text and issue text
parser.add_argument('--side_info_vector_size',type=int, default=64,help='')
parser.add_argument('--side_info_window',type=int, default=8,help='')
parser.add_argument('--side_info_min_count',type=int,default=3,help='')
parser.add_argument('--side_info_epoch',type=int,default=8,help='')


# Model params
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--num_heads', type=int, default=1, help='')
parser.add_argument('--repr_dim', type=int, default=16, help='')
parser.add_argument('--hidden_size', type=int, default=64, help='')
parser.add_argument('--meta_path_steps', type=str, default='2,2,2,2,2', help='')	#2,2,2,2,2,2,2,2,2(for small) #2,2,2,2,2,2,2,2,2,2,2,2,2(for 25m) #2,2,2,2,2,2,2,2,2,2,2 (for yelp)
parser.add_argument('--channel_aggr', type=str, default='att', help='')
parser.add_argument('--ssl', type=str, default='true', help='')
parser.add_argument('--ssl_coff', type=float, default=0.1, help='')
parser.add_argument('--feat_drop', type=float, default=0.2)
parser.add_argument('--tau', type=float, default=0.5, help='') ## ssl


# Train params
parser.add_argument('--init_eval', type=str, default='true', help='')
parser.add_argument('--num_negative_samples', type=int, default=6, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cuda', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=1, help='')             #5(for Tensorflow)
parser.add_argument('--epochs', type=int, default=30, help='')          #30(for Tensorflow)
parser.add_argument('--batch_size', type=int, default=1024, help='')    #1024(for Tensorflow)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='10,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=15, help='')        #15(for Tensorflow)
parser.add_argument('--metapath_test', type=str, default='true', help='')
parser.add_argument('--emb_cat', type=str, default='false', help='')
parser.add_argument('--head_tail_test', type=str, default='false', help='')

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset, loss_type=LOSS_TYPE)

# Setup device
if not torch.cuda.is_available() or args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset,
    'if_use_features': args.if_use_features.lower() == 'true',
    'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core,
    'num_feat_core': args.num_feat_core,
    'cf_loss_type': LOSS_TYPE, 'type': GRAPH_TYPE,
    'sampling_strategy': args.sampling_strategy,
    'ssl': args.entity_aware.lower() == 'true',
    'side_info': args.side_info.lower() == 'true',
    'side_info_vector_size': args.side_info_vector_size,
    'side_info_window': args.side_info_window,
    'side_info_min_count': args.side_info_min_count,
    'side_info_epoch': args.side_info_epoch,
    'model': MODEL
}
model_args = {
    'model_type': MODEL_TYPE,
    'if_use_features': args.if_use_features.lower() == 'true',
    'emb_dim': args.emb_dim,
    'hidden_size': args.hidden_size,
    'repr_dim': args.repr_dim,
    'dropout': args.dropout,
    'num_heads': args.num_heads,
    'meta_path_steps': [int(i) for i in args.meta_path_steps.split(',')],
    'channel_aggr': args.channel_aggr,
    'side_info':args.side_info.lower() == 'true',
    'side_info_vector_size': args.side_info_vector_size,
    'emb_cat': args.emb_cat == 'true',
    'ssl': args.ssl.lower() == 'true',
    'ssl_coff': args.ssl_coff,
    'feat_drop': args.feat_drop,
    'tau': args.tau
}
path_args = model_args.copy()
path_args['meta_path_steps'] = len(path_args['meta_path_steps'])
train_args = {
    'init_eval': args.init_eval.lower() == 'true',
    'num_negative_samples': args.num_negative_samples,
    'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,
    'device': device,
    'lr': args.lr,
    'num_workers': args.num_workers,
    'weights_folder': os.path.join(weights_folder),
    'logger_folder': os.path.join(logger_folder),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')],
    'save_every_epoch': args.save_every_epoch,
    'metapath_test': args.metapath_test.lower() == 'true',
    'head_tail_test': args.head_tail_test.lower() == 'true'
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))
class HGDRRecsysModel(PEAGATRecsysModel):
    def update_graph_input(self, dataset):
        return update_pea_graph_input(dataset_args, train_args, dataset)

if __name__ == '__main__':
    if MODEL == 'HGDR':
        solver = BaseSolver(HGDRRecsysModel, dataset_args, model_args, train_args)
    solver.run()