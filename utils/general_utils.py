'''
@Date : 2021/8/12
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
import os
import pickle
import numpy as np
import torch
import gc
import os.path as osp
from dataset import DataProcess

#加载训练结果全局日志
def load_global_logger(global_logger_filepath):
    if os.path.isfile(global_logger_filepath):
        with open(global_logger_filepath,'rb') as f:
            HRs_per_run, NDCGs_per_run, AUC_per_run,train_loss_per_run,eval_loss_per_run = pickle.load(f)
    else:
        print("No loggers found at '{}'".format(global_logger_filepath))
        HRs_per_run, NDCGs_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run = \
            np.zeros((0,16)),np.zeros((0,16)),np.zeros((0,1)),np.zeros((0,1)),np.zeros((0,1))
    return HRs_per_run, NDCGs_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run, HRs_per_run.shape[0]

#反向传播算法
def get_opt_class(opt):
    if opt.lower() == 'adam':
        return torch.optim.Adam
    elif opt.lower() == 'sgd':
        return torch.optim.SGD
    elif opt.lower() == 'sparseadam':
        return torch.optim.SparseAdam
    else:
        raise NotImplementedError('No such optims!')

#从保存的文件加载模型的参数
def load_model(file_path, model, optim, device):
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_states']['model'])
        optim.load_state_dict(checkpoint['optim_states']['optim'])
        rec_metrics = checkpoint['rec_metrics']
        for state in optim.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Loaded checkpoint_backup '{}'".format(file_path))
    else:
        print("No checkpoint_backup found at '{}'".format(file_path))
        epoch = 0
        rec_metrics = np.zeros((0,16)),np.zeros((0,16)), np.zeros((0,1)), np.zeros((0,1)),np.zeros((0,1))
    return model, optim, epoch, rec_metrics

#用于强制写入与给定文件描述符关联的文件
def instantwrite(filename):
    filename.flush()
    os.fsync(filename.fileno())

#清理变量缓存
def clearcache():
    gc.collect() #释放内存
    torch.cuda.empty_cache() #释放显存

#将数据转为图格式
def update_pea_graph_input(dataset_args, train_args, dataset):
    if dataset_args['dataset'] == "Tensorflow":
        issue2developer_edge_index = torch.from_numpy(dataset.edge_index_nps['issue2developer']).long().to(train_args['device'])
        tag2issue = torch.from_numpy(dataset.edge_index_nps['tag2issue']).long().to(train_args['device'])
        source_code2developer = torch.from_numpy(dataset.edge_index_nps['source_code2developer']).long().to(train_args['device'])
        source_code2issue = torch.from_numpy(dataset.edge_index_nps['source_code2issue']).long().to(train_args['device'])

        meta_path_edge_indicis_1 = [issue2developer_edge_index, torch.flip(issue2developer_edge_index, dims=[0])] # flip按照维度对输入进行翻转
        meta_path_edge_indicis_2 = [torch.flip(tag2issue, dims=[0]),tag2issue]
        meta_path_edge_indicis_6 = [torch.flip(source_code2developer,dims=[0]), source_code2issue]
        meta_path_edge_indicis_4 = [torch.flip(issue2developer_edge_index, dims=[0]), issue2developer_edge_index]
        meta_path_edge_indicis_3 = [torch.flip(torch.flip(source_code2issue, dims=[0]), dims=[0]), source_code2developer]
        meta_path_edge_indicis_5 = [source_code2developer, torch.flip(issue2developer_edge_index, dims=[0])] # flip按照维度对输入进行翻转
        meta_path_edge_indicis_7 = [source_code2issue, issue2developer_edge_index]
        meta_path_edge_indicis_list = [meta_path_edge_indicis_1,meta_path_edge_indicis_2,meta_path_edge_indicis_3, meta_path_edge_indicis_4,meta_path_edge_indicis_5]
    elif dataset_args['dataset'] == "Flutter":
        issue2developer_edge_index = torch.from_numpy(dataset.edge_index_nps['issue2developer']).long().to(train_args['device'])
        source_code2developer = torch.from_numpy(dataset.edge_index_nps['source_code2developer']).long().to(train_args['device'])
        source_code2issue = torch.from_numpy(dataset.edge_index_nps['source_code2issue']).long().to(train_args['device'])

        meta_path_edge_indicis_1 = [issue2developer_edge_index, torch.flip(issue2developer_edge_index, dims=[0])] # flip按照维度对输入进行翻转
        meta_path_edge_indicis_6 = [torch.flip(source_code2developer,dims=[0]), source_code2issue]
        meta_path_edge_indicis_4 = [torch.flip(issue2developer_edge_index, dims=[0]), issue2developer_edge_index]
        #meta_path_edge_indicis_3 = [torch.flip(source_code2issue, dims=[0]), source_code2developer]
        meta_path_edge_indicis_5 = [source_code2developer, torch.flip(issue2developer_edge_index, dims=[0])] # flip按照维度对输入进行翻转
        #meta_path_edge_indicis_7 = [source_code2issue, issue2developer_edge_index]
        meta_path_edge_indicis_list = [meta_path_edge_indicis_1, meta_path_edge_indicis_4,meta_path_edge_indicis_5,meta_path_edge_indicis_6]
    return meta_path_edge_indicis_list



#获取文件路径
def get_folder_path(model, dataset, loss_type):
    data_folder = osp.join('data',dataset)
    weights_folder = osp.join('data','weights', dataset, model, loss_type)
    logger_folder = osp.join('data','loggers', dataset, model, loss_type)
    data_folder = osp.expanduser(osp.normpath(data_folder)) #把path中包含的"~"和"~user"转换成用户目录,规范path字符串形式
    weights_folder = osp.expanduser(osp.normpath(weights_folder))
    logger_folder = osp.expanduser(osp.normpath(logger_folder))
    return data_folder, weights_folder, logger_folder

#加载数据集
def load_dataset(dataset_args):
  return DataProcess(**dataset_args)


# 保存训练好的模型
def save_model(file_path, model, optim, epoch, rec_metrics, silent=False):
    model_states = {'model':model.state_dict()}
    optim_states = {'optim':optim.state_dict()}
    states = {
        'epoch': epoch,
        'model_states': model_states,
        'optim_states': optim_states,
        'rec_metrics': rec_metrics
    }
    with open(file_path, mode='wb+') as f:
        torch.save(states, f)
    if not silent:
        print("Saved checkpoint_backup '{}'".format(file_path))

# 保存日志文件
def save_global_logger(global_logger_filepath,
                       HR_per_run, NDCG_per_run, AUC_per_run,
                       train_loss_per_run, eval_loss_per_run):
    with open(global_logger_filepath, 'wb') as f:
        pickle.dump([HR_per_run, NDCG_per_run, AUC_per_run, train_loss_per_run, eval_loss_per_run], f)
