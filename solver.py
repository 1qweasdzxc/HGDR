'''
@Date : 2021/8/9
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
import os
import random as rd
import numpy as np
import torch
import time
import pandas as pd
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import *



class BaseSolver(object):
    def __init__(self, model_class, dataset_args, model_args, train_args):
        self.dataset_args = dataset_args
        self.model_args = model_args
        self.train_args = train_args
        self.model_class = model_class

    def generate_candidates(self, dataset, i_nid):
        """
        Return the recommendation candidates to the algorithms to rank
        """
        pos_d_nids = dataset.test_pos_inid_dnid_map[i_nid]
        neg_d_nids = list(
            np.random.choice(dataset.neg_inid_dnid_map[i_nid], size=(self.train_args['num_neg_candidates'],)))

        return pos_d_nids, neg_d_nids

    def metrics(self, run, epoch, model, dataset):
        """
        Generate the positive and negative candidates for the recsys evaluation
        """
        HRs, NDCGs, AUC, eval_losses = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1))
        if self.train_args['head_tail_test']:
              HRs_head,  NDCGs_head, AUC_head, eval_losses_head = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1))
              HRs_tail, NDCGs_tail, AUC_tail, eval_losses_tail = np.zeros((0, 16)), np.zeros((0, 16)), np.zeros((0, 1)), np.zeros((0, 1))
        test_pos_inid_dnid_map, neg_inid_dnid_map = \
            dataset.test_pos_inid_dnid_map, dataset.neg_inid_dnid_map
        i_nids = list(test_pos_inid_dnid_map.keys())
        test_bar = tqdm.tqdm(i_nids, total=len(i_nids))
        
        # 从测试集中抽样正负样本
        for i_idx, i_nid in enumerate(test_bar):
          pos_d_nids, neg_d_nids = self.generate_candidates(dataset, i_nid)
          if len(pos_d_nids) == 0 or len(neg_d_nids) == 0:
              raise ValueError("No pos or neg samples found in evaluation!")
          pos_d_nid_df = pd.DataFrame({'i_nid': [i_nid for _ in range(len(pos_d_nids))], 'pos_i_nid': pos_d_nids})
          neg_d_nid_df = pd.DataFrame({'i_nid': [i_nid for _ in range(len(neg_d_nids))], 'neg_i_nid': neg_d_nids})
          pos_neg_pair_t = torch.from_numpy(
              pd.merge(pos_d_nid_df, neg_d_nid_df, how='inner', on='i_nid').to_numpy()).to(self.train_args['device'])
          loss = model.loss(pos_neg_pair_t).detach().cpu().item()
          # pos_i_nids_t = torch.from_numpy(np.array(pos_d_nids)).to(self.train_args['device'])
          # neg_d_nids_t = torch.from_numpy(np.array([i_nid for _ in range(len(neg_d_nids))])).to(
          #     self.train_args['device'])
          pos_i_nids_t = torch.from_numpy(np.array([i_nid for _ in range(len(pos_d_nids))])).to(self.train_args['device'])
          pos_d_nids_t = torch.from_numpy(np.array(pos_d_nids)).to(self.train_args['device'])
          neg_i_nids_t = torch.from_numpy(np.array([i_nid for _ in range(len(neg_d_nids))])).to(
              self.train_args['device'])
          neg_d_nids_t = torch.from_numpy(np.array(neg_d_nids)).to(self.train_args['device'])
          pos_pred = model.predict(pos_i_nids_t,pos_d_nids_t).reshape(-1) # 经过多层感知机得到的分数
          neg_pred = model.predict(neg_i_nids_t,neg_d_nids_t).reshape(-1)

          _, indices = torch.sort(torch.cat([pos_pred,neg_pred]),descending=True) #descending=True 从大到小排序,dim=1 按行排序，默认
          hit_vec = (indices < len(pos_d_nids)).cpu().detach().numpy()#后面当我们进行反向传播时，到该调用detach()的tensor就会停止，不能再继续向前进行传播
          pos_pred = pos_pred.cpu().detach().numpy()
          neg_pred = neg_pred.cpu().detach().numpy()
          if self.train_args['head_tail_test']:
              if pos_d_nids in dataset.head_developers:
                HRs_head = np.vstack([HRs_head,hit(hit_vec)])
                # print("recall(hit_vec):",recall(hit_vec))
                # print("Recalls:",Recalls)
                NDCGs_head = np.vstack([NDCGs_head,ndcg(hit_vec)])
                AUC_head = np.vstack([AUC_head,auc(pos_pred,neg_pred)])
                eval_losses_head = np.vstack([eval_losses_head,loss])
                test_bar.set_description('Run {}, epoch: {}, HR_head@5: {:.4f}, HR_head@10: {:.4f}, HR_head@15: {:.4f}, HR_head@20: {:.4f}, '  
                                        'NDCG_head@5: {:.4f}, NDCG_head@10: {:.4f}, NDCG_head@15: {:.4f}, NDCG_head@20: {:.4f}, AUC: {:.4f}, eval loss: {:.4f} '.format(
                                        run, epoch, HRs_head.mean(axis=0)[0], HRs_head.mean(axis=0)[5], HRs_head.mean(axis=0)[10], HRs_head.mean(axis=0)[15],
                                        NDCGs_head.mean(axis=0)[0], NDCGs_head.mean(axis=0)[5], NDCGs_head.mean(axis=0)[10], NDCGs_head.mean(axis=0)[15],
                                        AUC_head.mean(axis=0)[0], eval_losses_head.mean(axis=0)[0])
                )

              else:
                HRs_tail = np.vstack([HRs_tail,hit(hit_vec)])
                # print("recall(hit_vec):",recall(hit_vec))
                # print("Recalls:",Recalls)
                NDCGs_tail = np.vstack([NDCGs_tail,ndcg(hit_vec)])
                AUC_tail = np.vstack([AUC_tail,auc(pos_pred,neg_pred)])
                eval_losses_tail = np.vstack([eval_losses_tail,loss])
                test_bar.set_description('Run {}, epoch: {}, HR_tail@5: {:.4f}, HR_tail@10: {:.4f}, HR_tail@15: {:.4f}, HR_tail@20: {:.4f}, '
                                        'NDCG_tail@5: {:.4f}, NDCG_tail@10: {:.4f}, NDCG_tail@15: {:.4f}, NDCG_tail@20: {:.4f}, AUC: {:.4f}, eval loss: {:.4f} '.format(
                                        run, epoch, HRs_tail.mean(axis=0)[0], HRs_tail.mean(axis=0)[5], HRs_tail.mean(axis=0)[10], HRs_tail.mean(axis=0)[15],
                                        NDCGs_tail.mean(axis=0)[0], NDCGs_tail.mean(axis=0)[5], NDCGs_tail.mean(axis=0)[10], NDCGs_tail.mean(axis=0)[15],
                                        AUC_tail.mean(axis=0)[0], eval_losses_tail.mean(axis=0)[0])
                )
          else:
           
            HRs = np.vstack([HRs,hit(hit_vec)])
            NDCGs = np.vstack([NDCGs,ndcg(hit_vec)])
            AUC = np.vstack([AUC,auc(pos_pred,neg_pred)])
            eval_losses = np.vstack([eval_losses,loss])
            test_bar.set_description('Run {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                     'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, eval loss: {:.4f} '.format(
                                     run, epoch, HRs.mean(axis=0)[0], HRs.mean(axis=0)[5], HRs.mean(axis=0)[10], HRs.mean(axis=0)[15],
                                     NDCGs.mean(axis=0)[0], NDCGs.mean(axis=0)[5], NDCGs.mean(axis=0)[10], NDCGs.mean(axis=0)[15],
                                     AUC.mean(axis=0)[0], eval_losses.mean(axis=0)[0])
            )
        if self.train_args['head_tail_test']:
          return np.mean(HRs_head, axis=0), np.mean(NDCGs_head, axis=0), np.mean(AUC_head, axis=0), np.mean(eval_losses_head, axis=0),np.mean(HRs_tail, axis=0), np.mean(NDCGs_tail, axis=0), np.mean(AUC_tail, axis=0), np.mean(eval_losses_tail, axis=0)
        else:
          return np.mean(HRs, axis=0), np.mean(NDCGs, axis=0), np.mean(AUC, axis=0), np.mean(eval_losses, axis=0)





    def run(self):
        #日志
        global_logger_path = self.train_args['logger_folder']
        if not os.path.exists(global_logger_path):
            os.makedirs(global_logger_path, exist_ok=True)
        global_logger_file_path = os.path.join(global_logger_path, 'global_logger.pkl')
        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np, last_run = \
            load_global_logger(global_logger_file_path)

        #待加入我自己的数据集
        dataset = load_dataset(self.dataset_args)

        logger_file_path = os.path.join(global_logger_path, 'logger_file.txt')
        with open(logger_file_path, 'a') as logger_file:
            start_run = last_run + 1
            if start_run <= self.train_args['runs']:
                for run in range(start_run, self.train_args['runs'] + 1):
                    # Fix the random seed
                    seed = 2021 + run
                    rd.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True

                    # Create model and optimizer
                    if self.model_args['model_type'] == 'Graph':
                        if self.model_args['if_use_features']:
                            self.model_args['emb_dim'] = dataset.data.x.shape[1]
                        self.model_args['num_nodes'] = dataset.num_nodes
                        self.model_args['dataset'] = dataset

                    model = self.model_class(**self.model_args).to(self.train_args['device'])
                    opt_class = get_opt_class(self.train_args['opt'])
                    optimizer = opt_class(
                        params=model.parameters(),
                        lr=self.train_args['lr'],
                        weight_decay=self.train_args['weight_decay']
                    )

                    # Load models
                    weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path, exist_ok=True)
                    weights_file = os.path.join(weights_path, 'lastest.pkl')
                    model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                           self.train_args['device'])
                    HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np, train_loss_per_run_np, eval_loss_per_run_np = \
                        rec_metrics
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # 同步程序执行

                    start_epoch = last_epoch + 1
                    # if start_epoch == 1 and self.train_args['init_eval']:
                    #     model.eval()  # 防止test会改变权值
                    #     with torch.no_grad():
                    #         HRs_before_np, NDCGs_before_np, AUC_before_np, cf_eval_loss_before_np = \
                    #             self.metrics(run, 0, model, dataset)
                    #     print(
                    #         'Initial performance HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                    #         'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, '
                    #         'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                    #             HRs_before_np[0], HRs_before_np[5], HRs_before_np[10], HRs_before_np[15],
                    #             NDCGs_before_np[0], NDCGs_before_np[5], NDCGs_before_np[10], NDCGs_before_np[15],
                    #             AUC_before_np[0], cf_eval_loss_before_np[0]))
                    #     logger_file.write(
                    #         'Initial performance HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '                             
                    #         'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, '                             
                    #         'AUC: {:.4f}, eval loss: {:.4f} \n'.format(
                    #             HRs_before_np[0], HRs_before_np[5], HRs_before_np[10], HRs_before_np[15],
                    #             NDCGs_before_np[0], NDCGs_before_np[5], NDCGs_before_np[10], NDCGs_before_np[15],
                    #             AUC_before_np[0], cf_eval_loss_before_np[0]))
                    #     instantwrite(logger_file)
                    #     clearcache()

                    t_start = time.perf_counter()
                    if start_epoch <= self.train_args['epochs']:
                        # Start training model
                        for epoch in range(start_epoch, self.train_args['epochs']+ 1):
                             loss_per_batch = []
                             model.train()
                             # 待加入负采样策略算法，防止图数据过大
                             dataset.cf_negative_sampling()
                             train_dataloader = DataLoader(dataset, shuffle=True,
                                                           batch_size=self.train_args['batch_size'],
                                                           num_workers=self.train_args['num_workers'])
                             train_bar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))

                             for _,batch in enumerate(train_bar):
                                 if self.model_args['model_type'] == 'MF':
                                     if self.model_args['loss_type'] == 'BPR':
                                         batch[:, 0] -= dataset.e2nid_dict['uid'][0]
                                         batch[:, 1] -= dataset.e2nid_dict['iid'][0]
                                 batch = batch.to(self.train_args['device'])

                                 optimizer.zero_grad()
                                 loss = model.loss(batch)
                                 loss.backward()
                                 optimizer.step()

                                 loss_per_batch.append(loss.detach().cpu().item())
                                 train_loss = np.mean(loss_per_batch)
                                 train_bar.set_description('Run: {}, epoch: {}, train loss: {:.4f}'.format(run, epoch, train_loss))

                             if model.__class__.__name__[:3]== 'PEA' and self.train_args['metapath_test']:
                                 if self.dataset_args['dataset']=='Tensorflow':
                                     for metapath_idx in range(len(self.model_args['meta_path_steps'])):
                                         model.eval(metapath_idx)
                                         HRs, NDCGs, AUC, eval_loss = self.metrics(run, epoch, model, dataset)
                                         print(
                                             'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                             'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                             'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                                 run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0],
                                                 NDCGs[5], NDCGs[10], NDCGs[15],
                                                 AUC[0], train_loss, eval_loss[0]
                                             )
                                         )

                                         logger_file.write(
                                             'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                             'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                             'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                                 run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0],
                                                 NDCGs[5], NDCGs[10], NDCGs[15], AUC[0], train_loss, eval_loss[0]
                                             ))
                             model.eval()
                             with torch.no_grad():
                                if self.train_args['head_tail_test']:
                                  HRs_head, NDCGs_head, AUC_head, eval_loss_head,HRs_tail,  NDCGs_tail, AUC_tail, eval_loss_tail = self.metrics(run, epoch, model, dataset)
                                  print(
                                  'Run: {}, epoch: {}, HR_head@5: {:.4f}, HR_head@10: {:.4f}, HR_head@15: {:.4f}, HR_head@20: {:.4f}, '
                                  'NDCG_head@5: {:.4f}, NDCG_head@10: {:.4f}, NDCG_head@15: {:.4f}, NDCG_head@20: {:.4f}, AUC_head: {:.4f}, '
                                  'HR_tail@5: {:.4f}, HR_tail@10: {:.4f}, HR_tail@15: {:.4f}, HR_tail@20: {:.4f}, '
                                  'NDCG_tail@5: {:.4f}, NDCG_tail@10: {:.4f}, NDCG_tail@15: {:.4f}, NDCG_tail@20: {:.4f}, AUC_tail: {:.4f}, '
                                  'train_tail loss: {:.4f}, eval_head loss: {:.4f},eval_tail loss: {:.4f} \n'.format(
                                      run, epoch,  HRs_head[0], HRs_head[5], HRs_head[10], HRs_head[15], NDCGs_head[0],
                                      NDCGs_head[5], NDCGs_head[10], NDCGs_head[15],
                                      AUC_head[0],HRs_tail[0], HRs_tail[5], HRs_tail[10], HRs_tail[15],  NDCGs_tail[0],
                                      NDCGs_tail[5], NDCGs_tail[10], NDCGs_tail[15],
                                      AUC_tail[0], train_loss, eval_loss_head[0],eval_loss_tail[0]
                                  )
                                  )

                                  logger_file.write(
                                      'Run: {}, epoch: {}, HR_head@5: {:.4f}, HR_head@10: {:.4f}, HR_head@15: {:.4f}, HR_head@20: {:.4f}, '
                                  'NDCG_head@5: {:.4f}, NDCG_head@10: {:.4f}, NDCG_head@15: {:.4f}, NDCG_head@20: {:.4f}, AUC_head: {:.4f}, '
                                  'HR_tail@5: {:.4f}, HR_tail@10: {:.4f}, HR_tail@15: {:.4f}, HR_tail@20: {:.4f}, '
                                  'NDCG_tail@5: {:.4f}, NDCG_tail@10: {:.4f}, NDCG_tail@15: {:.4f}, NDCG_tail@20: {:.4f}, AUC_tail: {:.4f}, '
                                  'train_tail loss: {:.4f}, eval_head loss: {:.4f},eval_tail loss: {:.4f} \n'.format(
                                      run, epoch,  HRs_head[0], HRs_head[5], HRs_head[10], HRs_head[15], NDCGs_head[0],
                                      NDCGs_head[5], NDCGs_head[10], NDCGs_head[15],
                                      AUC_head[0], HRs_tail[0], HRs_tail[5], HRs_tail[10], HRs_tail[15], NDCGs_tail[0],
                                      NDCGs_tail[5], NDCGs_tail[10], NDCGs_tail[15],
                                      AUC_tail[0], train_loss, eval_loss_head[0],eval_loss_tail[0]
                                  )
                                  )
                                else:
                                  HRs, NDCGs, AUC, eval_loss = self.metrics(run, epoch, model, dataset)

                                  # Sumarize the epoch
                                  HRs_per_epoch_np = np.vstack([HRs_per_run_np, HRs])
                                  NDCGs_per_epoch_np = np.vstack([NDCGs_per_run_np, NDCGs])
                                  AUC_per_epoch_np = np.vstack([AUC_per_run_np, AUC])
                                  train_loss_per_epoch_np = np.vstack([train_loss_per_run_np, np.array([train_loss])])
                                  eval_loss_per_epoch_np = np.vstack([eval_loss_per_run_np, np.array([eval_loss])])

                                  if epoch in self.train_args['save_epochs']:
                                      weightpath = os.path.join(weights_path, '{}.pkl'.format(epoch))
                                      save_model(
                                          weightpath,
                                          model, optimizer, epoch,
                                          rec_metrics=(
                                              HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np,
                                              train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                      )
                                  if epoch > self.train_args['save_every_epoch']:
                                      weightpath = os.path.join(weights_path, 'latest.pkl')
                                      save_model(
                                          weightpath,
                                          model, optimizer, epoch,
                                          rec_metrics=(
                                              HRs_per_epoch_np, NDCGs_per_epoch_np, AUC_per_epoch_np,
                                              train_loss_per_epoch_np, eval_loss_per_epoch_np)
                                      )
                                  print(
                                      'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                      'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                      'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                          run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                                          NDCGs[15],
                                     AUC[0], train_loss, eval_loss[0]
                                 )
                                  )
                                  logger_file.write(
                                      'Run: {}, epoch: {}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                                      'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                                      'train loss: {:.4f}, eval loss: {:.4f} \n'.format(
                                          run, epoch, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                                          NDCGs[15],
                                          AUC[0], train_loss, eval_loss[0]
                                      )
                                  )
                             instantwrite(logger_file)
                             clearcache()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    t_end = time.perf_counter()

                    HRs_per_run_np = np.vstack([HRs_per_run_np, np.max(HRs_per_epoch_np, axis=0)])
                    NDCGs_per_run_np = np.vstack([NDCGs_per_run_np, np.max(NDCGs_per_epoch_np, axis=0)])
                    AUC_per_run_np = np.vstack([AUC_per_run_np, np.max(AUC_per_epoch_np, axis=0)])
                    train_loss_per_run_np = np.vstack([train_loss_per_run_np, np.mean(train_loss_per_epoch_np, axis=0)])
                    eval_loss_per_run_np = np.vstack([eval_loss_per_run_np, np.mean(eval_loss_per_epoch_np, axis=0)])

                    save_global_logger(
                        global_logger_file_path,
                        HRs_per_run_np, NDCGs_per_run_np, AUC_per_run_np,
                        train_loss_per_run_np, eval_loss_per_run_np
                    )
                    print(
                        'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0], np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                            np.max(NDCGs_per_epoch_np, axis=0)[10],
                            np.max(NDCGs_per_epoch_np, axis=0)[15], np.max(AUC_per_epoch_np, axis=0)[0],
                            train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                    )
                    logger_file.write(
                        'Run: {}, Duration: {:.4f}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                        'train_loss: {:.4f}, eval loss: {:.4f}\n'.format(
                            run, t_end - t_start, np.max(HRs_per_epoch_np, axis=0)[0], np.max(HRs_per_epoch_np, axis=0)[5],
                            np.max(HRs_per_epoch_np, axis=0)[10], np.max(HRs_per_epoch_np, axis=0)[15],
                            np.max(NDCGs_per_epoch_np, axis=0)[0], np.max(NDCGs_per_epoch_np, axis=0)[5],
                            np.max(NDCGs_per_epoch_np, axis=0)[10], np.max(NDCGs_per_epoch_np, axis=0)[15],
                            np.max(AUC_per_epoch_np, axis=0)[0],
                            train_loss_per_epoch_np[-1][0], eval_loss_per_epoch_np[-1][0])
                    )
                    instantwrite(logger_file)
                    del model, optimizer, loss, loss_per_batch, rec_metrics, train_dataloader
                    clearcache()


            if self.dataset_args['model'][:3] == 'HGD' and self.train_args['metapath_test']:
                run = 1
                if self.dataset_args['dataset'] == 'Tensorflow':
                    epoch = 30
                seed = 2019 + run
                rd.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                # Create model and optimizer
                if self.model_args['model_type'] == 'Graph':
                    if self.model_args['if_use_features']:
                        self.model_args['emb_dim'] = dataset.data.x.shape[1]
                    self.model_args['num_nodes'] = dataset.num_nodes
                    self.model_args['dataset'] = dataset
                elif self.model_args['model_type'] == 'MF':
                    self.model_args['num_users'] = dataset.num_uids
                    self.model_args['num_items'] = dataset.num_iids

                model = self.model_class(**self.model_args).to(self.train_args['device'])

                opt_class = get_opt_class(self.train_args['opt'])
                optimizer = opt_class(
                    params=model.parameters(),
                    lr=self.train_args['lr'],
                    weight_decay=self.train_args['weight_decay']
                )

                # Load models
                weights_path = os.path.join(self.train_args['weights_folder'], 'run_{}'.format(str(run)))
                if not os.path.exists(weights_path):
                    os.makedirs(weights_path, exist_ok=True)
                weights_file = os.path.join(weights_path, 'latest.pkl')
                model, optimizer, last_epoch, rec_metrics = load_model(weights_file, model, optimizer,
                                                                       self.train_args['device'])
                for metapath_idx in range(len(self.model_args['meta_path_steps'])):
                    model.eval(metapath_idx)
                    HRs, NDCGs, AUC, eval_loss = self.metrics(run, epoch, model, dataset)
                    print(
                        'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                        '\n'.format(
                            run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                            NDCGs[15],
                            AUC[0]
                        )
                    )
                    logger_file.write(
                        'Run: {}, epoch: {}, exclude path:{}, HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                        'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, '
                        '\n'.format(
                            run, epoch, metapath_idx, HRs[0], HRs[5], HRs[10], HRs[15], NDCGs[0], NDCGs[5], NDCGs[10],
                            NDCGs[15],
                            AUC[0]
                        )
                    )
                    instantwrite(logger_file)

            print(
                'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                    HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                    HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                    NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                    NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                    train_loss_per_run_np.mean(axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                )
            )
            logger_file.write(
                'Overall HR@5: {:.4f}, HR@10: {:.4f}, HR@15: {:.4f}, HR@20: {:.4f}, '
                'NDCG@5: {:.4f}, NDCG@10: {:.4f}, NDCG@15: {:.4f}, NDCG@20: {:.4f}, AUC: {:.4f}, train loss: {:.4f}, eval loss: {:.4f}\n'.format(
                    HRs_per_run_np.mean(axis=0)[0], HRs_per_run_np.mean(axis=0)[5], HRs_per_run_np.mean(axis=0)[10],
                    HRs_per_run_np.mean(axis=0)[15], NDCGs_per_run_np.mean(axis=0)[0],
                    NDCGs_per_run_np.mean(axis=0)[5], NDCGs_per_run_np.mean(axis=0)[10],
                    NDCGs_per_run_np.mean(axis=0)[15], AUC_per_run_np.mean(axis=0)[0],
                    train_loss_per_run_np.mean(axis=0)[0], eval_loss_per_run_np.mean(axis=0)[0]
                )
            )
            instantwrite(logger_file)

