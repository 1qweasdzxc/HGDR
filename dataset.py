'''
@Date : 2021/11/5
@Author : lailai
@encoding: utf-8
@email: lailai_zxy@tju.edu.cn
'''
from torch.utils.data import Dataset
import pickle
import os.path as osp
import warnings
import torch
import re
import os
import collections
import pandas as pd
from os.path import join
import numpy as np
from os.path import isfile
from collections import Counter
import tqdm
import random as rd

def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

def to_list(x):
    if not isinstance(x, collections.Iterable) or isinstance(x, str):
        x = [x]
    return x

def makedirs(path):
    try:
        path = osp.expanduser(osp.normpath(path))
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def files_exist(files):
    return all([osp.exists(f) for f in files])

class DataProcess(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 **kwargs):
        self.type = kwargs['type']
        assert self.type in ['hete']
        self.num_core = kwargs['num_core']
        self.num_feat_core = kwargs['num_feat_core']

        # self.entity_aware = kwargs['entity_aware']

        self.num_negative_samples = kwargs['num_negative_samples']
        self.sampling_strategy = kwargs['sampling_strategy']
        self.cf_loss_type = kwargs['cf_loss_type']

        super(DataProcess,self).__init__()
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.__indices__ = None
        if 'process' in self.__class__.__dict__.keys():
            self._process()

        #加载预处理好的数据集
        with open(self.processed_paths[0],'rb') as f:
            dataset_property_dict = pickle.load(f)
        for k,v in dataset_property_dict.items():
            self[k] = v


    def _process(self):
        f = osp.join(self.processed_dir,'pre_transform.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            warnings.warn(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            warnings.warn(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))
        if files_exist(self.processed_paths):  # pragma: no cover
            return

        print('Processing...')
        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')

    @property
    def processed_dir(self):
        return osp.join(self.root,'processed')

    @property
    def processed_paths(self):
        r"""The filepaths to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['ml_core_{}_type_{}.pkl'.format(self.num_core, self.type)]

    #数据预处理
    #需要保证Sync、Remove duplicates
    def process(self):
        print(os.getcwd())
        issues_developer_inter = pd.read_csv(join(self.processed_dir, 'vscode_inter30.csv'), sep='\t').fillna('')
        issues = pd.read_csv(join(self.processed_dir, 'issues_info.csv'), sep=',')
        developer_source_code = pd.read_table(join(self.processed_dir, 'developer_source_code.csv'), sep='\t',names=['developer_id','source_code_id'])
        issues_source_code = pd.read_table(join(self.processed_dir, 'issues_source_code.txt'), sep='\t', names=['issue_id','source_code_id'])
        print('Read data frame from {}!'.format(self.processed_dir))

        # Generate and save graph
        if self.type == 'hete':
            dataset_property_dict = generate_hete_graph(issues_developer_inter, issues, developer_source_code, issues_source_code)
        else:
            raise NotImplementedError
        with open(self.processed_paths[0], 'wb') as f:
            pickle.dump(dataset_property_dict, f)

    def cf_negative_sampling(self):
        """
        Replace positive items with random/unseen items
        """
        developer_source_code = pd.read_table(join(self.processed_dir, 'developer_source_code.csv'), sep='\t',
                                              names=['developer_id', 'source_code_id'])
        issues = pd.read_csv(join(self.processed_dir, 'issues_info.csv'), sep=',')
        issues_source_code = pd.read_table(join(self.processed_dir, 'issues_source_code.txt'), sep='\t',
                                           names=['issue_id', 'source_code_id'])
        print("negative sampling")
        pos_edge_index_trans_np = self.edge_index_nps['issue2developer'].T
        num_interactions = pos_edge_index_trans_np.shape[0]
        if self.cf_loss_type == 'BPR':
            train_data_np = np.repeat(pos_edge_index_trans_np,repeats=self.num_negative_samples, axis=0)
            if self.sampling_strategy == 'random':
                neg_inid_np = np.random.randint(low=self.type_accs['did'],
                                                high= self.type_accs['did'] + self.num_dids,
                                                size = (num_interactions * self.num_negative_samples, 1)
                                                )
            elif self.sampling_strategy == 'unseen':
                neg_inids = []
                i_nids = pos_edge_index_trans_np[:, 0]
                p_bar  = tqdm.tqdm(i_nids)
                for i_nid in p_bar:
                    negative_inids = self.test_pos_inid_dnid_map[i_nid] + self.neg_inid_dnid_map[i_nid]
                    negative_inids = rd.choice(negative_inids, k = self.num_negative_samples)
                    negative_inids = np.array(negative_inids, dtype=np.long).reshape(-1,1)
                    neg_inids.append(negative_inids)
                neg_inid_np = np.vstack(neg_inids)
            else:
                raise NotImplementedError
            train_data_np = np.hstack([train_data_np, neg_inid_np])
        #     if self.entity_aware and not hasattr(self, 'did_feat_nids'):
        #         # build developer feature
        #         did_feat_nids = []
        #         pbar = tqdm.tqdm(self.unique_dids, total=len(self.unique_dids))
        #         for did in pbar:
        #             pbar.set_description('Sampling item entities')
        #             feat_nids = []
        #             source_code_nids = [self.e2nid_dict['sid'][sid] for sid in developer_source_code[developer_source_code.developer_id==did].source_code_id]
        #             feat_nids += source_code_nids
        #             did_feat_nids.append(feat_nids)
        #         self.did_feat_nids = did_feat_nids

        #         # build issue feature
        #         iid_feat_nids = []
        #         pbar  = tqdm.tqdm(self.unique_iids, total=len(self.unique_iids))
        #         for iid in pbar:
        #             pbar.set_description('Sampling issue entities')
        #             feat_nids = []
        #             tag_nids = [self.e2nid_dict['tag'][tag] for tag in self.unique_tags if issues[issues.issue_id==iid][tag].item()]
        #             feat_nids += tag_nids
        #             source_code_nids = [self.e2nid_dict['sid'][sid] for sid in issues_source_code[issues_source_code.issue_id==iid].source_id]
        #             feat_nids += source_code_nids
        #         self.iid_feat_nids = iid_feat_nids
        # else:
        #     raise NotImplementedError
        train_data_t = torch.from_numpy(train_data_np).long()
        shuffle_idx = torch.randperm(train_data_t.shape[0])
        self.train_data = train_data_t[shuffle_idx]
        self.train_data_length = train_data_t.shape[0]

    def __len__(self):
        return self.train_data_length

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        if isinstance(key, str):
            setattr(self, key, value)
        else:
            raise NotImplementedError('Assignment can\'t be done outside of constructor')

    def __getitem__(self, idx):
        r"""
        Gets the data object at index:obj:'idx' and transorms it (in case
        a: obj :self.transform` is given)
         In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx,str):
            return getattr(self,idx,None)
        else:
            idx = idx.to_list() if torch.is_tensor(idx) else idx
            train_data_t = self.train_data[idx]
            # if self.entity_aware:
            #     did = train_data_t[1].cpu().detach().item()
            #     feat_nids = self.did_feat_nids[int(did - self.type_accs['did'])]

            #     if len(feat_nids) == 0:
            #         pos_developer_entity_nid = 0
            #         neg_developer_entity_nid = 0
            #         developer_entity_mask = 0
            #     else:
            #         pos_developer_entity_nid = rd.choice(feat_nids)
            #         entity_type = self.nid2e_dict[pos_developer_entity_nid][0]
            #         lower_bound = self.type_accs.get(entity_type)
            #         upper_bound = lower_bound + getattr(self, 'num_' + entity_type + 's')
            #         neg_developer_entity_nid = rd.choice(range(lower_bound,upper_bound))
            #         developer_entity_mask = 1

            #     iid = train_data_t[0].cpu().detach().item()
            #     feat_nids = self.iid_feat_nids[int(iid - self.type_accs['iid'])]
            #     if len(feat_nids) == 0:
            #         pos_issue_entity_nid = 0
            #         neg_issue_entity_nid = 0
            #         issue_entity_mask = 0
            #     else:
            #         pos_issue_entity_nid = rd.choice(feat_nids)
            #         entity_type = self.nid2e_dict[pos_issue_entity_nid]
            #         lower_bound = self.type_accs.get(entity_type)
            #         upper_bound = lower_bound + getattr(self, 'num_' + entity_type + 's')
            #         neg_issue_entity_nid = rd.choice(range(lower_bound, upper_bound))
            #         issue_entity_mask = 1

                # pos_neg_entities = torch.tensor([pos_developer_entity_nid, neg_developer_entity_nid, developer_entity_mask, pos_issue_entity_nid, neg_issue_entity_nid, issue_entity_mask],dtype=torch.long)
                # train_data_t = torch.cat([train_data_t, pos_neg_entities], dim = -1)
            return train_data_t





    def __repr__(self):
        return '{}-{}'.format(self.__class__.__name__, self.name.capitalize())


def generate_hete_graph(issues_developer_inter, issues, developer_source_code, issues_source_code):
    unique_iids = list(np.sort(issues_developer_inter.issues_id.unique()))

    developer_inter_num = dict(issues_developer_inter.developers_id.value_counts())
    developer_inter_num = sorted(developer_inter_num.items(),key=lambda x:x[1], reverse=True)

    head_developer = developer_inter_num[:(int)(0.1*len(developer_inter_num))]

    num_iids = len(unique_iids)

    unique_dids = list(np.sort(developer_source_code.developer_id.unique()))
    num_dids = len(unique_dids)

    unique_tags = list(issues.keys()[1:])
    num_tags = len(unique_tags)

    unique_sids = list(np.sort(developer_source_code.source_code_id.unique()))
    num_sids = len(unique_sids)

    dataset_property_dict = {}
    dataset_property_dict['unique_iids'] = unique_iids
    dataset_property_dict['num_iids'] = num_iids
    dataset_property_dict['unique_dids'] = unique_dids
    dataset_property_dict['num_dids'] = num_dids
    dataset_property_dict['unique_tags'] = unique_tags
    dataset_property_dict['num_tags'] = num_tags
    dataset_property_dict['unique_sids'] = unique_sids
    dataset_property_dict['num_sids'] = num_sids
    

    # Define number of entities
    num_nodes = num_iids + num_dids + num_sids + num_tags
    num_node_types = 4
    dataset_property_dict['num_nodes'] = num_nodes
    dataset_property_dict['num_node_types'] = num_node_types
    types = ["iid","did","tag","sid"]
    num_nodes_dict = {'iid':num_iids,'did':num_dids,'sid':num_sids,'tag':num_tags}

    # Define entities to node id map
    type_accs = {}
    nid2e_dict = {}
    acc = 0
    type_accs['iid'] = acc
    iid2nid = {iid: i+acc for i, iid in enumerate(unique_iids)}
    for i,iid in enumerate(unique_iids):
        nid2e_dict[i+acc] = ('iid',iid)
    acc += num_iids
    type_accs['did'] = acc
    did2nid = {did: i + acc for i, did in enumerate(unique_dids)}
    for i, did in enumerate(unique_dids):
        nid2e_dict[i + acc] = ('did', did)
    acc += num_dids
    sid2nid = {sid: i + acc for i, sid in enumerate(unique_sids)}
    for i, sid in enumerate(unique_sids):
        nid2e_dict[i + acc] = ('sid', sid)
    acc += num_sids
    tag2nid = {tag: i + acc for i, tag in enumerate(unique_tags)}
    for i, tag in enumerate(unique_tags):
        nid2e_dict[i + acc] = ('tag', tag)
    acc += num_tags
    e2nid_dict ={'iid':iid2nid, 'did':did2nid, 'sid':sid2nid, 'tag':tag2nid}
    dataset_property_dict['e2nid_dict'] = e2nid_dict
    dataset_property_dict['nid2e_dict'] = nid2e_dict

    # create graphs
    edge_index_nps = {}
    print('Creating issue attributes edges..')
    tag_nids = []
    inids = []
    for tag in unique_tags:
        iids = issues[issues[tag]].issue_id
        inids += [e2nid_dict['iid'][iid] for iid in iids]
        tag_nids += [e2nid_dict['tag'][tag] for _ in range(iids.shape[0])]
    tag2item_edge_index_np = np.vstack((np.array(tag_nids), np.array(inids)))
    edge_index_nps['tag2issue'] = tag2item_edge_index_np

    print('Creating inter property edges...')
    test_pos_inid_dnid_map, neg_inid_dnid_map = {}, {}
    issue2item_edge_index_np = np.zeros((2,0))
    pbar = tqdm.tqdm(unique_iids, total = len(unique_iids))
    for iid in pbar:
        pbar.set_description('Creating the edges for the issue {}'.format(iid))
        iid_inter = issues_developer_inter[issues_developer_inter.issues_id == iid].sort_values('date')
        iid_dids = iid_inter.developers_id.to_numpy()
        inid = e2nid_dict['iid'][iid]
        train_pos_iid_dids = list(iid_dids[:-1])
        train_pos_iid_dnids = [e2nid_dict['did'][did] for did in train_pos_iid_dids]
        test_pos_iid_dids = list(iid_dids[-1:])
        test_pos_iid_dnids = [e2nid_dict['did'][did] for did in test_pos_iid_dids]
        neg_iid_dids  = list(set(unique_dids) - set(iid_dids))
        neg_iid_dnids = [e2nid_dict['did'][did] for did in neg_iid_dids]

        test_pos_inid_dnid_map[inid] = test_pos_iid_dnids
        neg_inid_dnid_map[inid] = neg_iid_dnids

        inid_issue2item_edge_index_np = np.array([[inid for _ in range(len(train_pos_iid_dnids))], train_pos_iid_dnids])
        issue2item_edge_index_np = np.hstack([issue2item_edge_index_np,inid_issue2item_edge_index_np])

    edge_index_nps['issue2developer'] = issue2item_edge_index_np
    dataset_property_dict['edge_index_nps'] = edge_index_nps
    dataset_property_dict['test_pos_inid_dnid_map'],dataset_property_dict['neg_inid_dnid_map'] = test_pos_inid_dnid_map, neg_inid_dnid_map

    inids = [e2nid_dict['iid'][iid] for iid in issues_source_code.issue_id]
    snids = [e2nid_dict['sid'][sid] for sid in issues_source_code.source_code_id]
    source_code2issue_edge_index_np = np.vstack((np.array(snids), np.array(inids)))
    dnids = [e2nid_dict['did'][did] for did in developer_source_code.developer_id]
    snids = [e2nid_dict['sid'][sid] for sid in developer_source_code.source_code_id]
    source_code2develoepr_edge_index_np = np.vstack((np.array(snids), np.array(dnids)))
    edge_index_nps['source_code2issue'] = source_code2issue_edge_index_np
    edge_index_nps['source_code2developer'] = source_code2develoepr_edge_index_np

    print('Building edge type map...')
    edge_type_dict = {edge_type: edge_type_idx for edge_type_idx, edge_type in enumerate(list(edge_index_nps.keys()))}
    dataset_property_dict['edge_type_dict'] = edge_type_dict
    dataset_property_dict['num_edge_types'] = len(list(edge_index_nps.keys()))

    print('Building the item occurrence map')
    developer_count = developer_source_code['developer_id'].value_counts()
    developer_nid_occs = {}
    for did in unique_dids:
        developer_nid_occs[e2nid_dict['did'][did]] = developer_count[did]
    dataset_property_dict['developer_nid_occs'] = developer_nid_occs

    dataset_property_dict['types'] = types
    dataset_property_dict['num_node_types'] = num_nodes_dict
    dataset_property_dict['type_accs'] = type_accs
    dataset_property_dict['head_developers'] = [[e2nid_dict['did'][did[0]]] for did in head_developer]

    return dataset_property_dict







