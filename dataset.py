import torch
import bisect
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler
from data_pipe import padding

class CoNLLDataset(Dataset):
    '''输入数据的迭代器'''
    def __init__(self, sentences, data_dealer):
        super(CoNLLDataset, self).__init__()
        data = []
        for sent in sentences:
            data.append(data_dealer.get_one_sample(sent))
        self.data = data
    
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

class GroupBatchRandomSampler(Sampler):
    '''
    得到按长度分组后采样的样本序号
    '''
    def __init__(self, data_source, batch_size, group_interval):
        '''
        data_source: 可迭代对象
        batch_size: batch的大小
        group_interval: 为了减少pad，把样本按长度分组，在同一组中执行BatchSampler
        '''
        super(GroupBatchRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.group_interval = group_interval

        max_len = max([len(d["enc_src_ids"]) for d in self.data_source])
        breakpoints = np.arange(group_interval, max_len, group_interval)
        self.groups = [[] for _ in range(len(breakpoints) + 1)]
        for i, data in enumerate(self.data_source):
            group_id = bisect.bisect_right(breakpoints, len(data["enc_src_ids"]))
            self.groups[group_id].append(i)
        self.batch_indices = []
        for g in self.groups:
            self.batch_indices.extend(list(
                BatchSampler(SubsetRandomSampler(g), 
                self.batch_size, False)
            ))
    
    def __iter__(self):
        batch_indices = []
        for g in self.groups:
            batch_indices.extend(list(
                BatchSampler(SubsetRandomSampler(g), self.batch_size, False)
            ))
        return (batch_indices[i] for i in torch.randperm(len(batch_indices)))
    
    def __len__(self):
        return len(self.batch_indices)

def collate_fn(batch_data, pad_value=0, device=torch.device("cpu")):
    padded_batch = defaultdict(list)
    enc_src_ids, mask = padding(
        [d["enc_src_ids"] for d in batch_data], pad_value)
    padded_batch["enc_src_ids"] = torch.tensor(enc_src_ids, dtype=torch.long, device=device)
    padded_batch["enc_src_len"] = torch.tensor([d["enc_src_len"] for d in batch_data], device=device)
    padded_batch["enc_mask"] = torch.tensor(mask, dtype=torch.bool, device=device)
    
    for i in range(len(batch_data[0]['dec_src_ids'])):
        dec_src_ids, mask = padding(
            [d["dec_src_ids"][i] for d in batch_data], pad_value)
        dec_src_pos, mask = padding(
            [d["dec_src_pos"][i] for d in batch_data], pad_value)
        padded_batch["dec_src_ids_bund"].append(torch.tensor(dec_src_ids, dtype=torch.long, device=device))
        padded_batch["dec_src_pos_bund"].append(torch.tensor(dec_src_pos, dtype=torch.long, device=device))
        padded_batch["dec_mask_bund"].append(torch.tensor(mask, dtype=torch.bool, device=device))
        
        dec_targ_pos, mask = padding(
            [d["dec_targ_pos"][i] for d in batch_data], pad_value)
        padded_batch["dec_targ_pos_bund"].append(torch.tensor(dec_targ_pos, dtype=torch.long, device=device))
    #     print(padded_batch["dec_src_ids_bund"][-1].shape)
    # print('dataset 83', len(padded_batch["dec_src_ids_bund"]))
    targ_ents = [d["targ_ents"] for d in batch_data]
    padded_batch["targ_ents"] = targ_ents
    return padded_batch