import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, BatchSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoNLLDataset(Dataset):
    def __init__(self, data_dealer):
        super(CoNLLDataset, self).__init__()
        self.data_dealer = data_dealer
    
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def collate_fn(batch_data):
    padded_batch = dict()
    enc_src_ids, mask = padding([d["enc_src_ids"] for d in batch_data])
    padded_batch["enc_src_ids"] = torch.tensor(enc_src_ids, dtype=torch.long, device=device)
    padded_batch["enc_mask"] = torch.tensor(mask, dtype=torch.bool, device=device)
    
    dec_src_ids, mask = padding([d["dec_src_ids"] for d in batch_data], dim=3)
    padded_batch["dec_src_ids"] = torch.tensor(dec_src_ids, dtype=torch.long, device=device)
    padded_batch["dec_mask"] = torch.tensor(mask, dtype=torch.bool, device=device)
    
    dec_targ_pos, mask = padding([d["dec_targ_pos"] for d in batch_data], dim=3)
    padded_batch["dec_targ_pos"] = torch.tensor(dec_targ_pos, dtype=torch.long, device=device)

    targ_ents = [d["targ_ents"] for d in batch_data]
    padded_batch["targ_ents"] = targ_ents
    return padded_batch