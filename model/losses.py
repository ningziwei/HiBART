import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossWithMask(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithMask).__init__()
    
    def forward(self, logits, dec_targ_pos, dec_mask):
        '''
        logits: bsz*max_dec_len*max_enc_len
        dec_targ_pos: bsz*max_dec_len
        dec_mask: bsz*max_dec_len
        cross_entropy的ignore_index是-100，即目标值为-100时
        忽略其对应的loss
        '''
        dec_mask = dec_mask.eq(0)
        dec_targ_pos = dec_targ_pos.masked_fill(dec_mask, -100)
        # pred_tokens = torch.argmax(logits,dim=-1)
        loss_tgt = F.cross_entropy(
            input=logits.transpose(1, 2), 
            target=dec_targ_pos)
        return loss_tgt



