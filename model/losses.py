import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossWithMask(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithMask, self).__init__()
    
    def test_func(self):
        minmum = -1e32
        bsz = 2
        dec_num = 6
        size_ = [bsz,dec_num]
        enc_num = 4

        logits = torch.torch.randn(size_ + [enc_num])

        enc_mask = torch.randint(0,2,[bsz,enc_num])
        enc_mask = enc_mask.unsqueeze(1)
        logits = logits.masked_fill(enc_mask, minmum)

        dec_mask = torch.randint(0,2,size_)
        dec_mask = dec_mask.eq(0)
        logits = logits.masked_fill(dec_mask.unsqueeze(-1), minmum)

        dec_targ_pos = torch.randint(0,enc_num,size_)
        dec_targ_pos = dec_targ_pos.masked_fill(dec_mask, -100)
        print('losses 28', logits.shape, dec_targ_pos.shape)
        print(logits)
        print(dec_targ_pos)
        loss_tgt = F.cross_entropy(
            input=logits.transpose(1, 2), 
            target=dec_targ_pos)
        print(loss_tgt)

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
        # self.test_func()
        # print('losses 47', logits.shape, dec_targ_pos.shape)
        # print(type(logits), type(dec_targ_pos))
        # print(logits.dtype, dec_targ_pos.dtype)
        # print(dec_targ_pos)
        loss_tgt = F.cross_entropy(
            input=logits.transpose(1, 2), 
            target=dec_targ_pos)
        return loss_tgt



