import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CrossEntropyLossWithMask

class HiDecoder(nn.Module):
    def __init__(self, decoder, args):
        super(HiDecoder, self).__init__()
        self.decoder = decoder
        self.args = args
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_mask', causal_mask.float())
        self.dropout_layer = nn.Dropout(0.3)
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self, enc_output, enc_src_ids, enc_len, enc_mask, 
        dec_src_ids, dec_mask):
        '''
        预测一次输出，得到token是每个候选token的概率
        enc_output: bsz*max_len*emb_dim
        enc_len: bsz*1
        '''
        # 过解码器，得到隐藏状态
        causal_mask = self.causal_mask[
            :dec_src_ids.size(1), :dec_src_ids.size(1)]
        dec_state_dic = self.bart.decoder(
            input_ids=dec_src_ids,
            encoder_hidden_states=enc_output,
            encoder_padding_mask=enc_mask,
            decoder_padding_mask=dec_mask,
            decoder_causal_mask=causal_mask,
            return_dict=True
        )
        hidden_state = dec_state_dic.last_hidden_state
        hidden_state = self.dropout_layer(hidden_state)
        # 初始化预测结果，bsz*dec_out_max_len*enc_out_max_len
        logits = hidden_state.new_full(
            list(hidden_state.size())+[enc_output.size(-1)],
            fill_value=-1e24)
        
        if self.args['static_eos']:
            # 静态结束符，用词表中的结束符嵌入向量作为目标
            eos_id = self.args['eos_id']
            eos_emb = self.decoder.embed_tokens.weight[eos_id:eos_id+1]
            eos_scores = F.linear(hidden_state, self.dropout_layer(eos_emb))
        else:
            # 动态结束符，用结束符的编码向量作为目标
            eos_emb = enc_output[range(len(enc_output)), enc_len-1, :]
            eos_emb = eos_emb.unsqueeze(1)
            eos_scores = torch.bmm(hidden_state, eos_emb.permute(0,2,1))
        
        enc_src_embed = self.decoder.embed_tokens(enc_src_ids)
        enc_src_embed = self.dropout_layer(enc_src_embed)
        enc_output = self.dropout_layer(enc_output)
        src_embed = (enc_src_embed + enc_output)/2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_embed)
        
        # word_scores: bsz*max_dec_len*max_enc_len
        # enc_mask: bsz*1*max_enc_len, 结束标记的得分不算在word_score中
        # dec_mask: bsz*max_dec_len*1
        enc_mask = enc_mask.eq(0)
        enc_mask[range(len(enc_mask)), enc_len-1] = True
        enc_mask = enc_mask.unsqueeze(1)
        dec_mask = dec_mask.eq(0).unsqueeze(-1)
        word_scores = word_scores.masked_fill(enc_mask, -1e32)
        word_scores = word_scores.masked_fill(dec_mask, -1e32)

        logits[:,:,1:2] = eos_scores
        logits[:,:,0:1] = word_scores[:,:,0:1]
        logits[:,:,2:] = word_scores[:,:,1:]
        return logits

class HiBart(nn.Module):
    def __init__(self, bart):
        super(HiBart, self).__init__()
        self.encoder = bart.encoder
        self.decoder = HiDecoder(bart.decoder)
        self.loss_fn = CrossEntropyLossWithMask()
    
    def forward(
        self, enc_src_ids, enc_len, enc_mask, 
        dec_src_ids_bund, dec_mask_bund, dec_targ_pos_bund):
        '''
        enc_src_ids: batch_size*max_len
        enc_mask: batch_size*max_len
        dec_src_ids_bund: batch_size*3*max_len
        dec_mask: batch_size*3*max_len
        dec_targ_pos: batch_size*3*max_len
        经过三次解码器得到三个loss
        '''
        enc_state_dic = self.encoder(
            input_ids=enc_src_ids, attention_mask=enc_mask, 
            return_dict=True, output_hidden_states=True,
        )
        enc_output = enc_state_dic.last_hidden_state
        dec_src_ids_bund = dec_src_ids_bund.permute(1,0,2)
        dec_mask_bund = dec_mask_bund.permute(1,0,2)
        dec_targ_pos_bund = dec_targ_pos_bund.permute(1,0,2)

        batch_loss = 0
        for i in range(len(dec_targ_pos_bund)):
            dec_src_ids = dec_src_ids_bund[i]
            dec_mask = dec_mask_bund[i]
            dec_targ_pos = dec_targ_pos_bund[i]
            pred = self.decoder(
                enc_output, enc_src_ids, enc_len, enc_mask,
                dec_src_ids, dec_mask)
            batch_loss += self.loss_fn(pred, dec_targ_pos)


