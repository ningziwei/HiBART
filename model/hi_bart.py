import torch
import torch.nn as nn
import torch.nn.functional as F

class HiDecoder(nn.Module):
    def __init__(self, decoder, args):
        super(HiDecoder, self).__init__()
        self.decoder = decoder
        self.args = args
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_mask', causal_mask.float())
        self.dropout_layer = nn.Dropout(0.3)
    
    def forward(self, enc_output, enc_len, enc_mask, dec_src, dec_mask):
        '''
        预测一次输出，得到token是每个候选token的概率
        enc_output: bsz*max_len*emb_dim
        enc_len: bsz*1
        '''
        causal_mask = self.causal_mask[
            :dec_src.size(1), :dec_src.size(1)]
        dec_state_dic = self.bart.decoder(
            input_ids=dec_src,
            encoder_hidden_states=enc_output,
            encoder_padding_mask=enc_mask,
            decoder_padding_mask=dec_mask,
            decoder_causal_mask=causal_mask,
            return_dict=True
        )

        dec_output = dict.last_hidden_state
        dec_output = self.dropout_layer(dec_output)
        logits = dec_output.new_full(
            list(dec_output.size())+[enc_output.size(-1)+1],
            fill_value=-1e24)
        
        if self.args['static_eos']:
            # 静态结束符，用词表中的结束符嵌入向量作为目标
            eos_id = self.args['eos_id']
            eos_emb = self.decoder.embed_tokens.weight[eos_id:eos_id+1]
            eos_scores = F.linear(dec_output, self.dropout_layer(eos_emb))
        else:
            # 动态结束符，用结束符的编码向量作为目标
            eos_emb = enc_output[range(len(enc_output)), enc_len-1, :]
            eos_emb = eos_emb.unsqueeze(1)
            eos_scores = torch.bmm(dec_output, eos_emb.permute(0,2,1))

        

class HiBart(nn.Module):
    def __init__(self, bart):
        super(HiBart, self).__init__()
        self.encoder = bart.encoder
        self.decoder = HiDecoder(bart.decoder)
        self.loss_fn = Seq2SeqLoss()
    
    def forward(
        self, enc_src_ids, enc_mask, 
        dec_src_ids, dec_masks, dec_targ_pos):
        '''
        enc_src_ids: batch_size*max_len
        enc_mask: batch_size*max_len
        dec_src_ids: batch_size*3*max_len
        dec_mask: batch_size*3*max_len
        dec_targ_pos: batch_size*3*max_len
        经过三次解码器得到三个loss
        '''
        enc_state_dic = self.encoder(
            input_ids=enc_src_ids, attention_mask=enc_mask, 
            return_dict=True, output_hidden_states=True,
        )
        enc_output = enc_state_dic.last_hidden_state
        dec_src_ids = dec_src_ids.permute(1,0,2)
        dec_masks = dec_masks.permute(1,0,2)
        dec_targ_pos = dec_targ_pos.permute(1,0,2)

        batch_loss = 0
        for i in range(len(dec_targ_pos)):
            dec_src = dec_src_ids[i]
            dec_mask = dec_masks[i]
            dec_targ = dec_targ_pos[i]
            pred = self.decoder(enc_output, enc_mask,
                dec_src, dec_mask)
            batch_loss += self.loss_fn(pred, dec_targ)


