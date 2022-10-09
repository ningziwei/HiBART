
def test_data_reader():
    from data_pipe import DataReader
    filename = '/data1/nzw/CNER/weibo_conll/test.test'
    data_reader = DataReader()
    data_reader.parse_data(filename)
    sent_bundles = data_reader.sentence_bundle[:5]
    targ_bundles = data_reader.target_bundle[:5]
    for sent_bund, targ_bund in zip(sent_bundles, targ_bundles):
        for sent in sent_bund:
            print(''.join([e['word'] for e in sent]))
            print(' '.join([e['tag'] for e in sent]))
        for targ in targ_bund:
            print(''.join(targ))

def test_my_tokenizer():
    from data_pipe import parse_CoNLL_file, parse_label, MyTokenizer
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    sentences = parse_CoNLL_file(file_path)
    cls_token_cache = parse_label(sentences)
    cls_tok_dic = cls_token_cache['cls_tok_dic']
    new_tokens_bundle = cls_token_cache['new_tokens_bundle']

    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(cls_tok_dic, new_tokens_bundle)
    print(my_tknzr.dic_cls_id)
    print(my_tknzr.dic_cls_order)
    print(my_tknzr.dic_hir_pos_cls)

def test_data_dealer():
    from data_pipe import parse_CoNLL_file, parse_label
    from data_pipe import MyTokenizer, DataDealer
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    sentences = parse_CoNLL_file(file_path)
    cls_tok_dic, new_tokens_bundle = parse_label(sentences)

    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(cls_tok_dic, new_tokens_bundle)
    
    data_dealer = DataDealer(my_tknzr)
    sent = sentences[0]
    sent = [
        {'word':'感','tag':'o'},{'word':'动','tag':'o'},
        {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'}]
    # sent = [
    #     {'word':'感','tag':'o'},{'word':'动','tag':'o'}]
    # sent = [
    #     {'word':'感','tag':'o'},{'word':'动','tag':'o'},
    #     {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'},
    #     {'word':'人','tag':'b-per.nam'},{'word':'物','tag':'i-per.nam'},
    #     {'word':'物','tag':'o'}]
    # sent = [
    #     {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'},
    #     {'word':'人','tag':'b-per.nam'},{'word':'物','tag':'i-per.nam'},
    #     ]
    # res = data_dealer.get_hier_sent(sent)
    # print(res)
    # for s, t in zip(res[0], res[1]):
    #     print(s)
    #     print(t)
    # print('55', res[0][3])
    # ents = data_dealer.get_targ_ents(res[2][-1])
    # print('实体', ents)
    samp = data_dealer.get_one_sample(sent)
    print(samp)
    # for sent in data_reader.sentences:
    #     data_dealer.get_one_sample(sent)

from torch.utils.data import DataLoader
def test_random_sampler():
    from data_pipe import parse_CoNLL_file, parse_label
    from data_pipe import MyTokenizer, DataDealer
    from dataset import CoNLLDataset, GroupBatchRandomSampler, collate_fn
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    sentences = parse_CoNLL_file(file_path)
    new_tokens_bundle = parse_label(sentences)

    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(new_tokens_bundle)
    
    data_dealer = DataDealer(my_tknzr)
    dataset = CoNLLDataset(sentences, data_dealer)
    samp = GroupBatchRandomSampler(dataset, 10, 20)
    train_loader = DataLoader(
        dataset=dataset, 
        batch_sampler=samp, 
        collate_fn=collate_fn)
    for s in samp:
        print('79', s)

import json
def test_data_loader():
    from model import get_data_loader
    with open('config.json', encoding="utf-8") as fp:
        config = json.load(fp)
    loaders = get_data_loader(config)
    for i,batch in enumerate(loaders[1]):
        print(i, batch['dec_src_ids'].shape)

import torch
from data_pipe import *  
def test_flat_seq():
    batch_pred_list = [
        [[19, 20, 10, 22, 1]],
        [[19, 20, 10, 21, 22, 18]],
        [[19, 20, 10, 21, 22, 18, 11]]
    ]
    batch_enc_src_ids = [[
        101, 21128, 21129, 21130, 21131, 21132, 21133, 21134, 21135, 21136, 
        21137, 21138, 21139, 21140, 21141, 21142, 21143, 21144, 2697, 1220, 
        704, 1744, 102
    ]]
    batch_dec_src_ids = torch.tensor([[101, 2697, 1220, 704, 1744,-1,-1,-1]])
    dic_order_cls = {0: 21128, 1: 21129, 2: 21130, 3: 21131, 4: 21132, 5: 21133, 6: 21134, 7: 21135, 8: 21136, 9: 21137, 10: 21138, 11: 21139, 12: 21140, 13: 21141, 14: 21142, 15: 21143, 16: 21144}
    dic_hir_pos_cls = [
        {2: '<<per.nam-s>>', 4: '<<gpe.nam-s>>', 6: '<<org.nom-s>>', 8: '<<per.nom-s>>', 10: '<<loc.nam-s>>', 12: '<<loc.nom-s>>', 14: '<<gpe.nom-s>>', 16: '<<org.nam-s>>'}, 
        {18: '<<ent_end>>'}, 
        {3: '<<per.nam-e>>', 5: '<<gpe.nam-e>>', 7: '<<org.nom-e>>', 9: '<<per.nom-e>>', 11: '<<loc.nam-e>>', 13: '<<loc.nom-e>>', 15: '<<gpe.nom-e>>', 17: '<<org.nam-e>>'}]

    for i in range(3):
        batch_pred = batch_pred_list[i]
        dic_pos_cls = dic_hir_pos_cls[i]
        res = flat_sequence(
            batch_pred, 
            batch_enc_src_ids, 
            batch_dec_src_ids,
            dic_pos_cls,
            pad_value=0,
            device=torch.device("cpu")
        )
        print(res)
        batch_dec_src_ids = res[0]



if __name__ == '__main__':
    # test_data_pipe()
    # test_my_tokenizer()
    # test_data_dealer()
    # test_random_sampler()
    # test_data_loader()
    test_flat_seq()