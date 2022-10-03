
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
    from data_pipe import DataReader, MyTokenizer
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    data_reader = DataReader()
    data_reader.parse_data(file_path)
    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(data_reader.new_tokens)
    print(my_tknzr.dic_cls_tok_id)
    print(my_tknzr.dic_cls_order)

def test_data_dealer():
    from data_pipe import parse_CoNLL_file, parse_label
    from data_pipe import MyTokenizer, DataDealer
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    sentences = parse_CoNLL_file(file_path)
    new_tokens_bundle = parse_label(sentences)

    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(new_tokens_bundle)
    
    data_dealer = DataDealer(my_tknzr)
    sent = sentences[0]
    sent = [
        {'word':'感','tag':'o'},{'word':'动','tag':'o'},
        {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'}]
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
    data_dealer.get_one_sample(sent)
    # for sent in data_reader.sentences:
    #     data_dealer.get_one_sample(sent)

from data_pipe import *
from dataset import *
def test_random_sampler():
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    sentences = parse_CoNLL_file(file_path)
    new_tokens_bundle = parse_label(sentences)

    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(new_tokens_bundle)
    
    data_dealer = DataDealer(my_tknzr)
    dataset = CoNLLDataset(sentences, data_dealer)
    samp = GroupBatchRandomSampler(dataset, 10, 20)
    for s in samp:
        print('79', s)

if __name__ == '__main__':
    # test_data_pipe()
    # test_my_tokenizer()
    # test_data_dealer()
    test_random_sampler()