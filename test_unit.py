
def test_data_pipe():
    from data_pipe import DataDealer
    filename = '/data1/nzw/CNER/weibo_conll/test.test'
    data_dealer = DataDealer()
    data_dealer.parse_data(filename)
    sent_bundles = data_dealer.sentence_bundle[:5]
    targ_bundles = data_dealer.target_bundle[:5]
    for sent_bund, targ_bund in zip(sent_bundles, targ_bundles):
        for sent in sent_bund:
            print(''.join([e['word'] for e in sent]))
            print(' '.join([e['tag'] for e in sent]))
        for targ in targ_bund:
            print(''.join(targ))

def test_my_tokenizer():
    from data_pipe import DataLoader, MyTokenizer
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    data_dealer = DataLoader()
    data_dealer.parse_data(file_path)
    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(data_dealer.new_tokens)
    print(my_tknzr.dic_cls_tok_id)
    print(my_tknzr.dic_cls_order)

def test_data_dealer():
    from data_pipe import DataLoader, MyTokenizer, DataDealer
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    data_loader = DataLoader()
    data_loader.parse_data(file_path)
    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tknzr = MyTokenizer.from_pretrained(model_path)
    my_tknzr.add_special_tokens(data_loader.new_tokens)
    
    data_dealer = DataDealer(data_loader, my_tknzr)
    sent = data_loader.sentences[0]
    sent = [
        {'word':'感','tag':'o'},{'word':'动','tag':'o'},
        {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'}]
    res = data_dealer.get_hier_sent(sent)
    print(res)
    for s, t in zip(res[0], res[2]):
        print(s)
        print(t)
    for s, t in zip(res[1], res[3]):
        print(s)
        print(t)

if __name__ == '__main__':
    # test_data_pipe()
    # test_my_tokenizer()
    test_data_dealer()