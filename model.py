import os
from data_pipe import parse_CoNLL_file, parse_label
from data_pipe import MyTokenizer, DataDealer
from dataset import *

def get_data_loader(config):
    file_path = '/data1/nzw/CNER/weibo_conll/test.test'
    sentences = parse_CoNLL_file(file_path)
    new_tokens_bundle = parse_label(sentences)

    model_path = '/data1/nzw/model/bart-base-chinese/'
    my_tokenizer = MyTokenizer.from_pretrained(model_path)
    my_tokenizer.add_special_tokens(new_tokens_bundle)
    
    data_dealer = DataDealer(my_tokenizer)
    def get_loader(subset):
        file_path = os.path.join(
            config['dataset_root'], f'{subset}.{subset}')
        sentences = parse_CoNLL_file(file_path)
        __dataset = CoNLLDataset(sentences, data_dealer)
        __sampler = GroupBatchRandomSampler(__dataset, config["batch_size"], 20)
        __loader = DataLoader(dataset=__dataset, batch_sampler=__sampler, collate_fn=dataset.collate_fn)
        return __loader