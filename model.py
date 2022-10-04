import os
from torch.utils.data import DataLoader
from data_pipe import parse_CoNLL_file, parse_label
from data_pipe import MyTokenizer, DataDealer
from dataset import *

def get_data_loader(config):
    '''得到三个数据集的dataloader'''
    dataset_dir = os.path.join(config['data_dir'], config['dataset'])
    file_path = os.path.join(dataset_dir, 'train.train')
    sentences = parse_CoNLL_file(file_path)
    new_tokens_bundle = parse_label(sentences)

    model_path = config['bart_path']
    my_tokenizer = MyTokenizer.from_pretrained(model_path)
    my_tokenizer.add_special_tokens(new_tokens_bundle)
    pad_value = my_tokenizer.pad_token_id

    data_dealer = DataDealer(my_tokenizer)

    def get_loader(subset):
        file_path = os.path.join(dataset_dir, f'{subset}.{subset}')
        sentences = parse_CoNLL_file(file_path)
        __dataset = CoNLLDataset(sentences, data_dealer)
        __sampler = GroupBatchRandomSampler(
            __dataset, config["batch_size"], config["group_interval"])
        __loader = DataLoader(
            dataset=__dataset, 
            batch_sampler=__sampler, 
            collate_fn=lambda x: collate_fn(x, pad_value))
        return __loader
    
    train_loader = get_loader("train")
    test_loader = get_loader("test")
    valid_loader = get_loader("dev")
    return train_loader, test_loader, valid_loader