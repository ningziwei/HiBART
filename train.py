import os
import json
import time
from torch.utils.data import DataLoader

import utils
from data_pipe import *
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

def get_config_dir_logger(config_path):
    with open(config_path, encoding="utf-8") as fp:
        config = json.load(fp)
    output_path = config["output_path"]
    prefix = config.get("prefix", "HiBart") + '_'
    curr_time = time.strftime("%Y%m%d%H%M", time.localtime())
    OUTPUT_DIR = os.path.join(output_path, prefix + curr_time)
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rf %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.system("cp %s %s" % (config_path, OUTPUT_DIR))
    logger = utils.Logger(open(os.path.join(OUTPUT_DIR, "log.txt"), 'w'))
    length = max([len(arg) for arg in config.keys()])
    for arg, value in config.items():
        logger("%s | %s" % (arg.ljust(length).replace('_', ' '), str(value)))
    return config, OUTPUT_DIR, logger

def train():
    config_path = 'config.json'
    config, OUTPUT_DIR, logger = get_config_dir_logger(config_path)
    loaders = get_data_loader(config)
    train_loader, test_loader, valid_loader = loaders

