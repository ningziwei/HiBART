import os
import json
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import utils
from data_pipe import *
from dataset import *
from model.modeling_bart import BartModel
from model.hi_bart import HiBart
from model.losses import CrossEntropyLossWithMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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

def get_tokenizer(config):
    '''得到模型的tokenizer'''
    dataset_dir = os.path.join(config['data_dir'], config['dataset'])
    file_path = os.path.join(dataset_dir, 'train.train')
    sentences = parse_CoNLL_file(file_path)
    cls_tok_dic, new_tokens_bundle = parse_label(sentences)

    model_path = config['bart_path']
    my_tokenizer = MyTokenizer.from_pretrained(model_path)
    my_tokenizer.add_special_tokens(cls_tok_dic, new_tokens_bundle)
    
    return my_tokenizer

def get_data_loader(config, tokenizer):
    '''得到三个数据集的dataloader'''
    dataset_dir = os.path.join(config['data_dir'], config['dataset'])
    data_dealer = DataDealer(tokenizer)
    pad_value = tokenizer.pad_token_id

    def get_loader(subset):
        file_path = os.path.join(dataset_dir, f'{subset}.{subset}')
        sentences = parse_CoNLL_file(file_path)
        __dataset = CoNLLDataset(sentences, data_dealer)
        __sampler = GroupBatchRandomSampler(
            __dataset, config["batch_size"], config["group_interval"])
        __loader = DataLoader(
            dataset=__dataset, 
            batch_sampler=__sampler, 
            collate_fn=lambda x: collate_fn(x, pad_value, device))
        return __loader
    
    train_loader = get_loader("train")
    test_loader = get_loader("test")
    valid_loader = get_loader("dev")
    return train_loader, test_loader, valid_loader

def init_cls_token(bart, dic_cls_id, triv_tokenizer):
    '''在bart模型中初始化特殊标记的嵌入向量'''
    num_tokens, _ = bart.encoder.embed_tokens.weight.shape
    bart.resize_token_embeddings(len(dic_cls_id)+num_tokens)
    for tok,val in dic_cls_id.items():
        char_idx = triv_tokenizer.convert_tokens_to_ids(
            triv_tokenizer.tokenize(tok.strip('<>'))
        )
        embed = bart.encoder.embed_tokens.weight.data[char_idx[0]]
        for i in char_idx:
            embed += bart.encoder.embed_tokens.weight.data[i]
        embed /= len(char_idx)
        bart.encoder.embed_tokens.weight.data[val] = embed

def get_model(config, dic_cls_id):
    '''初始化模型、优化器、学习率函数'''
    model_path = config['bart_path']
    bart = BartModel.from_pretrained(model_path).to(device)
    triv_tokenizer = MyTokenizer.from_pretrained(model_path)
    init_cls_token(bart, dic_cls_id, triv_tokenizer)
    # print('77', bart.decoder.embed_tokens.weight.data[21144])
    loss_fn = CrossEntropyLossWithMask()
    model = HiBart(bart, loss_fn, config).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    total_step = config["total_steps"]
    batch_sched = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.1*total_step,
        num_training_steps=total_step)
    epoch_sched = optim.lr_scheduler.MultiStepLR(
        optimizer, [config["scheduler_step"]], gamma=0.1)
    return model, optimizer, batch_sched, epoch_sched

def train():
    config_path = 'config.json'
    config, OUTPUT_DIR, logger = get_config_dir_logger(config_path)
    try:
        tokenizer = get_tokenizer(config)
        loaders = get_data_loader(config, tokenizer)
        train_loader, test_loader, valid_loader = loaders
        config['eos_id'] = tokenizer.eos_token_id
        config["total_steps"] = config["epochs"] * len(train_loader)
        models = get_model(config, tokenizer.dic_cls_id)
        model, optimizer, batch_sched, epoch_sched = models
        logger("Init model.")
    except KeyboardInterrupt:
        logger("Interrupted.")
        logger.fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
        print(traceback.format_exc())
    
    try:
        logger("Begin training.")
        accum_loss = []
        best_f1, best_f1_word = 0., 0.
        best_epoch = -1
        optimizer.zero_grad()
        for epoch in range(config["epochs"]):
            model.train()
            for i, batch in enumerate(train_loader):
                loss = model(
                    batch['enc_src_ids'],
                    batch['enc_src_len'],
                    batch['enc_mask'],
                    batch['dec_src_ids_bund'],
                    batch['dec_mask_bund'],
                    batch['dec_targ_pos_bund']
                )
                loss.backward()
                accum_loss.append(loss.item())
                if (i + 1) % int(config["grad_accum_step"]) == 0:
                    optimizer.step()
                    batch_sched.step()
                    optimizer.zero_grad()
                if (i + 1) % int(config["show_loss_step"]) == 0:
                    mean_loss = sum(accum_loss) / len(accum_loss)
                    logger("Epoch %d, step %d / %d, loss = %.4f" % (
                        epoch+1, i+1, len(train_loader), mean_loss
                    ))
                    accum_loss = []
            epoch_sched.step()

    except KeyboardInterrupt:
        logger("Interrupted.")
        logger.fp.close()
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        print(traceback.format_exc())
    
if __name__=='__main__':
    train()
