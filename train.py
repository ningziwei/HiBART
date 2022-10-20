import os
import json
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from transformers import get_linear_schedule_with_warmup

import utils
from data_pipe import *
from dataset import *
from model.modeling_bart import BartModel
from model.hi_bart import HiBart
from model.losses import CrossEntropyLossWithMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def get_logger_dir(config):
    '''
    input
        config: 配置参数
    output
        logger: 日志记录仪
        OUTPUT_DIR: 模型存储路径
    '''
    output_path = config["output_path"]
    prefix = config.get("prefix", "HiBart") + '_'
    curr_time = time.strftime("%Y%m%d%H%M", time.localtime())
    OUTPUT_DIR = os.path.join(output_path, prefix + curr_time)
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rf %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.system("cp %s %s" % (config['config_path'], OUTPUT_DIR))
    logger = utils.Logger(open(os.path.join(OUTPUT_DIR, "log.txt"), 'w'))
    length = max([len(arg) for arg in config.keys()])
    for arg, value in config.items():
        logger("%s | %s" % (arg.ljust(length).replace('_', ' '), str(value)))
    return logger, OUTPUT_DIR

def get_tokenizer(config):
    '''
    解析训练集标签，得到模型添加特殊标记的tokenizer
    '''
    dataset_dir = os.path.join(config['data_dir'], config['dataset'])
    cls_token_path = os.path.join(dataset_dir, 'cls_token.json')
    if not os.path.exists(cls_token_path):
        file_path = os.path.join(dataset_dir, 'train.train')
        sentences = parse_CoNLL_file(file_path)
        cls_token_cache = parse_label(sentences, cls_token_path)
    else:
        with open(cls_token_path, encoding="utf-8") as fp:
            cls_token_cache = json.load(fp)
    cls_tok_dic = cls_token_cache['cls_tok_dic']
    new_tokens_bundle = cls_token_cache['new_tokens_bundle']
    my_tokenizer = MyTokenizer.from_pretrained(config['model_path'])
    my_tokenizer.add_special_tokens(
        cls_tok_dic, new_tokens_bundle, config['fold'])
    
    return my_tokenizer

def get_data_loader(config, data_dealer):
    '''得到三个数据集的dataloader'''
    device = config['device']
    pad_value = config['pad_value']
    dataset_dir = os.path.join(config['data_dir'], config['dataset'])

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
    for tok, val in dic_cls_id.items():
        char_idx = triv_tokenizer.convert_tokens_to_ids(
            triv_tokenizer.tokenize(tok.strip('<>'))
        )
        embed = bart.encoder.embed_tokens.weight.data[char_idx[0]]
        for c_i in char_idx[1:]:
            embed += bart.encoder.embed_tokens.weight.data[c_i]
        embed /= len(char_idx)
        embed = embed.new_tensor(embed, requires_grad=True)
        bart.encoder.embed_tokens.weight.data[val] = embed

def get_model_optim_sched(config, dic_cls_id):
    '''初始化模型、优化器、学习率函数'''
    device = config['device']
    model_path = config['model_path']
    bart = BartModel.from_pretrained(model_path).to(device)
    triv_tokenizer = MyTokenizer.from_pretrained(model_path)
    init_cls_token(bart, dic_cls_id, triv_tokenizer)
    # print('77', bart.decoder.embed_tokens.weight.data[21144])
    loss_fn = CrossEntropyLossWithMask()
    model = HiBart(bart, loss_fn, config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    total_step = config["total_steps"]
    batch_sched = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.1*total_step,
        num_training_steps=total_step)
    epoch_sched = optim.lr_scheduler.MultiStepLR(
        optimizer, config["scheduler_step"], gamma=0.1)
    return model, optimizer, batch_sched, epoch_sched

def calib_pred(preds, end_pos, fold):
    '''矫正解码的错误'''
    new_preds = []
    for ent in preds:
        for i in range(1,len(ent)):
            if ent[i] in end_pos and i<len(ent)-(fold-2):
                new_preds.append(ent[:i+fold-1])
                new_ent = ent[1:i]
                for j in range(1,len(new_ent)):
                    if new_ent[j]-new_ent[j-1]!=1:
                        # new_preds[-1] = ent[j:i+2]
                        new_preds = new_preds[:-1]
                        # print('159', ent)
                        # print(ent[j+1:i+2])
                        break
                break
    return new_preds

def evaluate(model, loader, rotate_pos_cls, ent_end_pos, fold):
    with torch.no_grad():
        model.eval()
        predicts, labels = [], []
        for batch in loader:
            # if len(batch['targ_ents'][0])==0: continue
            pred = model(
                batch['enc_src_ids'],
                batch['enc_src_len'],
                batch['enc_mask'],
                dec_src_ids_bund=batch['dec_src_ids_bund'],
                dec_src_pos_bund=batch['dec_src_pos_bund'],
                dec_mask_bund=batch['dec_mask_bund']
            )
            # pred = [get_targ_ents(p, rotate_pos_cls) for p in pred]
            # pred = [calib_pred(p, ent_end_pos) for p in pred]
            if fold==2:
                get_ents = get_targ_ents_2
                end_pos = rotate_pos_cls[1]
            else:
                get_ents = get_targ_ents_3
                end_pos = [ent_end_pos]
            ent_pred = [get_ents(p, rotate_pos_cls, ent_end_pos) for p in pred]
            ent_pred = [calib_pred(p, end_pos, fold) for p in ent_pred]

            predicts += ent_pred
            labels += batch['targ_ents']
            # print('train 194')
            # print(pred[0])
            # print(ent_pred[0])
            # print(batch['targ_ents'][0])
            # for p, lab in zip(pred, batch['targ_ents']):
            #     print('163', p, lab)
        model.train()
        ep, er, ef = utils.micro_metrics(predicts, labels)
    return ep, er, ef

def get_train_range(epoch, fold):
    '''
    根据epoch生成不同的range
    控制用哪一阶段做训练
    '''
    if fold==2: return range(2)
    return range(3)
    if epoch<10:
        return [0]
    elif epoch<20:
        return [1]
    elif epoch<25:
        return [2]
    else:
        return range(3)

def deal_pre_conf(model_path, tokenizer):
    pre_conf = PretrainedConfig.from_pretrained(model_path)
    pre_conf.tag_num = len(tokenizer.dic_cls_id)
    pre_conf.save_pretrained(model_path)

def train(config):
    logger, OUTPUT_DIR = get_logger_dir(config)
    config['device'] = device
    # 初始化分词器、数据集和模型
    try:
        tokenizer = get_tokenizer(config)
        deal_pre_conf(config['model_path'], tokenizer)
        config['eos_id'] = tokenizer.eos_token_id
        config['pad_value'] = tokenizer.pad_token_id
        config['dic_hir_pos_cls'] = tokenizer.dic_hir_pos_cls
        data_dealer = DataDealer(tokenizer, fold=config['fold'])
        rotate_pos_cls = data_dealer.rotate_pos_cls
        ent_end_pos = list(tokenizer.dic_ent_end_pos_cls.keys())[0]
        loaders = get_data_loader(config, data_dealer)
        train_loader, test_loader, valid_loader = loaders
        config["total_steps"] = config["epochs"] * len(train_loader)
        m_o_s = get_model_optim_sched(config, tokenizer.dic_cls_id)
        model, optimizer, batch_sched, epoch_sched = m_o_s
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
    
    # 训练模型
    torch.set_printoptions(precision=6)
    try:
        logger("Begin training.")
        accum_loss = []
        best_f1 = 0.
        best_epoch = -1
        optimizer.zero_grad()
        step = 0
        for epoch in range(config["epochs"]):
            model.train()
            train_range = get_train_range(epoch, config['fold'])
            for batch in train_loader:
                # print('261 targ_ents')
                # for k,v in batch.items():
                #     print(k,v[0])
                step += 1
                loss, pred = model(
                    batch['enc_src_ids'],
                    batch['enc_src_len'],
                    batch['enc_mask'],
                    dec_src_ids_bund=batch['dec_src_ids_bund'],
                    dec_mask_bund=batch['dec_mask_bund'],
                    dec_targ_pos_bund=batch['dec_targ_pos_bund'],
                    train_range=train_range
                )
                # if (step+1)%70==0:
                #     for p, lab in zip(pred, batch['targ_ents']):
                #         p = [x.item() for x in p]
                #         print('219', p, lab)
                loss.backward()
                if step % int(config["grad_accum_step"]) == 0:
                    optimizer.step()
                    batch_sched.step()
                    optimizer.zero_grad()
                accum_loss.append(loss.item())
                if step % int(config["show_loss_step"]) == 0:
                    # print('train 273', model.encoder.embed_tokens.weight.data[21144][5:10])
                    # print('train 274', model.encoder.embed_tokens.weight.data[21128][5:10])
                    # print('train 275', model.encoder.embed_tokens.weight.data[101][5:10])
                    # print('train 276', model.encoder.embed_tokens.weight.data[3173][5:10])
                    # print('train 273', model.encoder.embed_tokens.weight.data[21144].requires_grad)
                    # print('train 273', model.encoder.embed_tokens.weight.data[21144][0].requires_grad)
                    # print('train 274', model.encoder.embed_tokens.weight.data[21128].requires_grad)
                    # print('train 275', model.encoder.embed_tokens.weight.data[101].requires_grad)
                    # print('train 276', model.encoder.embed_tokens.weight.data[3173].requires_grad)
                    mean_loss = sum(accum_loss) / len(accum_loss)
                    logger("Epoch %d, step %d / %d, loss = %.4f" % (
                        epoch+1, step, len(train_loader), mean_loss
                    ))
                    accum_loss = []
            epoch_sched.step()

            if epoch>=20:
                valid_metrics = evaluate(
                    model, valid_loader, rotate_pos_cls, ent_end_pos, config['fold'])
                vep, ver, vef = [m*100 for m in valid_metrics]
                logger("Epoch %d, valid entity p = %.2f%%, r = %.2f%%, f = %.2f%%" % (epoch + 1, vep, ver, vef))
                test_metrics = evaluate(
                    model, test_loader, rotate_pos_cls, ent_end_pos, config['fold'])
                tep, ter, tef = [m*100 for m in test_metrics]
                logger("Epoch %d, test  entity p = %.2f%%, r = %.2f%%, f = %.2f%%" % (epoch + 1, tep, ter, tef))
                if tef > best_f1:
                    best_f1 = tef
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "snapshot.model"))
                    logger("Epoch %d, save model." % (epoch+1))
        logger("Best epoch %d, best entity f1: %.2f%%" % (best_epoch+1, best_f1))
        logger.fp.close()
    except KeyboardInterrupt:
        logger("Interrupted.")
        logger.fp.close()
    except Exception as e:
        import traceback
        logger("Got exception.")
        logger.fp.close()
        print(traceback.format_exc())

def predict(config):
    state_dict_path = 'HiBart_202210121512//snapshot.model'
    state_dict_path = os.path.join(config["output_path"], state_dict_path)
    config['device'] = device
    # 初始化分词器、数据集和模型
    tokenizer = get_tokenizer(config)
    config['eos_id'] = tokenizer.eos_token_id
    config['pad_value'] = tokenizer.pad_token_id
    config['dic_hir_pos_cls'] = tokenizer.dic_hir_pos_cls
    data_dealer = DataDealer(tokenizer, fold=config['fold'])
    rotate_pos_cls = data_dealer.rotate_pos_cls
    ent_end_pos = list(tokenizer.dic_ent_end_pos_cls.keys())[0]
    loaders = get_data_loader(config, data_dealer)
    train_loader, test_loader, valid_loader = loaders
    config["total_steps"] = config["epochs"] * len(train_loader)
    model = get_model_optim_sched(config, tokenizer.dic_cls_id)[0]

    model.load_state_dict(torch.load(state_dict_path))
    valid_metrics = evaluate(
        model, valid_loader, rotate_pos_cls, ent_end_pos, config['fold'])
    vep, ver, vef = [m*100 for m in valid_metrics]
    print("Epoch %d, valid entity p = %.2f%%, r = %.2f%%, f = %.2f%%" % (1, vep, ver, vef))
    test_metrics = evaluate(
        model, test_loader, rotate_pos_cls, ent_end_pos, config['fold'])
    tep, ter, tef = [m*100 for m in test_metrics]
    print("Epoch %d, test  entity p = %.2f%%, r = %.2f%%, f = %.2f%%" % (1, tep, ter, tef))

    

if __name__=='__main__':
    config_path = 'config.json'
    with open(config_path, encoding="utf-8") as fp:
        config = json.load(fp)
    config['config_path'] = config_path
    torch.autograd.set_detect_anomaly(True)
    with torch.autograd.detect_anomaly():
        train(config)
    # predict(config)
