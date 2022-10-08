
from model.tokenizer_full import BertTokenizer

def parse_CoNLL_file(filename):
    '''
    加载CoNLL格式的数据集
    sentences: [
        [
            {'word':'感','tag':'o'},{'word':'动','tag':'o'},
            {'word':'中','tag':'b-loc.nam'},{'word':'国','tag':'i-loc.nam'}
        ]
    ]
    '''
    sentences = [] 
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        line = line.strip()
        if not line:
            if not sentences or len(sentences[-1]):
                sentences.append([])
            continue
        line_strlist = line.split()
        if line_strlist[0] != "-DOCSTART-":
            word = line_strlist[0]
            tag = line_strlist[-1].lower()
            sentences[-1].append({'word':word, 'tag':tag})
    if not sentences[-1]:
        sentences.pop()
    return sentences

def parse_label(sentences, ent_end_tok='<<ent_end>>'):
    '''
    得到实体抽取数据集的所有标签和实体类别
    new_tokens_bundle: [
        ['<<b-loc.nam-s>>', '<<b-loc.nam-e>>', '<<ent_end>>'],
        ['<<b-loc.nam-s>>'],
        ['<<b-loc.nam-e>>', '<<ent_end>>']
    ]
    '''
    label_dic = {}
    for sent in sentences:
        for s in sent:
            label_dic[s['tag']] = True
    classes = [
        lab[2:] for lab in label_dic if '-' in lab]
    cls_tok_dic = {
        lab: [f'<<{lab}-s>>', f'<<{lab}-e>>'] for lab in classes
    }
    new_tokens = []
    start_tokens = []
    end_tokens = []
    for _, v in cls_tok_dic.items():
        new_tokens += v
        start_tokens.append(v[0])
        end_tokens.append(v[1])
    new_tokens.append(ent_end_tok)
    end_tokens.append(ent_end_tok)

    return cls_tok_dic, [new_tokens, start_tokens, end_tokens]

class MyTokenizer(BertTokenizer):
    def add_special_tokens(self, cls_tok_dic, new_tokens_bundle):
        '''将解码序列中要用到的特殊标记添加到分词器中''' 
        self.cls_tok_dic = cls_tok_dic
        new_tokens, start_tokens, end_tokens = new_tokens_bundle
        self.unique_no_split_tokens += new_tokens
        self.add_tokens(new_tokens)

        dic_cls_id = {}
        dic_cls_order = {}
        for tok in new_tokens:
            dic_cls_id[tok] = self.convert_tokens_to_ids(tok)
            dic_cls_order[tok] = len(dic_cls_order)
        dic_cls_pos = {k:v+2 for k,v in dic_cls_order.items()}
        dic_start_pos_cls = {dic_cls_pos[k]:k for k in start_tokens}
        dic_end_pos_cls = {dic_cls_pos[k]:k for k in end_tokens}
        
        self.dic_cls_id = dic_cls_id
        self.dic_cls_order = dic_cls_order
        self.dic_cls_pos = dic_cls_pos
        self.dic_start_pos_cls = dic_start_pos_cls
        self.dic_end_pos_cls = dic_end_pos_cls
        
class DataDealer:
    def __init__(
        self, tokenizer, ent_end_tok='<<ent_end>>'
    ):
        self.tokenizer = tokenizer
        self.ent_end_tok = ent_end_tok
        self.tokens_to_ids = self.tokenizer.convert_tokens_to_ids

    def get_one_sample(self, sent):
        '''
        生成一个样本的输入数据和目标数据
        sent_bund: [
            ['[CLS]', '感', '动', '中', '国', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '<<loc.nam-e>>', '[SEP]']]
        targ_bund: [
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '国', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '[SEP]'], 
            ['[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '<<loc.nam-e>>', '[SEP]']]
        sent_ids_bund: [
            [101, 2697, 1220, 704, 1744, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 21137, 102]]
        sent_pos_bund: [
            [0, 19, 20, 21, 22, 1], 
            [0, 19, 20, 10, 21, 22, 1], 
            [0, 19, 20, 10, 21, 22, 18, 1], 
            [0, 19, 20, 10, 21, 22, 18, 11, 1]]
        targ_ids_bund: [
            [101, 2697, 1220, 21136, 1744, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 102], 
            [101, 2697, 1220, 21136, 704, 1744, 21144, 21137, 102]] 
        targ_pos_bund: [
            [0, 19, 20, 10, 22, 1], 
            [0, 19, 20, 10, 21, 22, 18, 1], 
            [0, 19, 20, 10, 21, 22, 18, 11, 1]]
        
        return: {
            'raw_chars': ['[CLS]', '感', '动', '中', '国', '[SEP]'], 
            'src_toks': [
                '[CLS]', '<<per.nam-s>>', '<<per.nam-e>>', '<<gpe.nam-s>>', 
                '<<gpe.nam-e>>', '<<org.nom-s>>', '<<org.nom-e>>', '<<per.nom-s>>', 
                '<<per.nom-e>>', '<<loc.nam-s>>', '<<loc.nam-e>>', '<<loc.nom-s>>', 
                '<<loc.nom-e>>', '<<gpe.nom-s>>', '<<gpe.nom-e>>', '<<org.nam-s>>', 
                '<<org.nam-e>>', '<<ent_end>>', '感', '动', '中', '国', '[SEP]'], 
            'targ_toks': [
                '[CLS]', '感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', 
                '<<loc.nam-e>>', '[SEP]'], 
            'enc_src_ids': [
                101, 21128, 21129, 21130, 21131, 21132, 21133, 21134, 21135, 21136, 
                21137, 21138, 21139, 21140, 21141, 21142, 21143, 21144, 2697, 1220, 
                704, 1744, 102], 
            'enc_src_len': 23, 
            'dec_src_ids': [
                [101, 2697, 1220, 704, 1744], 
                [101, 2697, 1220, 21136, 704, 1744, 102], 
                [101, 2697, 1220, 21136, 704, 1744, 21144, 102]], 
            'dec_targ_pos': [
                [19, 20, 10, 22, 1], 
                [19, 20, 10, 21, 22, 18, 1], 
                [19, 20, 10, 21, 22, 18, 11, 1]], 
            'targ_ents': [[10, 21, 22, 18, 11]]
        }
        '''
        sent_bund, targ_bund, sent_pos_bund, targ_pos_bund = self.get_hier_sent(sent)
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        sent_bund = [[bos_token]+sent+[eos_token] for sent in sent_bund]
        targ_bund = [[bos_token]+targ+[eos_token] for targ in targ_bund]
        sent_ids_bund = [self.tokens_to_ids(tks) for tks in sent_bund]
        targ_ids_bund = [self.tokens_to_ids(tks) for tks in targ_bund]
        sent_pos_bund = [[0]+pos+[1] for pos in sent_pos_bund]
        targ_pos_bund = [[0]+pos+[1] for pos in targ_pos_bund]
        targ_ents = self.get_targ_ents(sent_pos_bund[-1])
        # print(sent_bund, targ_bund)
        # print(sent_ids_bund, sent_pos_bund)
        # print(targ_ids_bund, targ_pos_bund)
        dec_src_ids, dec_targ_pos = self.get_dec_src_tar(
            sent_bund, targ_bund,
            sent_ids_bund, sent_pos_bund, 
            targ_ids_bund, targ_pos_bund)
        '''在编码器的输入序列中添加表示实体的特殊标记'''
        src_toks = sent_bund[0]
        cls_toks = list(self.tokenizer.dic_cls_id.keys())
        src_toks = [src_toks[0]] + cls_toks + src_toks[1:]
        enc_src_ids = [self.tokens_to_ids(tk) for tk in src_toks]
        # print('147', src_toks, enc_src_ids)
        return {
            'raw_chars': sent_bund[0],
            'src_toks': src_toks,
            'targ_toks': sent_bund[-1],
            'enc_src_ids': enc_src_ids,
            'enc_src_len': len(enc_src_ids),
            'dec_src_ids': dec_src_ids,
            'dec_targ_pos': dec_targ_pos,
            'targ_ents': targ_ents
        }

    def get_hier_sent(self, sent):
        '''
        生成分层的输入文本和目标文本以及在源文本的位置
        认为0对应bos，1对应eos，特殊标记从2开始算，特殊标记后面才是原始文本
        输入: 源文本sent
        sent: [
            {'word':'感','tag':'O'},{'word':'动','tag':'O'},
            {'word':'中','tag':'B-LOC'},{'word':'国','tag':'I-LOC'}]
        输出: 渐进式序列生成任务解码器的输入和输出序列及对应位置序列
        sent_bund: [
            ['感', '动', '中', '国'], 
            ['感', '动', '[loc.nam-s]', '中', '国'], 
            ['感', '动', '[loc.nam-s]', '中', '国', '<<ent_end>>'], 
            ['感', '动', '[loc.nam-s]', '中', '国', '<<ent_end>>', '[loc.nam-e]']]
        targ_bund: [
            ['感', '动', '<<loc.nam-s>>', '国'], 
            ['感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>'], 
            ['感', '动', '<<loc.nam-s>>', '中', '国', '<<ent_end>>', '<<loc.nam-e>>']]
        sent_pos_bund: [
            [19, 20, 21, 22], 
            [19, 20, 10, 21, 22], 
            [19, 20, 10, 21, 22, 18], 
            [19, 20, 10, 21, 22, 18, 11]]
        targ_pos_bund:  [
            [19, 20, 10, 22], 
            [19, 20, 10, 21, 22, 18], 
            [19, 20, 10, 21, 22, 18, 11]]
        '''
        last_w = {'word':'', 'tag':'o'}
        ent_end_tok = self.ent_end_tok
        cls_tok_dic = self.tokenizer.cls_tok_dic
        dic_cls_pos = self.tokenizer.dic_cls_pos     
        word_shift = len(dic_cls_pos) + 2
        for i,s in enumerate(sent):
            s['pos'] = i + word_shift
        
        sent_bund = [[s['word'] for s in sent]]
        targ_bund = []
        sent_pos_bund = [[s['pos'] for s in sent]]
        targ_pos_bund = []
        sent1 = []
        targ1 = []
        sent = sent + [last_w]
        targ_sent = sent[1:] + [last_w]
        for w, w_tar in zip(sent, targ_sent):
            if w['tag'].startswith('b-'):
                word = cls_tok_dic[w['tag'][2:]][0]
                sent1.append({
                    'word':word,'tag':'begin','pos':dic_cls_pos[word]
                })
                targ1 = targ1[:-1]
                targ1.append(sent1[-1])
            if w_tar['word']:
                targ1.append(w_tar)
            if w['word']:
                sent1.append(w)
        sent_bund.append([x['word'] for x in sent1])
        targ_bund.append([x['word'] for x in targ1])
        sent_pos_bund.append([x['pos'] for x in sent1])
        targ_pos_bund.append([x['pos'] for x in targ1])

        sent = sent1
        sent1, sent2 = [], []
        targ1, targ2 = [], []
        sent = sent + [last_w]
        targ_sent = sent[1:] + [last_w]
        w_ = {'word':'', 'tag':''}
        # print('209', sent)
        for w, w_tar in zip(sent, targ_sent):
            if w['tag'] in ['o','begin'] and w_['tag'][:2] in ['i-','b-']:  
                # 添加实体结束标记
                sent1.append({
                    'word':ent_end_tok,'tag':'','pos':dic_cls_pos[ent_end_tok]
                })
                if w['word']: targ1 = targ1[:-1]
                targ1.append(sent1[-1])
                # 添加实体类别结束标记
                word = cls_tok_dic[w_['tag'][2:]][1]
                sent2.append({
                    'word':ent_end_tok,'tag':'','pos':dic_cls_pos[ent_end_tok]
                })
                sent2.append({
                    'word':word,'tag':'','pos':dic_cls_pos[word]
                })
                if w['word']: targ2 = targ2[:-1]
                targ2.append(sent2[-2])
                targ2.append(sent2[-1])
            if w_tar['word']:
                targ1.append(w_tar)
                targ2.append(w_tar)
            if w['word']:
                sent1.append(w)
                sent2.append(w)
            w_ = w
        sent_bund.append([x['word'] for x in sent1])
        sent_bund.append([x['word'] for x in sent2])
        targ_bund.append([x['word'] for x in targ1])
        targ_bund.append([x['word'] for x in targ2])
        sent_pos_bund.append([x['pos'] for x in sent1])
        targ_pos_bund.append([x['pos'] for x in targ1])
        sent_pos_bund.append([x['pos'] for x in sent2])
        targ_pos_bund.append([x['pos'] for x in targ2])
        for i, (sent, targ, sent_pos, targ_pos) in enumerate(zip(
            sent_bund, targ_bund, sent_pos_bund, targ_pos_bund)):
            if targ[0] not in dic_cls_pos:
                targ_bund[i] = [sent[0]] + targ
                targ_pos_bund[i] = [sent_pos[0]] + targ_pos
        return sent_bund, targ_bund, sent_pos_bund, targ_pos_bund

    def get_dec_src_tar(
        self, sent_bund, targ_bund,
        sent_ids_bund, sent_pos_bund, 
        targ_ids_bund, targ_pos_bund):
        '''
        得到解码器输入和目标的token、id、pos列表
        实际上有用的只有sent的ids和targ的pos
        '''
        dec_src_toks = []
        dec_targ_toks = []
        dec_src_ids = []
        dec_src_pos = []
        dec_targ_ids = []
        dec_targ_pos = []
        # print('278')
        for i in range(len(targ_pos_bund)):
            sent_toks = sent_bund[i]
            sent_ids = sent_ids_bund[i]
            sent_pos = sent_pos_bund[i]
            targ_toks = targ_bund[i]
            targ_ids = targ_ids_bund[i]
            targ_pos = targ_pos_bund[i]
            # 目标序列的最后一位一定要是[SEP]
            dec_src_toks.append(sent_toks[:len(targ_pos)-1])
            dec_src_ids.append(sent_ids[:len(targ_pos)-1])
            dec_src_pos.append(sent_pos[:len(targ_pos)-1])
            dec_targ_toks.append(targ_toks[1:])
            dec_targ_ids.append(targ_ids[1:])
            dec_targ_pos.append(targ_pos[1:])
            # print(dec_src_toks[-1])
            # print(dec_targ_toks[-1])
            # print(dec_src_ids[-1], dec_targ_ids[-1])
            
        return dec_src_ids, dec_targ_pos

    def get_targ_ents(self, pos_list):
        '''得到序列中的实体'''
        i, N = 0, len(pos_list)
        rotate_pos_cls = []
        rotate_pos_cls.append(self.tokenizer.dic_start_pos_cls)
        rotate_pos_cls.append(self.tokenizer.dic_end_pos_cls)

        ents = []
        while i<N:
            # 碰到实体开始符
            if pos_list[i] in rotate_pos_cls[0]:
                ent = [pos_list[i]]
                i += 1
                while i<N:
                    ent.append(pos_list[i])
                    # 碰到实体结束符
                    if pos_list[i] in rotate_pos_cls[1]:
                        i += 1
                        while i<N:
                            if pos_list[i] in rotate_pos_cls[1]:
                                ent.append(pos_list[i])
                            else:
                                break
                            i += 1
                        break
                    i += 1
                ents.append(ent)
            else:
                i += 1
        return ents

def padding(data, pad_value=0, dim=2):
    '''
    pad data to maximum length
    data: list(list), unpadded data
    pad_value: int, filled value
    dim: int, dimension of padded data
        dim=2, result=(batch_size, sen_len)
        dim=3, result=(batch_size, sen_len, word_len)
    '''
    sen_len = max([len(d) for d in data])
    if dim == 2:
        padded_data = [d + [pad_value] * (sen_len-len(d)) for d in data]
        padded_mask = [[1] * len(d) + [0] * (sen_len-len(d)) for d in data]
        return padded_data, padded_mask
    elif dim == 3:
        word_len = max([max([len(dd) for dd in d]) for d in data])
        padded_data = []
        padded_mask = []
        for d in data:
            padded_data.append([])
            padded_mask.append([])
            for dd in d:
                padded_data[-1].append(dd + [pad_value] * (word_len-len(dd)))
                padded_mask[-1].append([1] * len(dd) + [0] * (word_len-len(dd)))
            for _ in range(sen_len - len(d)):
                padded_data[-1].append([pad_value] * word_len)
                padded_mask[-1].append([0] * word_len)
        return padded_data, padded_mask
    else:
        raise NotImplementedError("Dimension %d not supported! Legal option: 2 or 3." % dim)




    









