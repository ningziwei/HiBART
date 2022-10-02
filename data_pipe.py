
import enum
from transformers import BertTokenizer

class DataLoader:
    '''
    读取数据集，解析标签类别
    '''
    def __init__(self, ent_end_tok='[ent_end]'):
        self.ent_end_tok = ent_end_tok
    
    def parse_data(self, filename):
        '''解析训练数据'''
        self.sentences = self.parse_CoNLL_file(filename)
        self.class_dic, self.new_tokens = self.parse_label()
        self.sentence_bundle, self.target_bundle = self.hier_data()

    def parse_CoNLL_file(self, filename):
        '''
        加载CoNLL格式的数据集
        sentences: [
            [{'word':w1, 'tag':t1}, {...}, ...], 
            [...]
        ]
        '''
        # 
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

    def parse_label(self):
        '''
        得到实体抽取数据集的所有标签和实体类别
        '''
        label_dic = {}
        for sent in self.sentences:
            for s in sent:
                label_dic[s['tag']] = True
        classes = [
            lab[2:] for lab in label_dic if '-' in lab]
        class_dic = {
            lab: [f'[[{lab}-s]]', f'[[{lab}-e]]'] for lab in classes
        }
        new_tokens = []
        for _, v in class_dic.items():
            new_tokens += v
        new_tokens.append(self.ent_end_tok)
        
        return class_dic, new_tokens

class MyTokenizer(BertTokenizer):
    def add_special_tokens(self, new_tokens):
        '''将解码序列中要用到的特殊标记添加到分词器中'''        
        self.unique_no_split_tokens += new_tokens
        self.add_tokens(new_tokens)

        dic_cls_id = {}
        dic_cls_order = {}
        for tok in new_tokens:
            dic_cls_id[tok] = self.convert_tokens_to_ids(tok)
            dic_cls_order[tok] = len(dic_cls_order)
        self.dic_cls_id = dic_cls_id
        self.dic_cls_order = dic_cls_order
        
class DataDealer:
    def __init__(
        self, data_loader, tokenizer,
        ent_end_tok='[ent_end]'
    ):
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.ent_end_tok = ent_end_tok
    
    def convert_tokens_to_ids(self, token_list):
        '''将一个token列表转化为id列表'''
        ids = [self.tokenizer.bos_token_id]
        token_to_id = self.tokenizer.convert_tokens_to_ids
        ids += [token_to_id(w) for w in token_list]
        ids += [self.tokenizer.eos_token_id]
        return ids

    def get_one_sample(self, sent):
        '''生成一个样本的输入数据和目标数据'''
        sent_bund, targ_bund, sent_pos_bund, targ_pos_bund = self.get_hier_sent(sent)
        sent_bund_ids = [self.convert_tokens_to_ids(tks) for tks in sent_bund]
        targ_bund_ids = [self.convert_tokens_to_ids(tks) for tks in targ_bund]
        sent_pos_bund = [[0]+pos+[1] for pos in sent_pos_bund]
        targ_pos_bund = [[0]+pos+[1] for pos in targ_pos_bund]
        
        return {
            'chars':sent,'sent_ids':sent_bund_ids,'targ_ids':targ_bund_ids,
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
            ['感', '动', '[loc.nam-s]', '中', '国', '[ent_end]'], 
            ['感', '动', '[loc.nam-s]', '中', '国', '[ent_end]', '[loc.nam-e]']]
        targ_bund: [
            ['动', '[loc.nam-s]', '国'], 
            ['动', '[loc.nam-s]', '中', '[ent_end]'], 
            ['动', '[loc.nam-s]', '中', '[ent_end]', '[loc.nam-e]']]
        sent_pos_bund: [
            [19, 20, 21, 22], 
            [19, 20, 10, 21, 22], 
            [19, 20, 10, 21, 22, 18], 
            [19, 20, 10, 21, 22, 18, 11]]
        targ_pos_bund: [
            [20, 10, 22], 
            [20, 10, 21, 18], 
            [20, 10, 21, 18, 11]]
        '''
        last_w = {'word':'', 'tag':'o'}
        ent_end_tok = self.ent_end_tok
        dic_cls_order = self.tokenizer.dic_cls_order     
        word_shift = len(dic_cls_order) + 2
        dic_cls_pos = {k:v+2 for k,v in dic_cls_order.items()}
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
                word = '[[{}-s]]'.format(w['tag'][2:])
                sent1.append({
                    'word':word,'tag':'','pos':dic_cls_pos[word]
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
        for w, w_tar in zip(sent, targ_sent):
            if w['tag']=='o' and w_['tag'][:2] in ['i-', 'b-']:  
                # 添加实体结束标记
                sent1.append({
                    'word':ent_end_tok,'tag':'','pos':dic_cls_pos[ent_end_tok]
                })
                targ1 = targ1[:-1]
                targ1.append(sent1[-1])
                # 添加实体类别结束标记
                word = '[[{}-e]]'.format(w_['tag'][2:])
                sent2.append({
                    'word':ent_end_tok,'tag':'','pos':dic_cls_pos[ent_end_tok]
                })
                sent2.append({
                    'word':word,'tag':'','pos':dic_cls_pos[word]
                })
                targ2 = targ2[:-1]
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

        return sent_bund, targ_bund, sent_pos_bund, targ_pos_bund

    def get_targ_ent(self, sent):
        '''得到序列中的实体'''




    









