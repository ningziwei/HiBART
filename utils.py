import time

class Logger(object):
    def __init__(self, fp=None):
        self.fp = fp

    def __call__(self, string, end='\n'):
        curr_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        new_string = '[%s] ' % curr_time + string
        print(new_string, end=end)
        if self.fp is not None:
            self.fp.write('%s%s' % (new_string, end))

def micro_metrics(predicts, labels):
    '''计算预测指标'''
    true_count, predict_count, gold_count = 0, 0, 0
    for pred_entity, gold_entity in zip(predicts, labels):
        pred_entity = [p[1:] for p in pred_entity]
        gold_entity = [g[1:] for g in gold_entity]
        # pred_entity = [p[:-1] for p in pred_entity]
        # gold_entity = [g[:-1] for g in gold_entity]
        for e in pred_entity:
            if e in gold_entity:
                true_count += 1
        predict_count += len(pred_entity)
        gold_count += len(gold_entity)
    ep = true_count / max(predict_count, 1)
    er = true_count / gold_count
    ef = 2 * ep * er / max((ep + er), 0.0001)
    return ep, er, ef