import json
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence


FILE = "data/fold%s_%s.json"
out_dir = 'predictions'
LEXICON = True


class ECEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class ECEDataset_pred(torch.utils.data.Dataset):
    def __init__(self, encodings, passages, doc_pairs, pred_sentiments_ids):
        self.encodings = encodings
        self.passages = passages
        self.doc_pairs = doc_pairs
        self.pred_sentiment_ids = pred_sentiments_ids
        self.clauses_positions = self.get_clauses_positions()

    def __getitem__(self, idx):
        result = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        result['doc_pairs'] = self.doc_pairs[idx]
        result['senti_ids'] = self.pred_sentiment_ids[idx]
        result['clauses_positions'] = self.clauses_positions[idx]
        return result

    def __len__(self):
        return len(self.encodings.input_ids)

    def get_clauses_positions(self):
        clauses_positions = []
        for passage in self.passages:
            clause_position = [0]
            temp = passage.split(sep='，')
            for item in temp:
                clause_position.append(clause_position[-1] + len(item) + 1)  # 加上clause的长度 再加上标点的长度
            clauses_positions.append(clause_position)

        return clauses_positions


def semc(doc):
    e_c_dict = {}
    for e, c in doc['pairs']:
        if e not in e_c_dict.keys():
            e_c_dict[e] = [c]
        else:
            e_c_dict[e].append(c)
    flag = False
    for cs in e_c_dict.values():
        cs.sort()
        if len(cs) > 1 and (cs[-1] - cs[0]) == len(cs) - 1:
            flag = True

    if not flag:
        return (flag, )
    else:
        passages = []
        sentiments = []
        causes = []
        causes_ids = []
        clauses = []
        for clause in doc['clauses']:
            clauses.append(clause['clause'])
        passage = '，'.join(clauses) + '。'
        for e, cs in e_c_dict.items():
            cs.sort()
            if len(cs) > 1 and (cs[-1] - cs[0]) == len(cs) - 1:
                sentiments.append(clauses[e-1])
                causes_ids.append('&'+'&'.join([str(c) for c in cs]))
                causes.append('&'.join([clauses[c-1] for c in cs]))
                passages.append(passage)
            else:
                for c in cs:
                    sentiments.append(clauses[e-1])
                    causes.append(clauses[c-1])
                    causes_ids.append(str(c))
                    passages.append(passage)

        return (flag, passages, sentiments, causes, causes_ids)


def read_from_json_pred(fold_id):
    pred_sentiments_text = []
    pred_sentiments_ids = []
    pred_docs = []
    doc_pairs = []
    # 以上三个数量等于总的预测情感句的个数
    filename = FILE % (fold_id, 'test')
    with open(filename, encoding='utf-8') as f:
        js = json.load(f)
    all_preds = []
    if not LEXICON:
        emotion_file = f"fold{fold_id}predicted_result.txt"
    else:
        emotion_file = f"fold{fold_id}predicted_result_revised.txt"
    with open(os.path.join(out_dir,emotion_file)) as f:
        for line in f.readlines():
            all_preds.append(int(line.strip()))
    current_index = 0
    for doc in js:

        doc_preds = all_preds[current_index:current_index+doc['doc_len']]
        current_index += doc['doc_len']
        # 验证多组合效果时使用
        # if len(doc['pairs']) < 2:
        #     continue
        doc_preds = [i+1 for i in range(doc['doc_len']) if doc_preds[i] == 1 ] # index从1开始
        clauses = []
        for clause in doc['clauses']:
            clauses.append(clause['clause'])
        passage = '，'.join(clauses) + '。'
        if len(passage) > 480:
            print("long doc!")
            doc_len1 = doc['doc_len'] // 2
            doc_len2 = doc['doc_len'] - doc_len1
            passage1 = '，'.join(clauses[:doc_len1]) + '。'
            passage2 = '，'.join(clauses[doc_len1:]) + '。'
            doc_preds1 = [item for item in doc_preds if item <= doc_len1]
            doc_preds2 = [item for item in doc_preds if item > doc_len1]
            for pred_s in doc_preds1:
                pred_sentiments_ids.append(pred_s)
                for clause in doc['clauses']:
                    if clause['clause_id'] == str(pred_s): # str类型
                        pred_sentiments_text.append(clause['clause'])

                pred_docs.append(passage1)
                doc_pairs.append([pair for pair in doc['pairs'] if pair[0]<=doc_len1])
            for pred_s in doc_preds2:
                pred_sentiments_ids.append(pred_s-doc_len1)
                for clause in doc['clauses']:
                    if clause['clause_id'] == str(pred_s): # str类型
                        pred_sentiments_text.append(clause['clause'])
                pred_docs.append(passage2)
                doc_pairs.append([[pair[0]-doc_len1,pair[1]-doc_len1] for pair in doc['pairs'] if pair[0]>doc_len1])
        else:
            for pred_s in doc_preds:
                pred_sentiments_ids.append(pred_s)
                for clause in doc['clauses']:
                    if clause['clause_id'] == str(pred_s): # str类型
                        pred_sentiments_text.append(clause['clause'])

                pred_docs.append(passage)
                doc_pairs.append(doc['pairs'])

    return pred_docs, pred_sentiments_text, doc_pairs, pred_sentiments_ids


def split_doc(doc, doc_len, clauses):
    passages = []
    sentiments = []
    causes = []
    causes_ids = []

    doc_len1 = doc_len // 2
    doc_len2 = doc_len - doc_len1
    passage1 = '，'.join(clauses[:doc_len1]) + '。'
    passage2 = '，'.join(clauses[doc_len1:]) + '。'

    for i in range(len(doc['pairs'])):
        passage = []
        for j,clause in enumerate(doc['clauses']):
            if clause['clause_id'] == str(doc['pairs'][i][0]):
                sentiments.append(clause['clause'])
            if clause['clause_id'] == str(doc['pairs'][i][1]):
                causes.append(clause['clause'])
                causes_ids.append(clause['clause_id'])
        if doc['pairs'][i][1] < doc_len1:
            passages.append(passage1)
        else:
            passages.append(passage2)
            causes_ids[-1] = str(int(clause['clause_id']) - doc_len1)

    return passages, sentiments, causes, causes_ids


def read_from_json(fold_id,data_type):
    passages = []
    sentiments = []
    causes = []
    causes_ids = []
    # 以上三个数量应大于总文档数，等于总的pair的个数
    filename = FILE %(fold_id, data_type)
    with open(filename, encoding='utf-8') as f:
        js = json.load(f)

    for doc in js:
        # 验证多组合效果时使用
        # if len(doc['pairs']) < 2:
        #     continue

        # 处理过长文档时使用
        clauses = []
        for clause in doc['clauses']:
            clauses.append(clause['clause'])
        doc_len = doc['doc_len']
        if len('，'.join(clauses) + '。') > 480:
            print("Long document. Deal with it.")
            # print(doc['pairs'])
            p, s, c, c_i= split_doc(doc, doc_len, clauses)
            passages.extend(p)
            sentiments.extend(s)
            causes.extend(c)
            causes_ids.extend(c_i)
            continue

        # 处理一感情对多原因时使用
        if semc(doc)[0] == False:
            for i in range(len(doc['pairs'])):
                passage = []
                for clause in doc['clauses']:
                    passage.append(clause['clause'])
                    if clause['clause_id'] == str(doc['pairs'][i][0]):
                        sentiments.append(clause['clause'])
                    if clause['clause_id'] == str(doc['pairs'][i][1]):
                        causes.append(clause['clause'])
                        causes_ids.append(clause['clause_id'])
                passage = '，'.join(passage)+'。'
                passages.append(passage)
        else:
            # print('semc!')
            # print(doc['pairs'])
            p, s, c, c_i = semc(doc)[1:]
            passages.extend(p)
            sentiments.extend(s)
            causes.extend(c)
            causes_ids.extend(c_i)
    return passages, sentiments, causes, causes_ids


def build_dataloader(fold_id, data_type, label_type, senti_type):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if senti_type != 'predict':
        passages, sentiments, causes, causes_ids = read_from_json(fold_id, data_type)

        # encodings = tokenizer(passages, sentiments, truncation=True, padding=True)
        # Ablation studies: removing sentiments
        temp = []
        for p,s in zip(passages,sentiments):
            pos = p.find(s)
            if pos == -1:
                print("Error")
            temp.append(p[:pos]+"开始"+s+"结束"+p[pos+len(s):])
        passages = temp
        encodings = tokenizer(passages, truncation=True, padding=True)

        causes = add_end_idx(causes, causes_ids, encodings, passages)
        add_encoding_positions(encodings, causes, passages)
        dataset = ECEDataset(encodings)

        if data_type == 'train':
            train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=batch_preprocessing)
            return train_loader
        elif data_type =='test':
            valid_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=batch_preprocessing)
            return valid_loader
        else:
            print('Unknown data_type')
            exit(0)

    else:
        pred_docs, pred_sentiments_text, doc_pairs, pred_sentiments_ids = read_from_json_pred(fold_id)
        print([len(i) for i in [pred_docs, pred_sentiments_text, doc_pairs, pred_sentiments_ids]])
        temp = []
        for p,s in zip(pred_docs, pred_sentiments_text):
            pos = p.find(s)
            if pos == -1:
                print("Error")
            temp.append(p[:pos]+"开始"+s+"结束"+p[pos+len(s):])
        pred_docs = temp
        encodings = tokenizer(pred_docs,  truncation=True, padding=True)
        # add_positions(encodings, pred_docs, doc_pairs, pred_sentiments_ids)
        dataset = ECEDataset_pred(encodings,pred_docs, doc_pairs, pred_sentiments_ids)
        # print(f"dataset:\n{dataset[0]}")
        valid_loader = DataLoader(dataset, batch_size=4, shuffle=False,collate_fn=b)

        return valid_loader


def add_end_idx(causes, causes_ids, encodings, passages):
    new_causes = []
    for cause, cause_id, input_id in zip(causes, causes_ids, encodings['input_ids']):
        new_cause = {}
        new_cause['text'] = cause
        clause_ps = [0] + [(index + 1) for index, token in enumerate(input_id) if token == 8024 or token == 511]
        # print(passages,cause)
        # print(cause_id,clause_ps)
        if cause_id[0] == '&': # 多原因
            ids = cause_id[1:].split('&')
            new_cause['start_idx'] = clause_ps[int(ids[0]) - 1]
            new_cause['end_idx'] = clause_ps[int(ids[-1])]
        else:
            new_cause['start_idx'] = clause_ps[int(cause_id)-1]
            new_cause['end_idx'] = clause_ps[int(cause_id)]
        new_causes.append(new_cause)
    return new_causes


def add_encoding_positions(encodings, causes, passages):
    start_positions = []
    end_positions = []
    clauses_positions = []
    for cause in causes:
        start_positions.append(cause['start_idx'])
        end_positions.append(cause['end_idx'])
    # for passage in passages:
    #     clause_position  = [0]
    #     temp = passage.split(sep='，')
    #     for item in temp:
    #         clause_position.append(clause_position[-1] + len(item) + 1) # 加上clause的长度 再加上标点的长度
    #     clauses_positions.append(clause_position)

    for input_id in encodings['input_ids']:
        clause_position = [0] + [(index + 1) for index, token in enumerate(input_id) if token == 8024 or token == 511]
        # print(clause_position)
        clauses_positions.append(clause_position)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions,
                      'clauses_positions':clauses_positions})


def batch_preprocessing(batch):  #因为clause_position不一样长 所以要进行pad操作
    new_batch = {}
    for key in batch[0].keys():
        if key != 'clauses_positions':
            # print(key)
            # print([batch[i][key] for i in range(len(batch))])
            new_batch[key] = torch.stack(tuple([batch[i][key] for i in range(len(batch))]), dim=0)
    clause_position = [item['clauses_positions'] for item in batch]
    clause_position = pad_sequence(clause_position, batch_first=True, padding_value=-1)
    new_batch['clauses_positions'] = clause_position
    return new_batch

def b(batch):
    # print([item for item in batch])
    new_batch = {}
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        new_batch[key] = torch.stack(tuple([batch[i][key] for i in range(len(batch))]), dim=0)
    new_batch['clauses_positions'] = [batch[i]['clauses_positions'] for i in range(len(batch))]
    new_batch['doc_pairs'] = [batch[i]['doc_pairs'] for i in range(len(batch))]
    new_batch['senti_ids'] = [batch[i]['senti_ids'] for i in range(len(batch))]
    return new_batch

# test = build_dataloader(1,'train','sentiment','truth')
# for batch in test:
#     pass