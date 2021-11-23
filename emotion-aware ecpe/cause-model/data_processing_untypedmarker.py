import json
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

DATASET_TYPE = "original"
if DATASET_TYPE == "original":
    FILE = "../ECPE-original/split10/fold%s_%s.json"
    out_dir = "../predictions_original"
elif DATASET_TYPE == "reconstructed":
    FILE = "../ECPE-reconstructed/split10/fold%s_%s.json"
    out_dir = "../predictions_reconstructed"
else:
    print("Unknown Dataset!")
    exit(1)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
new_tokens = ['<EMO_START>', '<EMO_END>']
tokenizer.add_tokens(new_tokens)


class ECEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class PredDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, passages, doc_pairs, pred_sentiments_ids):
        self.encodings = encodings
        self.passages = passages
        self.doc_pairs = doc_pairs
        self.pred_sentiment_ids = pred_sentiments_ids
        # self.clauses_positions = self.get_clauses_positions()

    def __getitem__(self, idx):
        result = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        result['doc_pairs'] = self.doc_pairs[idx]
        result['senti_ids'] = self.pred_sentiment_ids[idx]
        # result['clauses_positions'] = self.clauses_positions[idx]
        return result

    def __len__(self):
        return len(self.encodings.input_ids)


def compute_split(clauses):

    passage1 = '，'.join(clauses[:len(clauses)//2]) + '。'
    passage2 = '，'.join(clauses[len(clauses)//2:]) + '。'
    if len(passage1) < 480 and len(passage2) < 480:
        return 2
    else:
        return compute_split(clauses[:len(clauses)//2]) + compute_split(clauses[len(clauses)//2:])


def split_doc(doc, doc_len,clauses):

    passages = []
    sentiments = []
    causes = []
    doc_lens = []
    num_split = compute_split(clauses)
    print(f"split to {num_split} docs")
    ec_dict_list = []
    split_len = doc_len // num_split
    for i in range(1, num_split+1):
        for e, c in doc['pairs']:
            if c <= split_len*i:
                ec_dict = {}
                if e not in ec_dict.keys():
                    ec_dict[e] = [c]
                else:
                    ec_dict[e].append(c)
                ec_dict_list.append(ec_dict)

    while len(ec_dict_list) < num_split:
        print('Number of emotion-cause pairs is less than number of splits, but don\'t worry!')
        ec_dict_list.append({})

    for i in range(0, num_split-1):
        for e, cs in ec_dict_list[i-1].items():
            for clause in doc['clauses']:
                if clause['clause_id'] == str(e):
                    sentiments.append(clause['clause'])
            cause = [1 if i in cs else 0 for i in range(1 + split_len*i, split_len*(i+1) + 1)]
            passage = '，'.join(clauses[split_len*i:split_len*(i+1)]) + '。'
            passages.append(passage)
            causes.append(cause)
            doc_lens.append(split_len)

    for e, cs in ec_dict_list[num_split-1].items():
        for clause in doc['clauses']:
            if clause['clause_id'] == str(e):
                sentiments.append(clause['clause'])
        cause = [1 if i in cs else 0 for i in range(1 + split_len * (num_split-1), doc_len + 1)]
        passage = '，'.join(clauses[split_len * (num_split-1):]) + '。'
        passages.append(passage)
        causes.append(cause)
        doc_lens.append(doc_len - split_len * (num_split-1))

    return passages,sentiments,causes,doc_lens


def read_from_json(fold_id,data_type):

    passages = []
    sentiments = []
    causes = []
    doc_lens = []

    filename = FILE %(fold_id, data_type)
    with open(filename, encoding='utf-8') as f:
        js = json.load(f)
        for doc in js:
            clauses = []
            for clause in doc['clauses']:
                clauses.append(clause['clause'])

            passage = '，'.join(clauses) + '。'
            doc_len = doc['doc_len']
            # if len(passage) > 900:
            #     print("Sooooooooooooo long!")
            if len(passage) > 480:
                print(f"Long document in {data_type}. Deal with it.")
                p,s,c,dl = split_doc(doc,doc_len,clauses)
                passages.extend(p); sentiments.extend(s); causes.extend(c)
                doc_lens.extend(dl);
                continue
            e_c_dict = {}
            for e, c in doc['pairs']:
                if e not in e_c_dict.keys():
                    e_c_dict[e] = [c]
                else:
                    e_c_dict[e].append(c)

            for e, cs in e_c_dict.items():
                passage = []
                for clause in doc['clauses']:
                    passage.append(clause['clause'])
                    if clause['clause_id'] == str(e):
                        passage[-1] = '<EMO_START>' + clause['clause'] + '<EMO_END>'
                        sentiments.append(clause['clause'])

                cause = [1 if i in cs else 0 for i in range(1, len(passage)+1)] 
                passage = '，'.join(passage)+'。'
                passages.append(passage)
                causes.append(cause)
                doc_lens.append(doc_len)

    return passages, sentiments, causes, doc_lens


def split_pred(doc, doc_preds, clauses):
    pred_sentiments_ids, pred_sentiments_text, pred_docs, doc_pairs = [], [], [], []
    num_split = compute_split(clauses)
    print(f"split to {num_split} docs")
    split_len = doc['doc_len'] // num_split

    for pred_s in doc_preds:

        for clause in doc['clauses']:
            if clause['clause_id'] == str(pred_s):  # str类型
                pred_sentiments_text.append(clause['clause'])
        # 求当前情感到底在哪一个clip 例：len为4 那么 1234 在第0个clip; 5678在第1个clip
        clip_index = (pred_s-1) // split_len
        if clip_index == num_split - 1:
            passage = '，'.join(clauses[clip_index * split_len:]) + '。'
        else:
            passage = '，'.join(clauses[clip_index*split_len:(clip_index+1)*split_len]) + '。'
        pred_sentiments_ids.append(pred_s - clip_index*split_len)
        pred_docs.append(passage)
        doc_pairs.append([pair for pair in doc['pairs'] if
                          clip_index * split_len < pair[0] <= (clip_index + 1) * split_len])

    return pred_sentiments_ids, pred_sentiments_text, pred_docs, doc_pairs


def read_from_json_pred(fold_id, sub=False):
    num_of_true_pairs_all = 0
    pred_sentiments_text = []
    pred_sentiments_ids = []
    pred_docs = []
    doc_pairs = []
    # 以上三个数量等于总的预测出来的情感句的个数
    filename = FILE % (fold_id, 'test')
    with open(filename, encoding='utf-8') as f:
        js = json.load(f)
    all_preds = []
    with open(os.path.join(out_dir,f"fold{fold_id}predicted_result.txt")) as f:
        for line in f.readlines():
            all_preds.append(int(line.strip()))
    current_index = 0
    for doc in js:

        doc_preds = all_preds[current_index:current_index+doc['doc_len']]
        current_index += doc['doc_len']
        # 验证多组合效果时使用
        if len(doc['pairs']) < 2 and sub == True:
            continue
        num_of_true_pairs_all += len(doc['pairs'])
        doc_preds = [i+1 for i in range(doc['doc_len']) if doc_preds[i] == 1 ] # index从1开始
        clauses = []
        for clause in doc['clauses']:
            clauses.append(clause['clause'])
        passage = '，'.join(clauses) + '。'
        # if len(passage) > 900:
        #     print("Sooooooooooooooo long! in pred ecpe")
        if len(passage) > 480:
            # print("long doc! in ecpe pred")
            psi,pst,pdo,dp = split_pred(doc,doc_preds,clauses)
            pred_sentiments_ids.extend(psi)
            pred_sentiments_text.extend(pst)
            pred_docs.extend(pdo)
            doc_pairs.extend(dp)

        else:
            for pred_s in doc_preds:
                pred_sentiments_ids.append(pred_s)
                for clause in doc['clauses']:
                    if clause['clause_id'] == str(pred_s): # str类型
                        pred_sentiments_text.append(clause['clause'])

                pred_docs.append(passage)
                doc_pairs.append(doc['pairs'])

    print(f"number of ground truth pairs: {num_of_true_pairs_all}")

    return pred_docs, pred_sentiments_text, doc_pairs, pred_sentiments_ids, num_of_true_pairs_all


def build_dataloader(fold_id, data_type, label_type, train_batch_size=8):
    passages, sentiments, causes, doc_lens = read_from_json(fold_id, data_type)

    encodings = tokenizer(passages, truncation=True, padding=True)
    add_encoding_positions(encodings, causes, passages, doc_lens)
    dataset = ECEDataset(encodings)
    if data_type == 'train':
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=False, collate_fn=batch_preprocessing)
        return train_loader
    elif data_type =='test':
        valid_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=batch_preprocessing)
        return valid_loader
    else:
        print('Unknown data_type')
        exit(0)


def build_real_dataloader(fold_id, sub=False):
    pred_docs, pred_sentiments_text, doc_pairs, pred_sentiments_ids , num_of_true_pairs_all = read_from_json_pred(fold_id,sub=sub)
    temp = []
    for p, s in zip(pred_docs, pred_sentiments_text):
        pos = p.find(s)
        if pos == -1:
            print("Warning: Cannot find the emotion clause in the document context")
        temp.append(p[:pos] + "<EMO_START>" + s + "<EMO_END>" + p[pos + len(s):])
    pred_docs = temp

    encodings = tokenizer(pred_docs, truncation=True, padding=True)
    add_encoding_positions_pred(encodings,pred_docs)

    dataset = PredDataset(encodings, pred_docs, doc_pairs, pred_sentiments_ids)
    valid_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=batch_process_pred)

    return valid_loader, num_of_true_pairs_all


def add_encoding_positions(encodings, causes, passages, doc_lens):

    clauses_positions = []
    for input_id in encodings['input_ids']:
        clause_position = [1] + [(index + 1) for index, token in enumerate(input_id) if token == 8024 or token == 511]
        # print(clause_position)
        clauses_positions.append(clause_position)
    encodings.update({'causes': causes, 'clauses_positions':clauses_positions, 'doc_lens':doc_lens})


def add_encoding_positions_pred(encodings, passages):
    clauses_positions = []
    doc_lens = []
    for input_id in encodings['input_ids']:
        clause_position = [1] + [(index + 1) for index, token in enumerate(input_id) if token == 8024 or token == 511]
        # print(clause_position)
        clauses_positions.append(clause_position)
        doc_lens.append(len(clause_position)-1)
    encodings.update({'clauses_positions': clauses_positions, 'doc_lens': doc_lens})

def batch_preprocessing(batch):
    new_batch = {}
    for key in batch[0].keys():
        if key != 'clauses_positions' and key != 'causes':
            new_batch[key] = torch.stack(tuple([batch[i][key] for i in range(len(batch))]), dim=0)

    clause_position = [item['clauses_positions'] for item in batch]
    clause_position = pad_sequence(clause_position, batch_first=True, padding_value=-1)
    cause = [item['causes'] for item in batch]
    cause = pad_sequence(cause, batch_first=True, padding_value=-1)
    new_batch['clauses_positions'] = clause_position
    new_batch['causes'] = cause
    return new_batch


def batch_process_pred(batch):
    new_batch = {}
    for key in ['input_ids', 'token_type_ids', 'attention_mask']:
        new_batch[key] = torch.stack(tuple([batch[i][key] for i in range(len(batch))]), dim=0)
    new_batch['clauses_positions'] = [batch[i]['clauses_positions'] for i in range(len(batch))]
    new_batch['doc_pairs'] = [batch[i]['doc_pairs'] for i in range(len(batch))]
    new_batch['senti_ids'] = [batch[i]['senti_ids'] for i in range(len(batch))]
    new_batch['doc_lens'] = [batch[i]['doc_lens'] for i in range(len(batch))]
    return new_batch


