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
else:
    print("Unknown Dataset!")
    exit(1)


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

new_tokens = ['<EMO_START>', '<EMO_END>']
labels = ['sadness', 'disgust', 'surprise', 'fear', 'anger', 'happiness']
for l in labels:
    new_tokens.append('<EMO_START=%s>'%l)
    new_tokens.append('<EMO_END=%s>'%l)
tokenizer.add_tokens(new_tokens)


class ECEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

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
                        passage[-1] = '<EMO_START=%s>' % clause['emotion_category'] + clause['clause'] + '<EMO_END=%s>' % clause['emotion_category']
                        sentiments.append(clause['clause'])

                cause = [1 if i in cs else 0 for i in range(1, len(passage)+1)] 
                passage = '，'.join(passage)+'。'
                passages.append(passage)
                causes.append(cause)
                doc_lens.append(doc_len)

    return passages, sentiments, causes, doc_lens


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


def add_encoding_positions(encodings, causes, passages, doc_lens):

    clauses_positions = []
    for input_id in encodings['input_ids']:
        clause_position = [1] + [(index + 1) for index, token in enumerate(input_id) if token == 8024 or token == 511]
        # print(clause_position)
        clauses_positions.append(clause_position)
    encodings.update({'causes': causes, 'clauses_positions':clauses_positions, 'doc_lens':doc_lens})


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
