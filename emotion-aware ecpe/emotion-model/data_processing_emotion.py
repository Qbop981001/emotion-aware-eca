import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

DATASET_TYPE = "original"
if DATASET_TYPE == "original":
    FILE = "../ECPE-original/split10/fold%s_%s.json"
elif DATASET_TYPE == "reconstructed":
    FILE = "../ECPE-reconstructed/split10/fold%s_%s.json"
else:
    print("Unknown Dataset!")
    exit(1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def split_doc(doc,doc_len,clauses):

    passages = []
    emotions = []
    doc_lens = []

    doc_len1 = doc_len // 2
    doc_len2 = doc_len - doc_len1

    passage1 = '，'.join(clauses[:doc_len1]) + '。'
    passage2 = '，'.join(clauses[doc_len1:]) + '。'
    sents = []
    for pair in doc["pairs"]:
        if pair[0] not in sents:
            sents.append(pair[0])
    emotion1 = [1 if i in sents else 0 for i in range(1, doc_len1 + 1)]
    emotion2 = [1 if i in sents else 0 for i in range(doc_len1 + 1,doc_len+1)]

    return [passage1,passage2], [emotion1,emotion2],[doc_len1,doc_len2]


def read_from_json(fold_id,data_type):
    passages = []
    emotions = []
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
            # 解决过长文档问题
            if len(passage) > 480:
                print("Long document. Deal with it by splitting.")
                p,e,dl = split_doc(doc,doc_len,clauses)
                passages.extend(p);
                emotions.extend(e)
                doc_lens.extend(dl)
                continue
            sents = []
            for pair in doc["pairs"]:
                if pair[0] not in sents:
                    sents.append(pair[0])

            emotion = [1 if i in sents else 0 for i in range(1, doc_len+1)]
            passages.append(passage)
            emotions.append(emotion)
            doc_lens.append(doc_len)

    return passages, emotions, doc_lens


def build_dataloader(fold_id, data_type, label_type, train_batch_size=8):
    passages,  emotions, doc_lens = read_from_json(fold_id, data_type)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    encodings = tokenizer(passages,  truncation=True, padding=True)

    add_encoding_positions(encodings, emotions, passages, doc_lens)
    dataset = Dataset(encodings)

    if data_type == 'train':
        train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=False, collate_fn=batch_preprocessing)
        return train_loader
    elif data_type =='test':
        valid_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=batch_preprocessing)
        return valid_loader
    else:
        print('Unknown data_type')
        exit(0)


def add_encoding_positions(encodings, emotions, passages,  doc_lens):

    clauses_positions = []
    for i,input_id in enumerate(encodings['input_ids']):
        clause_position = [1] + [(index + 1) for index, token in enumerate(input_id) if token == 8024 or token == 511]
        # print(clause_position,input_id)
        # print(len(clause_position),len(emotions[i]),len(passages[i].split(sep="，")),doc_lens[i])
        assert len(clause_position)-1==len(emotions[i])==len(passages[i].split(sep="，"))==doc_lens[i]  # 避免过长文档问题
        clauses_positions.append(clause_position)
    encodings.update({'emotions': emotions, 'clauses_positions':clauses_positions, 'doc_lens':doc_lens})


def batch_preprocessing(batch):
    new_batch = {}
    for key in batch[0].keys():
        if key != 'clauses_positions' and key != 'emotions':
            new_batch[key] = torch.stack(tuple([batch[i][key] for i in range(len(batch))]), dim=0)
    clause_position = [item['clauses_positions'] for item in batch]
    clause_position = pad_sequence(clause_position, batch_first=True, padding_value=-1)
    emotion = [item['emotions'] for item in batch]
    emotion = pad_sequence(emotion, batch_first=True, padding_value=-1)
    new_batch['clauses_positions'] = clause_position
    new_batch['emotions'] = emotion
    return new_batch
