import json
import time
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
FILE = "data/fold%s_%s.json"
emotion_mapping = {'null': 0, 'sadness':1, 'disgust':2, 'surprise':3, 'fear':4, 'anger': 5, 'happiness':6 }



class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_from_json(fold_id,data_type):
    filename = FILE %(fold_id, data_type)
    X, sentiments, emotions = [], [], []
    with open(filename, encoding='utf-8') as f:
        js = json.load(f)
        for doc in js:
            clauses = doc['clauses']
            for clause in clauses:
                if clause['emotion_category'] == 'null':
                    sentiments.append(0)
                else:
                    sentiments.append(1)
                if '&' in clause['emotion_category']:  # 只有在做情绪分类的时候才会用到这个
                    # print(doc)
                    X.append(clause['clause'])
                else:
                    X.append(clause['clause'])
                    emotions.append(clause['emotion_category'])
    return X, sentiments, emotions


def build_dataloader(fold_id, data_type, label_type):
    X, sentiments, emotions = read_from_json(fold_id,data_type)
    if label_type == 'emotion':
        y = pd.Series(emotions).map(emotion_mapping).tolist()
    elif label_type == 'sentiment':
        y = sentiments
    else:
        print('Unknwon label_type.')
    print(f"Total clauses num: {len(y)},{len(X)}")
    # print(X[:5])

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    token_start = time.time()
    encodings = tokenizer(list(X), truncation=True, padding=True)
    tic = time.time() - token_start

    dataset = Dataset(encodings, y)
    if data_type == 'train':
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        return train_loader
    elif data_type == 'test':
        valid_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        return valid_loader
    else:
        print('Unknown data_type')
