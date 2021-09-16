import os
import time
import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from data_processing import *
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_emotion"

EPOCH = 10
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def lexicon_based_extraction(preds,docs_ids,docs_lens,truths):
    emotional_clauses = read_b('sentimental_clauses.pkl')
    # print(len(emotional_clauses))
    # print(len(docs_lens))
    # print(docs_ids[:10])
    # print(docs_lens[:10])
    # print(preds[:100])
    # print(truths[:100])
    current_index = 0
    revised_preds = []
    for doc_id,doc_len in zip(docs_ids, docs_lens):
        temp = []
        current_pred = [(index+1) for index in range(0,doc_len) if preds[current_index+index] == 1]
        current_truth = [(index + 1) for index in range(0,doc_len) if truths[current_index+index] == 1]
        for item in current_pred:
            if item in emotional_clauses[doc_id]:
                temp.append(item)
        if current_pred == []:
            temp = emotional_clauses[doc_id]
            # print("Not enough.")
        if len(current_pred) > len(temp):
            pass
            # print("Too many.")
        # print(f"current: {current_pred}, lexicon: {emotional_clauses[doc_id]}, truth: {current_truth}, revised = {temp}")
        # print([1 if j in temp else 0 for j in range(doc_len)])
        # print(doc_len)
        # if int(doc_id) > 100:
        #     exit(0)

        revised_preds.extend([1 if j in temp else 0 for j in range(1, doc_len+1)])
        current_index += doc_len
    return revised_preds


def predict(fold_id):
    data_loader = build_dataloader(fold_id, 'test', 'sentiment')
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
    model.to(device)
    model.load_state_dict(torch.load(f"checkpoints/fold{fold_id}.pth"))
    preds, truths = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pred = model(input_ids, attention_mask=attention_mask)[0]
            pred = torch.argmax(pred,dim=1).detach().cpu().numpy().tolist()
            truth=batch['labels'].squeeze().cpu().numpy().tolist()
            if type(truth) == type(0):
                truth = [truth]
            # print(truth)
            preds.extend(pred)
            truths.extend(truth)


    docs_ids,docs_lens = get_doc_info(fold_id)
    revised_preds = lexicon_based_extraction(preds,docs_ids,docs_lens,truths)
    print(len(preds),len(revised_preds))

    out = precision_recall_fscore_support(truths, preds, average=None, labels=[0, 1])
    p = out[0][1]
    r = out[1][1]
    f1 = out[2][1]
    metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}."
    print(f"fold{fold_id} : {metric}")

    out = precision_recall_fscore_support(truths, preds, average=None, labels=[0, 1])
    p = out[0][1]
    r = out[1][1]
    f1 = out[2][1]
    metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}."
    print(f"fold{fold_id} revised: {metric}")
    with open(f"fold{fold_id}predicted_result_revised.txt",'w') as f:
        out_str = '\n'.join([str(item) for item in revised_preds])
        f.write(out_str+'\n')
    return p,r,f1

if __name__ == '__main__':
    n_folds = 10
    metrics = []
    for fold_id in range(1, n_folds+1):
        metrics.append(predict(fold_id))
    print(np.array(metrics).mean(axis=0))