import os
import time
import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from data_processing_emotion import *
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_emotion"
out_dir = "predictions"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f"{out_dir} established")

EPOCH = 10
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def lexicon_based_extraction(preds,docs_ids,docs_lens,truths):
    emotional_clauses = read_b('data/sentimental_clauses.pkl')

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

        revised_preds.extend([1 if j in temp else 0 for j in range(1, doc_len+1)])
        current_index += doc_len
    return revised_preds


def predict(fold_id):
    data_loader = build_dataloader(fold_id, 'test', 'sentiment')
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
    model.to(device)
    model.load_state_dict(torch.load(f"checkpoint_emotion/fold{fold_id}_best.pth"))
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

    print(f"total predicted causes: {len(preds)}")
    docs_ids,docs_lens = get_doc_info(fold_id)
    revised_preds = lexicon_based_extraction(preds,docs_ids,docs_lens,truths)
    # print(truths)
    # print(preds)
    # print(revised_preds)

    out = precision_recall_fscore_support(truths, preds, average=None, labels=[0, 1])
    p = out[0][1]
    r = out[1][1]
    f1 = out[2][1]
    metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}."
    print(f"fold{fold_id} : {metric}")

    with open(os.path.join(out_dir,f"fold{fold_id}predicted_result.txt"), 'w') as f:
        out_str = '\n'.join([str(item) for item in preds])
        f.write(out_str+'\n')

    revised_out = precision_recall_fscore_support(truths, revised_preds, average=None, labels=[0, 1])
    revised_p = revised_out[0][1]
    revised_r = revised_out[1][1]
    revised_f1 = revised_out[2][1]
    revised_metric = f"p={revised_p:.5f}, r={revised_r:.5f}, f1={revised_f1:.5f}."
    print(f"revised metric: fold{fold_id} : {revised_metric}")

    with open(os.path.join(out_dir,f"fold{fold_id}predicted_result_revised.txt"), 'w') as f:
        out_str = '\n'.join([str(item) for item in revised_preds])
        f.write(out_str+'\n')


if __name__ == '__main__':
    n_folds = 10
    # metrics = []
    for fold_id in range(1, n_folds+1):
        predict(fold_id)
