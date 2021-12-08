import os
import time
import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from data_processing_emotion import *
from emotion_model import *
from train_emotion import ckpt_dir
import os

MULTI_THRESHOLD = 0.5

out_dir = "../predictions_split20"


if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    print(f"{out_dir} established for predicted emotion clauses in each fold.")


def eval_metric(preds, truths, doc_lens):

    pred_length, truth_length, catched = 0, 0, 0
    for pred, truth, doc_len in zip(preds,truths,doc_lens):
        pred = pred[:doc_len]
        truth = truth[:doc_len]
        assert(len(pred) == len(truth))
        for i in range(doc_len):
            if pred[i] == truth[i] == 1:
                catched += 1
        pred_length += pred.count(1)
        truth_length += truth.count(1)
    print(catched,pred_length,truth_length)

    p = catched / (pred_length + 1e-8)
    r = catched / (truth_length + 1e-8)
    f1 = 2*p*r / (p+r + 1e-8)
    return p,r,f1

def predict(fold_id):
    data_loader = build_dataloader(fold_id, 'test', 'sentiment')
    model = EmotionContextModel()
    model.to(device)

    print("Loading...")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold_id}_best.pth")))
    print("Loaded!")
    model.eval()
    with torch.no_grad():
        preds = []
        all_emotions = []
        doc_lens = []
        ls = []
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            emotions = batch['emotions'].tolist()
            doc_len = batch['doc_lens'].to(device)
            clauses_positions = batch['clauses_positions'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                           clauses_positions=clauses_positions, doc_lens=doc_len)
            logits = output[0]
            # print((logits.sigmoid() > 0.5).to(torch.uint8).shape)
            pred = (logits.sigmoid() > MULTI_THRESHOLD).to(torch.uint8).tolist()
            ls.append(logits.cpu().tolist())
            all_emotions.extend(emotions)
            preds.extend(pred)
            doc_lens.extend(doc_len)
    # print(f"logits: {len(ls)} ,{ls[:3]}")
    p,r,f1 = eval_metric(preds,all_emotions,doc_lens)
    print(p,r,f1)
    out = [unit for item, doc_len in zip(preds,doc_lens) for unit in item[:doc_len]]
    show = [item[:doc_len] for item, doc_len in zip(preds,doc_lens)]
    with open(os.path.join(out_dir,f"fold{fold_id}predicted_result.txt"), 'w') as f:
        out_str = '\n'.join([str(item) for item in out])
        f.write(out_str+'\n')

    return p,r,f1

if __name__ == '__main__':
    n_folds = 20
    metrics = []
    for fold_id in range(1, n_folds+1):
        print(f"====================fold{fold_id}===================")
        metric = predict(fold_id)
        metrics.append(metric)

    print("++++++Average++++++++")
    print(f"metrics: {np.array(metrics).mean(axis=0)}")