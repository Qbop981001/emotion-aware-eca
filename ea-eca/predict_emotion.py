import os
import time
import torch
import pickle
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.metrics import precision_recall_fscore_support
# from data_processing import *
# from transformers import BertForSequenceClassification, AdamW
# from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_emotion"

EPOCH = 10
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b




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

    print(len(preds))
    out = precision_recall_fscore_support(truths, preds, average=None, labels=[0, 1])
    p = out[0][1]
    r = out[1][1]
    f1 = out[2][1]
    metric = f"p={p:.5f}, r={r:.5f}, f1={f1:.5f}."
    print(f"fold{fold_id} : {metric}")
    with open("predicted_result.txt",'a') as f:
        out_str = '\n'.join([str(item) for item in preds])
        f.write(out_str+'\n')



if __name__ == '__main__':
    n_folds = 10
    # metrics = []
    for fold_id in range(1, n_folds+1):
        predict(fold_id)
