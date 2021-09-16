import os
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from data_processing import *
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
ckpt_dir = "checkpoint_emotion"
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
    print("Checkpoint directory established.")

EPOCH = 10
device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')


def evaluate(model, data_loader,device):
    model.eval()
    preds = []
    truths = []
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
    return accuracy_score(truths,preds), \
           precision_recall_fscore_support(truths,preds, average=None, labels=[0,1])


def main(fold_id, train_loader, valid_loader):
    TORCH_SEED = 42
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=7)
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = EPOCH * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.2*num_training_steps,
        num_training_steps=num_training_steps
    )

    model.train()
    best_f1 = 0.0
    best_epoch = -1
    for epoch in range(EPOCH):
        epoch_start = time.time()
        print(f"epoch {epoch} starts.")
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()

        acc, out = evaluate(model, valid_loader, device)
        p = out[0]; r = out[1]; f1 = out[2]
        metric = f"acc={acc:.5f}, p={p:.5f}, r={r:.5f}, f1={f1:.5f}."

        epoch_time = (time.time() - epoch_start)
        time_loss = "epoch time : %d min %d sec， loss = %f" % \
                    ((epoch_time % 3600) // 60, epoch_time % 60, loss)
        print(metric + time_loss)

        with open('sentiment_classification_results.txt','a') as f:
            f.write(f"Epoch {epoch}:")
            f.write(out+'\n')

        if f1 > best_f1:
            best_f1 = f1; best_acc = acc; best_p = p; best_r = r; best_epoch = epoch
            print("saving model")
           torch.save(model.state_dict(),os.path.join(ckpt_dir, f"fold{fold_id}_bestin_epoch{epoch}.pth"))
        model.train()

    return best_acc, best_p, best_r, best_f1


if __name__ == '__main__':
    n_folds = 10
    metrics = []
    for fold_id in range(1, n_folds+1):
        start = time.time()
        print('===== fold {} ====='.format(fold_id))
        train_loader = build_dataloader(fold_id, 'train', 'emotion')
        valid_loader = build_dataloader(fold_id, 'test', 'emotion')
        acc, p, r, f1 = main(fold_id,train_loader,valid_loader)
        metric_str = metric = f"Best in fold{fold_id}: acc={acc:.5f}, p={p:.5f}, r={r:.5f}, f1={f1:.5f}."
        with open('sentiment_classification_result.txt','a') as f:
            f.write(metric_str+'\n')
        fold_time = time.time() - start
        print(f"Cost {fold_time}s")
        metrics.append((acc, p, r, f1))

    avg_metric = np.array(metrics).mean(axis=0).tolist()
    print('===== Average =====')
    avg_metric_str = f"\noverall average: acc={avg_metric[0]:.5f}, p={avg_metric[1]:.5f}, r={avg_metric[2]:.5f}, f1={avg_metric[3]:.5f}"
    with open('sentiment_classification_result.txt', 'a') as f:
        f.write(avg_metric_str)
        
