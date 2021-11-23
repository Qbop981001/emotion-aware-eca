import os
import time
import torch
import numpy as np
from cause_model_typedmarker import *
from data_processing_typedmarker import build_dataloader
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.functional import sigmoid

ckpt_dir = 'ckpt_cause_typedmarker'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
    print(f"{ckpt_dir} established for model checkpoints.")

EPOCH = 2
MULTI_THRESHOLD = 0.5
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True
with open("multi_result.txt", 'w') as f:
    f.write("Here are the results!\n")


def metrics(preds, truths, doc_lens):
    pred_length, truth_length, catched = 0, 0, 0
    for pred, truth, doc_len in zip(preds, truths, doc_lens):
        pred = pred[:doc_len]
        truth = truth[:doc_len]
        assert (len(pred) == len(truth))
        for i in range(doc_len):
            if pred[i] == truth[i] == 1:
                catched += 1
        pred_length += pred.count(1)
        truth_length += truth.count(1)
    # print(catched, pred_length, truth_length)

    p = catched / (pred_length + 1e-8)
    r = catched / (truth_length + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        preds = []
        all_causes = []
        doc_lens = []
        ls = []
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            causes = batch['causes'].tolist()
            doc_len = batch['doc_lens'].to(device)
            clauses_positions = batch['clauses_positions'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                           clauses_positions=clauses_positions, doc_lens=doc_len)
            logits = output[0]
            # print((logits.sigmoid() > 0.5).to(torch.uint8).shape)
            pred = (logits.sigmoid() > MULTI_THRESHOLD).to(torch.uint8).tolist()
            ls.append(logits.cpu().tolist())
            all_causes.extend(causes)
            preds.extend(pred)
            doc_lens.extend(doc_len)

    p, r, f1 = metrics(preds, all_causes, doc_lens)

    return p, r, f1


def main(fold_id, train_loader, valid_loader):
    model = ECEmodel().to(device)
    # print(model)

    optim = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = EPOCH * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )

    upper_metric = []
    best = 0.0
    a = 0
    model.train()
    for epoch in range(EPOCH):
        epoch_start = time.time()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            causes = batch['causes'].to(device)
            doc_lens = batch['doc_lens'].to(device)
            clauses_positions = batch['clauses_positions'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                           clauses_positions=clauses_positions, doc_lens=doc_lens)
            loss = model.BCEloss(output[0], causes=causes, doc_lens=doc_lens)
            # if a % 100 == 0:
            #     print(f"loss = {loss}.")
            loss.backward()
            optim.step()
            lr_scheduler.step()
            a += 1

        p, r, f1 = evaluate(model, valid_loader, device)

        print(f"fold {fold_id} Epoch{epoch}: ece upper bound: {p}, {r}, {f1}. Cost {time.time() - epoch_start}s.")

        with open("multi_result.txt", 'a') as f:
            f.write(
                f"Epoch{epoch}: {p}, {r}, {f1}. cost {time.time() - epoch_start}s.\n")

        if f1 > best:
            best = f1
            # print("Saving best upper model....")
            # torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fold{fold_id}_best.pth"))

        upper_metric.append((p,r,f1))

        model.train()

    upper_metric = np.array(upper_metric)
    best_upper_epoch = upper_metric.argmax(axis=0)[-1]
    best_upper = upper_metric[best_upper_epoch,:]

    best_str = f"Best upper: {best_upper} in epoch {best_upper_epoch}"
    print(best_str)
    with open("multi_result.txt", 'a') as f:
        f.write(best_str+"\n")

    return best_upper

if __name__ == '__main__':
    n_folds = 1
    fold_metrics = []
    fold_reals = []
    for fold_id in range(1, n_folds+1):
        start = time.time()
        print('===== fold {} ====='.format(fold_id))
        train_loader = build_dataloader(fold_id, 'train', 'sentiment', train_batch_size=8)
        valid_loader = build_dataloader(fold_id, 'test', 'sentiment')

        metric_upper = main(fold_id, train_loader, valid_loader)
        fold_metrics.append(metric_upper)
        fold_time = time.time() - start
        print('Cost {}s.'.format(fold_time))

    print('===== Average =====')
    average_metric = np.array(fold_metrics).mean(axis=0)
    print(f"average_metric: \nupper bound:{average_metric}\n")

