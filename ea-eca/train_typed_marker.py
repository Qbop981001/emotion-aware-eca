import os, warnings
warnings.filterwarnings("ignore")
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup
from data_processing_typed_marker import build_dataloader
EPOCH = 10
result_dir = 'typed_evaluation'
ckpt_dir = 'checkpoint_typed_marker'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
    print(f"{ckpt_dir} established")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    print(f"{result_dir} established")
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def catched_metric(preds_start, preds_end, truths_start, truths_end):
    ps, rs = [], []
    for i in range(len(preds_start)):
        if preds_end[i] - preds_start[i] < 0 or preds_start[i] > truths_end[i]:
            p = 0
            r = 0
        else:
            if preds_start[i] < truths_start[i]:
                catched_length = max(0, min(preds_end[i],truths_end[i]) - truths_start[i] + 1)
            else:
                catched_length = min(preds_end[i],truths_end[i]) - preds_start[i] + 1
            p = catched_length / (preds_end[i] - preds_start[i] + 1)
            r = catched_length / (truths_end[i] - truths_start[i] + 1)
        ps.append(p)
        rs.append(r)
    return torch.tensor(ps).sum() / len(ps), torch.tensor(rs).sum() / len(rs)


def find_cause(ratios,absolute_lengths):
    if len([index for index in range(len(ratios)) if ratios[index] > 0.9]) > 1:
        result = [index for index in range(len(ratios)) if ratios[index] > 0.9]
    else:
        max_ratio_indexes = [index for index in range(len(ratios)) if ratios[index] == max(ratios) ]
        result = [absolute_lengths.index(max([absolute_lengths[index] for index in max_ratio_indexes]))]
    return result


def select(preds_start,preds_end,clauses_positions):
    selected_causes = []
    for start, end, clause_position in zip(preds_start,preds_end,clauses_positions):
        if start >= end: # QA预测错误
            selected_causes.append([0])
            continue
        ratios = []
        absolute_lengths = []
        doc_len = clause_position.index(-1) if clause_position[-1] == -1 else len(clause_position)
        doc_len = doc_len - 1
        for i in range(doc_len): # 计算各个clause的ratio与绝对长度 #新增: 由于我们为多出来的添加了padding，所以长度要重新调整
            absolute_length = max(0, min(end, clause_position[i + 1]) - max(start, clause_position[i]))
            ratio = absolute_length / (clause_position[i+1] - clause_position[i])
            ratios.append(ratio)
            absolute_lengths.append(absolute_length)
        selected_causes.append(find_cause(ratios,absolute_lengths))
    return selected_causes


def ece_metric(truths_start, truths_end, preds_start, preds_end,clauses_positions):
    true_causes = []
    # print(clauses_positions)
    for truth_start, truth_end, clause_position in zip(truths_start, truths_end, clauses_positions):
        # print(clause_position, truth_start)
        true_cause = clause_position.index(truth_start)
        if clause_position[true_cause+1] != truth_end:
            true_cause = list(range(true_cause, clause_position.index(truth_end)))
            true_causes.append(true_cause)
        else:
            true_causes.append([true_cause])
    selected_causes = select(preds_start,preds_end,clauses_positions)
    assert(len(true_causes) == len(selected_causes))
    # acc = torch.tensor([1 if a == b else 0 for a,b in zip(selected_causes, true_causes)]).sum() / float(len(true_causes))
    catched,pred,real =0,0,0
    for s, t in zip(selected_causes, true_causes):
        for ss in s:
            if ss in t:
                catched +=1
        pred+=len(s)
        real+=len(t)

    p = catched / pred
    r = catched / real
    f1 = 2 * p * r / (p + r + 1e-8)
    return (p,r,f1)

def evaluate(model, data_loader,  device):
    model.eval()
    preds_start, preds_end, truths_start, truths_end, clauses_positions = [], [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch['input_ids'] = list(batch['input_ids'])
            pred = model(input_ids, attention_mask=attention_mask)

            pred_start = torch.argmax(pred[0], dim=1).detach().cpu().numpy().tolist()
            pred_end = torch.argmax(pred[1], dim=1).detach().cpu().numpy().tolist()
            truth_start = batch['start_positions'].squeeze().cpu().numpy().tolist()
            truth_end = batch['end_positions'].squeeze().cpu().numpy().tolist()
            clause_position = batch['clauses_positions'].squeeze().cpu().numpy().tolist()
            if type(truth_start) != type([]):
                # print(type(truth_start))
                truth_start = [truth_start]; truth_end = [truth_end]; clause_position = [clause_position]
            preds_start.extend(pred_start)
            preds_end.extend(pred_end)
            truths_start.extend(truth_start)
            truths_end.extend(truth_end)
            clauses_positions.extend(clause_position)
    # print(preds_start[:5], truths_start[:5], preds_end[:5], truth_end[:5])

    return catched_metric(preds_start, preds_end, truths_start, truths_end), \
           ece_metric(truths_start, truths_end, preds_start, preds_end,clauses_positions), \


def main(fold_id,train_loader,valid_loader):
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    model.to(device)

    optim = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = EPOCH * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=0.2 * num_training_steps,
        num_training_steps=num_training_steps
    )
    model.train()
    qa_metrics, eces = [], []
    out_strs = []
    best_ece, best_qa = 0.0, 0.0

    for epoch in range(EPOCH):
        epoch_start = time.time()
        print(f"epoch {epoch} starts.")
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()

        # print("start evaluation")
        metric= evaluate(model, valid_loader,device)
        epoch_time = (time.time() - epoch_start)
        p = metric[0][0]; r = metric[0][1] ; f1 = 2*p*r/(p+r+1e-8)
        ece = metric[1]

        out_str = f"EPOCH{epoch} qa metric : p = {p}, r = {r}, f1 = {f1}  ece = {ece}    epoch_time: {epoch_time}s"
        print(out_str)
        out_strs.append(out_str)
        qa_metrics.append((p,r,f1))
        eces.append(ece)

        if ece[2] > best_ece:
            best_ece = ece[2]
            ece_epoch = epoch
            print("saving model")
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fold{fold_id}_best.pth"))
        if f1 > best_qa:
            best_qa = f1
            qa_epoch = epoch
        model.train()

    with open(os.path.join(result_dir,'fold'+str(fold_id)+'typed_result.txt'),'w') as f:
        for i in out_strs:
            f.write(i+'\n')
    return qa_metrics[qa_epoch], qa_epoch, eces[ece_epoch], ece_epoch


if __name__ == '__main__':
    n_folds = 10
    fold_f1s = []
    fold_accs = []

    metrics = []
    for fold_id in range(1, n_folds+1):
        start = time.time()
        print('===== fold {} ====='.format(fold_id))
        train_loader = build_dataloader(fold_id, 'train', 'emotion','truth')
        valid_loader = build_dataloader(fold_id, 'test', 'emotion','truth')
        metric_ece = main(fold_id,train_loader,valid_loader)
        best_str = f"fold {fold_id} best qa_metric: {metric_ece[0]}, in epoch {metric_ece[1]}; best ece: {metric_ece[2]}, in epoch {metric_ece[3]}"
        print(best_str)
        with open(os.path.join(result_dir,'fold'+str(fold_id)+'typed_result.txt'), 'a') as f:
            f.write(best_str+'\n')
        fold_time = time.time() - start

        print('Cost {}s.'.format(fold_time))
        fold_f1s.append(metric_ece[0])
        fold_accs.append(metric_ece[2])

    print('===== Average =====')
    average_f1 = np.array(fold_f1s).mean(axis=0)
    average_acc = np.array(fold_accs).mean(axis=0)

    print(f"average_qa_metric: {average_f1}, average_ece: {average_acc}")
    with open(os.path.join(result_dir,'fold'+str(fold_id)+'typed_result.txt'), 'a') as f:
        f.write(f"average_qa_metric: {average_f1}, average_ece: {average_acc}")
