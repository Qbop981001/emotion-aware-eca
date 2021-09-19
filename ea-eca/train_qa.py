import os, warnings
warnings.filterwarnings("ignore")
import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from transformers import BertForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup
from data_processing_qa import build_dataloader

EPOCH = 2
result_dir = 'qa_evaluation'
ckpt_dir = 'checkpoint_qa'
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
    # print(ps[:5], rs[:5])
    return torch.tensor(ps).sum() / len(ps), torch.tensor(rs).sum() / len(rs)


def find_cause(ratios,absolute_lengths):
    # print(ratios)
    # print(absolute_lengths)
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


def real_metric(model, real_valid_loader, num_of_pairs_all, device):
    preds_start, preds_end, clauses_positions, docs_pairs, pred_sentiments_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in real_valid_loader:
            # print(f"batch in real:\n{batch}")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch['input_ids'] = list(batch['input_ids'])
            doc_pairs = batch['doc_pairs']
            pred_sentiments_id = batch['senti_ids']
            clause_position = batch['clauses_positions']
            pred = model(input_ids, attention_mask=attention_mask)
            pred_start = torch.argmax(pred[0], dim=1).detach().cpu().numpy().tolist()
            pred_end = torch.argmax(pred[1], dim=1).detach().cpu().numpy().tolist()
            # exit(0)
            # if type(clause_position) != type([]):
            #     # print(type(truth_start))
            #     clause_position = [clause_position]
            preds_start.extend(pred_start)
            preds_end.extend(pred_end)
            clauses_positions.extend(clause_position)
            docs_pairs.extend(doc_pairs)
            pred_sentiments_ids.extend(pred_sentiments_id)

        selected_causes = select(preds_start, preds_end, clauses_positions)
        # 这个是从0开始的index,为每一个预测出来的emotion clause 所预测的selected causes
        # selected_causes = [i+1 for i in selected_causes]
        selected_causes = [[i + 1 for i in selected_cause] for selected_cause in selected_causes] # 多原因时使用
        selected_pairs = list(zip(pred_sentiments_ids,selected_causes))
        # print(selected_pairs)
        catched = 0
        for selected_pair,doc_pairs in zip(selected_pairs, docs_pairs):
            # print(selected_pair,doc_pairs)
            for c in selected_pair[1]:

                if [selected_pair[0],  c] in doc_pairs:
                    catched += 1
        print(catched, len(selected_pairs), num_of_pairs_all,len(docs_pairs))
        # exit(0)
        p = catched / len(selected_pairs)
        r = catched / num_of_pairs_all
        f1 = 2 * p * r / (p + r + 1e-8)

    return p,r,f1


def evaluate(model, data_loader, real_valid_loader, device):
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
           real_metric(model, real_valid_loader, len(truths_start), device)


def main(fold_id,train_loader,valid_loader,real_valid_loader):
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
    reals = []
    out_strs = []
    best_real, best_ece, best_qa = 0.0, 0.0,0.0
    # metric= evaluate(model, valid_loader, real_valid_loader, device)
    # print(metric)
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
        metric= evaluate(model, valid_loader, real_valid_loader,device)
        epoch_time = (time.time() - epoch_start)
        p = metric[0][0]; r = metric[0][1] ; f1 = 2*p*r/(p+r+1e-8)
        ece = metric[1]
        real_metric = metric[2]
        out_str = f" qa_metric: {p},{r},{f1}  ece = {ece}  real_ecpe: {real_metric} epoch_time: {epoch_time}s"
        print(out_str)
        out_strs.append(out_str)
        qa_metrics.append(metric[0])
        reals.append(real_metric)
        eces.append(ece)
        if real_metric[2] >= best_real or (real_metric[2] == best_real and ece[2] > best_ece):
            print("saving model")
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"fold{fold_id}_best.pth"))
            best_real = real_metric[2]
            real_epoch = epoch
        if ece[2] > best_ece:
            best_ece = ece[2]
            ece_epoch = epoch
        if f1 > best_qa:
            best_qa = f1
            qa_epoch = epoch
        model.train()

    with open(os.path.join(result_dir,'fold'+str(fold_id)+'qa_based_model_results.txt'),'w') as f:
        for i in out_strs:
            f.write(i+'\n')
    return qa_metrics[qa_epoch], qa_epoch, eces[ece_epoch], ece_epoch, reals[real_epoch], real_epoch


def predict(fold_id,valid_loader,real_valid_loader):
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese")
    model.to(device)
    model.load_state_dict(torch.load(f"checkpoint_qa/fold{fold_id}_best.pth"))
    metric = evaluate(model, valid_loader, real_valid_loader, device)
    p = metric[0][0]
    r = metric[0][1]
    f1 = 2 * p * r / (p + r + 1e-8)
    acc = metric[1]
    real_metric = metric[2]
    out_str = f"qa_metric: {p},{r},{f1}  ece = {acc}   ecpe_real: {real_metric} "
    print(out_str)
    return p,r,f1,acc[0],acc[1],acc[2],real_metric[0],real_metric[1],real_metric[2]


if __name__ == '__main__':
    n_folds = 10
    fold_f1s = []
    fold_accs = []
    fold_reals = []
    metrics = []
    for fold_id in range(1, 2):
        start = time.time()
        print('===== fold {} ====='.format(fold_id))
        train_loader = build_dataloader(fold_id, 'train', 'sentiment','truth')
        valid_loader = build_dataloader(fold_id, 'test', 'sentiment','truth')
        real_valid_loader = build_dataloader(fold_id, 'test', 'sentiment','predict')
        metric_ece = main(fold_id,train_loader,valid_loader,real_valid_loader)
        best_str = f"best qa: {metric_ece[0]}, in epoch {metric_ece[1]}; best ece: {metric_ece[2]}, in epoch {metric_ece[3]}, real_ecpe_f1_best: {metric_ece[4]} n epoch {metric_ece[5]}"
        print(best_str)
        with open(os.path.join(result_dir,'fold'+str(fold_id)+'qa_based_model_results.txt'), 'a') as f:
            f.write(best_str+'\n')
        fold_time = time.time() - start

        print(f'Cost {fold_time}s.')
        fold_f1s.append(metric_ece[0])
        fold_accs.append(metric_ece[2])
        fold_reals.append(metric_ece[4])
        # predict
        # metric_ece = predict(fold_id,valid_loader,real_valid_loader)
        # metrics.append(metric_ece)

    print('===== Average =====')
    average_f1 = np.array(fold_f1s).mean(axis=0)
    average_acc = np.array(fold_accs).mean(axis=0)
    average_real = np.array(fold_reals).mean(axis=0)
    print(f"average_f1: {average_f1}, average_acc: {average_acc}, average_real: {average_real}")
    with open(os.path.join(result_dir,'qa_result'), 'w') as f:
        f.write(f"average_qa_f1: {average_f1}, average_ece: {average_acc}, average_real_ecpe: {average_real}")

    # predict
    # average = np.array(metrics).mean(axis=0)
    # print(average)