import torch
import numpy
import time
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ECEmodel(nn.Module):  # 两种输出方式 两种loss
    def __init__(self):
        super(ECEmodel,self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.pooler = Pooler(1536)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(1536)
        self.multilabel_output = nn.Linear(1536,1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, \
                 clauses_positions=None, doc_lens=None):
        sep_position = [[index for index in range(len(input_id)) if input_id[index] == 102] for input_id in input_ids ]
        bert_output = self.bert(input_ids=input_ids.to(device),\
                                attention_mask=attention_mask.to(device),\
                                token_type_ids=token_type_ids.to(device))

        sequence_output = bert_output[0]
        context_h = bert_output[1]
        clause_h = self.clause_output(sequence_output, context_h, clauses_positions,doc_lens,sep_position)  # 尝试对每一个clause的输出取平均作为结果

        pooled_clause_h = self.pooler(clause_h)
        # print(f"clause_pooled:{pooled_clause_h.shape}")
        # pooled_clause_h = self.layer_norm(pooled_clause_h)
        # pooled_clause_h = self.dropout(pooled_clause_h)
        logits = self.multilabel_output(pooled_clause_h)
        logits = logits.squeeze(-1).contiguous()

        return (logits, sequence_output,context_h) + bert_output[2:]


    def BCEloss(self, pred_logit, causes, doc_lens):

        mask = [torch.ones(doc_len,dtype=torch.bool) for doc_len in doc_lens]
        mask = pad_sequence(mask,batch_first=True,padding_value=0).to(device)
        # print(f"loss shape {pred_logit.shape},{causes.shape},{mask.shape}")
        if pred_logit.size()[1] != causes.size()[1]:
            print("Long document, but error!")
            print(causes)
            causes = causes[:, :pred_logit.size()[1]]
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        pred_logit = pred_logit.masked_select(mask)

        causes = causes.masked_select(mask)
        loss = criterion(pred_logit,causes.float())
        return loss

    def clause_output(self,sequence_output, context_h, clauses_positions, doc_lens,sep_position):
        """构造每个clause的表示"""
        if type(clauses_positions) != type([]):
            clauses_positions = clauses_positions.cpu().numpy().tolist()
        sequence_outputs_batch = torch.split(sequence_output, 1,dim=0)
        c_h = []
        for s_o, clause_position, doc_len,sp in zip(sequence_outputs_batch,clauses_positions,doc_lens,sep_position):

            split_size = [1] + [clause_position[i+1] - clause_position[i] for i in range(doc_len)]
            split_size[-1] = split_size[-1] + 1  # 为最后一个clause加上sep标签 先这样处理


            split_size.append(s_o.size()[1] - 1 - (clause_position[doc_len]))
            cutted_hiddens = torch.split(s_o,split_size,dim=1)

            clause_hiddens = torch.cat(\
                                [torch.cat(\
                                    (cutted_hiddens[i].mean(dim=1, keepdim=True),cutted_hiddens[0]),dim=2) \
                                        for i in range(1,doc_len + 1)], dim=1)
            # 这里的 context output没有用pool过的 后期可以改变
            # print(clause_hiddens.shape)  #  torch.Size([1, 10, 1536])
            c_h.append(clause_hiddens.squeeze())

        pad_sequence(c_h,batch_first=True)
        return pad_sequence(c_h,batch_first=True)



