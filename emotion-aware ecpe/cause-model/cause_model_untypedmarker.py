import torch
import numpy
import time
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from data_processing_untypedmarker import tokenizer
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
        self.bert.resize_token_embeddings(len(tokenizer))
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
        # print(input_ids[0,:])
        # print(clauses_positions)
        # print(bert_output[0][0,:,:].shape)

        sequence_output = bert_output[0]
        context_h = bert_output[1]
        clause_h = self.clause_output(sequence_output, context_h, clauses_positions,doc_lens,sep_position)  # 尝试对每一个clause的输出取平均作为结果
        # clause_h = self.cls_output(sequence_output,clauses_positions,doc_lens) # 输入时插入cls符号 直接使用
        # print(f"clause:{clause_h.shape}")
        # exit(0)
        pooled_clause_h = self.pooler(clause_h)
        # print(f"clause_pooled:{pooled_clause_h.shape}")
        # pooled_clause_h = self.layer_norm(pooled_clause_h)
        # pooled_clause_h = self.dropout(pooled_clause_h)
        logits = self.multilabel_output(pooled_clause_h)
        logits = logits.squeeze(-1).contiguous()
        # print(f"logits:{logits.shape}")
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
        # print(f"sequence output:{sequence_output.shape}")
        # print(sequence_outputs_batch[0].shape)
        c_h = []
        for s_o, clause_position, doc_len,sp in zip(sequence_outputs_batch,clauses_positions,doc_lens,sep_position):
            if clause_position[-1] == -1:

                if clause_position.index(-1) != doc_len + 1:
                    print(clause_position)
                    print(doc_len)
                    print(' ')
            else:
                if len(clause_position) != doc_len + 1:
                    print(clause_position)
                    print(doc_len)
                    print(' ')
            split_size = [1] + [clause_position[i+1] - clause_position[i] for i in range(doc_len)]
            split_size[-1] = split_size[-1] + 1  # 为最后一个clause加上sep标签 先这样处理


            split_size.append(s_o.size()[1] - 1 - (clause_position[doc_len]))
            cutted_hiddens = torch.split(s_o,split_size,dim=1)
            # print(split_size)
            # print(len(cutted_hiddens))
            # print([cutted_hiddens[i].shape for i in range(doc_len + 1)])
            # print(context_h.shape,cutted_hiddens[0].shape)
            # print(context_h, cutted_hiddens[0])
            # senti_hiddens = s_o[:,sp[0]+1:sp[1],:].mean(dim=1, keepdim=True)
            clause_hiddens = torch.cat(\
                                [torch.cat(\
                                    (cutted_hiddens[i].mean(dim=1, keepdim=True),cutted_hiddens[0]),dim=2) \
                                        for i in range(1,doc_len + 1)], dim=1)
            # 这里的 context output没有用pool过的 后期可以改变
            # print(clause_hiddens.shape)  #  torch.Size([1, 10, 1536])
            c_h.append(clause_hiddens.squeeze())

        pad_sequence(c_h,batch_first=True)
        return pad_sequence(c_h,batch_first=True)

    def cls_output(self,sequence_output,clauses_positions,doc_lens):
        dummy = clauses_positions.unsqueeze(2).expand(clauses_positions.size(0), \
                            clauses_positions.size(1), sequence_output.size(2))
        doc_sents_h = sequence_output.gather(1, dummy)

        # print(sequence_output.shape)
        print(sequence_output[0,:50,:4])
        print(dummy[0,:,:4])
        print(doc_sents_h[0,:,:4])
        print(clauses_positions.shape)
        print(dummy.shape)
        print(doc_sents_h.shape)
        exit(0)
        return doc_sents_h


