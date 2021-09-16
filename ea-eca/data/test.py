# truths_start = [5,5,5,5,5,5,5,5,5]
# truths_end =   [10,10,10,10,10,10,10,10,10]
# preds_start = [0,0,0,5,5,5,11,11,11]
# preds_end =   [4,5,11,4,5,11,4,5,11]
# ps, rs = [], []
# for i in range(len(preds_start)):
#     if preds_end[i] - preds_start[i] < 0 or preds_start[i] > truths_end[i]:
#         p = 0
#         r = 0
#     else:
#         if preds_start[i] < truths_start[i]:
#             catched_length = max(0, min(preds_end[i], truths_end[i]) - truths_start[i] + 1)
#         else:
#             catched_length = min(preds_end[i], truths_end[i]) - preds_start[i] + 1
#         p = catched_length / (preds_end[i] - preds_start[i] + 1)
#         r = catched_length / (truths_end[i] - truths_start[i] + 1)
#     ps.append(p)
#     rs.append(r)
# print(ps[:10], rs[:10])
# result = ([0,1,2,3],[5,6,7,8])
# f = open('fold1_causes.txt', 'w')
# for start, end in zip(result[0], result[1]):
#     out = str(start) + '\t' + str(end) + '\n'
#     f.write(out)
# f.close()
import json
FILE = "fold%s_%s.json"
for data_type in ['train','test']:
    for i in range(1,11):
        print(i, data_type)
        filename = FILE %(4, 'train')
        with open(filename, encoding='utf-8') as f:
            js = json.load(f)




seed=42 warmup = 0.2
epoch 0 starts.
start evaluation
p = 0.6278839707374573, r = 0.6558372974395752, f1 = 0.6415563225746155  acc = 0.6233766078948975   epoch_time: 99.22755217552185s
epoch 1 starts.
start evaluation
p = 0.6486918330192566, r = 0.6748191714286804, f1 = 0.6614975929260254  acc = 0.6666666865348816   epoch_time: 102.26229596138s
epoch 2 starts.
start evaluation
p = 0.6705082058906555, r = 0.7005258798599243, f1 = 0.6851884126663208  acc = 0.6580086350440979   epoch_time: 103.02474594116211s
epoch 3 starts.
start evaluation
p = 0.7006794810295105, r = 0.7187885046005249, f1 = 0.7096185088157654  acc = 0.7359307408332825   epoch_time: 103.36959028244019s
epoch 4 starts.
start evaluation
p = 0.7028467655181885, r = 0.7302982211112976, f1 = 0.7163095474243164  acc = 0.709956705570221   epoch_time: 103.71773433685303s
epoch 5 starts.
start evaluation
p = 0.6786647439002991, r = 0.7315985560417175, f1 = 0.7041382193565369  acc = 0.7229437232017517   epoch_time: 103.96447467803955s
epoch 6 starts.
start evaluation
p = 0.6866806745529175, r = 0.7236077785491943, f1 = 0.7046607732772827  acc = 0.7056276798248291   epoch_time: 103.83560228347778s
epoch 7 starts.
start evaluation
p = 0.7025635242462158, r = 0.7425258159637451, f1 = 0.7219920754432678  acc = 0.7229437232017517   epoch_time: 103.80617380142212s
epoch 8 starts.
start evaluation
p = 0.6952624320983887, r = 0.719721257686615, f1 = 0.7072804570198059  acc = 0.709956705570221   epoch_time: 104.14520907402039s
epoch 9 starts.
start evaluation
p = 0.6929351091384888, r = 0.7214978933334351, f1 = 0.7069280743598938  acc = 0.7142857313156128   epoch_time: 103.85727071762085s
best f1: 0.7219920754432678, in epoch 7; best acc: 0.7359307408332825, in epoch 3


epoch 0 starts.
start evaluation
p = 0.6434680819511414, r = 0.6978124380111694, f1 = 0.6695393323898315  acc = 0.6233766078948975   epoch_time: 103.16735243797302s
epoch 1 starts.
start evaluation
p = 0.6884154677391052, r = 0.7151422500610352, f1 = 0.7015243768692017  acc = 0.7056276798248291   epoch_time: 103.79374384880066s
epoch 2 starts.
start evaluation
p = 0.669585108757019, r = 0.6935026049613953, f1 = 0.6813340783119202  acc = 0.6796537041664124   epoch_time: 103.63829493522644s
epoch 3 starts.
start evaluation
p = 0.696242094039917, r = 0.7171992659568787, f1 = 0.706565260887146  acc = 0.709956705570221   epoch_time: 103.72712588310242s
epoch 4 starts.
start evaluation
p = 0.6865456104278564, r = 0.698854923248291, f1 = 0.6926456093788147  acc = 0.701298713684082   epoch_time: 104.00303649902344s
epoch 5 starts.
start evaluation
p = 0.7033153772354126, r = 0.7576892375946045, f1 = 0.7294905185699463  acc = 0.7186146974563599   epoch_time: 103.66880798339844s
epoch 6 starts.
start evaluation
p = 0.7115274667739868, r = 0.7419435977935791, f1 = 0.7264173030853271  acc = 0.7272727489471436   epoch_time: 104.00354528427124s
epoch 7 starts.
start evaluation
p = 0.7033319473266602, r = 0.7373857498168945, f1 = 0.7199563384056091  acc = 0.7186146974563599   epoch_time: 103.62661027908325s
epoch 8 starts.
start evaluation
p = 0.7130979895591736, r = 0.7319040894508362, f1 = 0.7223786115646362  acc = 0.7229437232017517   epoch_time: 103.76812362670898s
epoch 9 starts.
start evaluation
p = 0.7128318548202515, r = 0.7380154728889465, f1 = 0.72520512342453  acc = 0.7229437232017517   epoch_time: 103.86197090148926s
best f1: 0.7294905185699463, in epoch 5; best acc: 0.7272727489471436, in epoch 6
