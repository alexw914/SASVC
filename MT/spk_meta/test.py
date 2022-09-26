# -*- coding: UTF-8 -*-
import os
import pickle as pk
import json


with open("spk_meta_eval.pk", "rb") as f:
     eval_set=pk.load(f)

with open("spk_meta_dev.pk", "rb") as f:
     dev_set=pk.load(f)

with open("spk_meta_trn.pk", "rb") as f:
     train_set=pk.load(f)


# print(len(eval_set))
# print(len(dev_set))

# with open("spk_meta/spk_meta_eval.pk", "rb") as f:
#      spk_eval=pk.load(f)
# cnt=0
# for key in spk_eval.keys():
#      if spk_eval[key]['spoof']==[]:
#           cnt=cnt+1
#           print(key)
# print(cnt)

with open("spk_meta_eval.json", 'w') as f:
     json.dump(eval_set, f, indent=2)
with open("spk_meta_dev.json", 'w') as f:
     json.dump(dev_set, f, indent=2)
with open("spk_meta_train.json", 'w') as f:
     json.dump(train_set, f, indent=2)