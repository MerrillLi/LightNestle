#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 05/07/2022 22:52
# @Author : YuHui Li(MerylLynch)
# @File : param_counter.py
# @Comment : Created By Liyuhui,22:52
# @Completed : No
# @Tested : No


from baselines.CoSTCo import CoSTCo
from baselines.NTM import NeuralTensorModel
from baselines.NTF import NeuralTensorFactorization
from baselines.NTC import NeuralTensorCompletion
from module.LightNMMF import LightTC

import torch as t
from collections import namedtuple
# Abilene

Args = namedtuple('Argument', ['users', 'items', 'times', 'rank', 'channels', 'windows', 'dim', 'attention'])
args = Args(users=12, items=12, times=400, rank=16, channels=16, windows=400, dim=16, attention=True)
model = LightTC(args)
idx = t.tensor([0])
model.forward(idx, idx, idx)
count = 0

for x in model.parameters():
    count += x.numel()

print(count)
