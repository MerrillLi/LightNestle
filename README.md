## LightNestle

### Introduction
This is an official PyTorch implementation of paper entitled "LightNestle: Quick and Accurate Neural Sequential Tensor Completion via Meta Learning". (IEEE INFOCOM 2023, CCF-A Tier Conference)

### Abstract
Network operation and maintenance rely heavily on network traffic monitoring. Due to the measurement overhead reduction, lack of measurement infrastructure, and unexpected transmission error, network traffic monitoring systems suffer from incomplete observed data and high data sparsity problems. Recent studies model missing data recovery as a tensor completion task and show good performance. Although promising, the current tensor completion models adopted in network traffic data recovery lack of an effective and efficient retraining scheme to adapt to newly arrived data while retaining historical information.

To solve the problem, we propose LightNestle, a novel sequential tensor completion scheme based on meta-learning, which designs (1) an expressive neural network to transfer spatial knowledge from previous embeddings to current embeddings; (2) an attention-based module to transfer temporal patterns into current embeddings in linear complexity; and (3) an meta-learning-based algorithms to iteratively recover missing data and update transfer modules to catch up with learned knowledge. We conduct extensive experiments on two real-world network traffic datasets to assess our performance. The result demonstrates that our proposed methods achieve both fast retraining and high recovery accuracy.

### Citation
```
@Article{LightNestle,
  title   = {LightNestle: Quick and Accurate Neural Sequential Tensor Completion via Meta Learning},
  author = {Li, Yuhui and Liang, Wei and Xie, Kun and Zhang, Dafang and Xie, Songyou and Li, Kuan-Ching},
  booktitle = {{{IEEE INFOCOM}} 2023 - {{IEEE Conference}} on {{Computer Communications}}},
  year    = {2023},
}
```
