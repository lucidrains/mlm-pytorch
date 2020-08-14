## MLM (Masked Language Modeling) Pytorch

This repository allows you to quickly setup unsupervised training for your transformer off a corpus of sequence data.

## Install

```bash
$ pip install mlm-pytorch
```

## Usage

First `pip install reformer-pytorch`, then run the following example to see what one iteration of the unsupervised training is like

```python
import torch
from torch import nn
from torch.optim import Adam
from mlm_pytorch.mlm_pytorch import MLM

# instantiate the language model

from reformer_pytorch import ReformerLM

transformer = ReformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 1,
    max_seq_len = 1024
)

# plugin the language model into the MLM trainer

trainer = MLM(
    transformer,
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15,           # masking probability for masked language modeling
    replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
    mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
).cuda()

# optimizer

opt = Adam(trainer.parameters(), lr=3e-4)

# one training step (do this for many steps in a for loop, getting new `data` each time)

data = torch.randint(0, 20000, (10, 1024)).cuda()

loss = trainer(data)
loss.backward()
opt.step()
opt.zero_grad()

# after much training, the model should have improved for downstream tasks

torch.save(transformer, f'./pretrained-model.pt')
```

Do the above for many steps, and your model should improve.

## Citation

```bibtex
@article{Devlin_2019, url={http://dx.doi.org/10.18653/v1/N19-1423},
   DOI={10.18653/v1/n19-1423},
   journal={Proceedings of the 2019 Conference of the North},
   publisher={Association for Computational Linguistics},
   author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
   year={2019}
}
```
