# M3PL: Identifying and Exploiting View Bias of Prompt Learning [TMLR 2024]

Official implementation of the paper "[M3PL: Identifying and Exploiting View Bias of Prompt Learning](https://openreview.net/forum?id=2rnTIBm19V)"

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md). 

## Data preparation
Please follow the instructions in [DATASETS.md](docs/DATASETS.md) to prepare all datasets.


## Training and Evaluation
Please refer to the [RUN.md](docs/RUN.md) for detailed instructions on training and evaluating.

Our parameter configuration file is in ./configs/trainers/M3PL/ directory and our run script files are in the . /scripts/m3pl/ directory.

For cross-dataset generalization and DG please use xd_train.sh and xd_test.sh.

For Base-to-New generalization please use base2new_train.sh and base2new_test.sh.

## Citation
```bibtex
@article{
zhao2024mpl,
title={M\${\textasciicircum}3\${PL}: Identifying and Exploiting View Bias of Prompt Learning},
author={Chujie Zhao and Tianren Zhang and Guanyu Chen and Yizhou Jiang and Feng Chen},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=2rnTIBm19V},
note={}
}
```


## Acknowledgements

Our code is based on [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) and [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.

