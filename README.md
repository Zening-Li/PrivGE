# PrivGE
This repository contains code for CIKM 2024 paper titled [Privacy-Preserving Graph Embedding based on Local Differential Privacy](https://arxiv.org/abs/2310.11060).

# Getting Started
## Requirements
* pyg 2.2.0
* pytorch 1.12.0
* pybind11 2.10.3

## Dataset
Get datasets through the [link](https://drive.google.com/drive/folders/1qPYp530NSM_yqg9eLxTXrwiOYDfyWNsT?usp=sharing) and put them to the corresponding directories. For example, Cora dataset should be placed into datasets/cora/.
## Usage
```shell
cd precompute
make
```
### Node Classification
Train with the following command, optional arguments could be found in classification.py.
```shell
bash node_class.sh
```
### Link Prediction
Train with the following command, optional arguments could be found in link_pred.py.
```shell
bash link_prediction.sh
```

# Citation
Please cite our paper if you use the code in your work:
```
@inproceedings{li2023locally,
  author = {Li, Zening and Li, Rong-Hua and Liao, Meihao and Jin, Fusheng and Wang, Guoren},
  title = {Privacy-Preserving Graph Embedding based on Local Differential Privacy},
  year = {2024},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3627673.3679759},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  series = {CIKM '24}
}
```