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
