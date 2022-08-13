# TGL: A General Framework for Temporal Graph Training on Billion-Scale Graphs

## Overview

This repo is the open-sourced code for our work *TGL: A General Framework for Temporal Graph Training on Billion-Scale Graphs*.

## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

Our temporal sampler is implemented using C++, please compile the sampler first with the following command
> python setup.py build_ext --inplace

## Dataset

The four datasets used in our paper are available to download from AWS S3 bucket using the `down.sh` script. The total download size is around 350GB.

To use your own dataset, you need to put the following files in the folder `\DATA\\<NameOfYourDataset>\`

1. `edges.csv`: The file that stores temporal edge informations. The csv should have the following columns with the header as `,src,dst,time,ext_roll` where each of the column refers to edge index (start with zero), source node index (start with zero), destination node index, time stamp, extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV should be sorted by time ascendingly.
2. `ext_full.npz`: The T-CSR representation of the temporal graph. We provide a script to generate this file from `edges.csv`. You can use the following command to use the script 
    >python gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): The torch tensor that stores the edge featrues row-wise with shape (num edges, dim edge features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
5. `labels.csv` (optional): The file contains node labels for dynamic node classification task. The csv should have the following columns with the header as `,node,time,label,ext_roll` where each of the column refers to node label index (start with zero), node index (start with zero), time stamp, node label, extrapolation roll (0 for training node labels, 1 for validation node labels, 2 for test node labels). The CSV should be sorted by time ascendingly.

## Configuration Files

We provide example configuration files for five temporal GNN methods: JODIE, DySAT, TGAT, TGN and TGAT. The configuration files for single GPU training are located at `/config/` while the multiple GPUs training configuration files are located at `/config/dist/`.

The provided configuration files are all tested to be working. If you want to use your own network architecture, please refer to `/config/readme.yml` for the meaining of each entry in the yaml configuration file. As our framework is still under development, it possible that some combination of the confiruations will lead to bug. 

## Run

Currently, our framework only supports extrapolation setting (inference for the future).

### Single GPU Link Prediction
>python train.py --data \<NameOfYourDataset> --config \<PathToConfigFile>

### MultiGPU Link Prediction
>python -m torch.distributed.launch --nproc_per_node=\<NumberOfGPUs+1> train_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs>

### Dynamic Node Classification

Currenlty, TGL only supports performing dynamic node classification using the dynamic node embedding generated in link prediction. 

For Single GPU models, directly run
>python train_node.py --data \<NameOfYourDATA> --config \<PathToConfigFile> --model \<PathToSavedModel>

For multi-GPU models, you need to first generate the dynamic node embedding
>python -m torch.distributed.launch --nproc_per_node=\<NumberOfGPUs+1> extract_node_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs> --model \<PathToSavedModel>

After generating the node embeding for multi-GPU models, run
>python train_node.py --data \<NameOfYourDATA> --model \<PathToSavedModel>

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Cite

If you use TGL in a scientific publication, we would appreciate citations to the following paper:

```
@article{zhou2022tgl,
    title={{TGL}: A General Framework for Temporal GNN Training on Billion-Scale Graphs},
    author={Zhou, Hongkuan and Zheng, Da and Nisa, Israt and Ioannidis, Vasileios and Song, Xiang and Karypis, George},
    year = {2022},
    journal = {Proc. VLDB Endow.},
    volume = {15},
    number = {8},
}
```

## License

This project is licensed under the Apache-2.0 License.
