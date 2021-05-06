# TransE-PyTorch
Implementation of TransE [[1]](#references) model in PyTorch.

## Table of Contents
1. [Results](#results)
    1. [Datasets](#datasets)
        1. [FB15k](#fb15k)
2. [Usage](#usage)
    1. [Training](#training)
        1. [Options](#options)
    2. [Unit tests](#unit-tests)
3. [References](#references)

## Results

### Datasets

#### FB15k

| Source/Metric  | Hits@1 (raw) | Hits@3 (raw) | Hits@10 (raw) | MRR (raw) |
| ---------------| ------------ | ------------ | ------------- | --------- |
| Paper [[1]](#references) | X | X | 34.9 | X |
| TransE-PyTorch | 11.1 | 25.33 | **46.53** | 22.29 |

 
## Usage

### Training
```bash
python3 train.py --dataset_path=<path_to_your_dataset> --model_dir=<path_to_your_model_params>
```


## References
[1] [Bordes et al., "Translating embeddings for modeling multi- relational data," in Adv. Neural Inf. Process. Syst., 2013](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
