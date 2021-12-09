import os
import argparse
from tqdm import tqdm
import logging
import json

import model.net as net
import model.data_loader as data_loader
import utils
from evaluate import evaluate

import torch
import numpy as np
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard

########## split
kg_path = './data/kg_final.txt'
train_path = './data/train/train.txt'
valid_path = './data/valid/valid.txt'
train_ratio = 0.9
valid_ratio = 0.1

kg_np = np.loadtxt(kg_path, dtype=np.int32)
n_ratings = kg_np.shape[0]

train_indices = np.random.choice(n_ratings, size=int(n_ratings * train_ratio), replace=False)
left = set(range(n_ratings)) - set(train_indices)
valid_indices = np.random.choice(list(left), size=int(n_ratings * valid_ratio), replace=False)

train_data = kg_np[train_indices]
valid_data = kg_np[valid_indices]

np.savetxt(train_path, train_data,  delimiter='\t',fmt='%d')
np.savetxt(valid_path, valid_data,  delimiter='\t',fmt='%d')

checkpoint_dir = './experiments/base_model/checkpoint'
model_dir = './experiments/base_model'
train_path = "./data/train/train.txt"
validation_path = "./data/valid/valid.txt"
params_path = './experiments/base_model/params.json'

entity2id, relation2id = data_loader.create_mappings(train_path, validation_path)

# params
params = utils.Params(params_path)
params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
model = net.Net(entity_count=len(entity2id), relation_count=len(relation2id), dim=params.embedding_dim,
                                margin=params.margin,
                                device=params.device, norm=params.norm)  # type: torch.nn.Module

optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
utils.load_checkpoint(checkpoint_dir, model, optimizer)
best_model = model.to(params.device)
best_model.eval()
entity_np = best_model.state_dict()['entities_emb.weight'].detach().cpu().numpy()
relation_np = best_model.state_dict()['relations_emb.weight'].detach().cpu().numpy()

########## save dict 

entity_dict_path = './data/entity.dict'
relation_dict_path = './data/relation.dict'
with open(entity_dict_path, "w") as f:
    json.dump(entity2id, f)

with open(relation_dict_path, "w") as f:
    json.dump(relation2id, f)

##########　save weight to numpy
np.save(kg_file + './data/entity.npy', entity_np)
np.save(kg_file + '.npy', relation_np)