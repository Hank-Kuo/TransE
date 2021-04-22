import os
from absl import app
from absl import flags
from typing import Tuple
from tqdm import tqdm

import model.net as net
import model.data_loader as data_loader
import utils

import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard

HITS_AT_1_SCORE = float
HITS_AT_3_SCORE = float
HITS_AT_10_SCORE = float
MRR_SCORE = float
METRICS = Tuple[HITS_AT_1_SCORE, HITS_AT_3_SCORE, HITS_AT_10_SCORE, MRR_SCORE]


def evaluate(model: torch.nn.Module, data_generator: torch_data.DataLoader, entities_count: int,
         summary_writer: tensorboard.SummaryWriter, device: torch.device, epoch_id: int, metric_suffix: str,
         ) -> METRICS:
    examples_count = 0.0
    hits_at_1 = 0.0
    hits_at_3 = 0.0
    hits_at_10 = 0.0
    mrr = 0.0

    entity_ids = torch.arange(end=entities_count, device=device).unsqueeze(0)
    
    for head, relation, tail in data_generator:
        current_batch_size = head.size()[0]

        head, relation, tail = head.to(device), relation.to(device), tail.to(device)
        all_entities = entity_ids.repeat(current_batch_size, 1)
        heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

        # Check all possible tails
        triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        tails_predictions = model.predict(triplets).reshape(current_batch_size, -1)
        # Check all possible heads
        triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
        heads_predictions = model.predict(triplets).reshape(current_batch_size, -1)

        # Concat predictions
        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))

        hits_at_1 += net.metric['hit_at_k'](predictions, ground_truth_entity_id, device=device, k=1)
        hits_at_3 += net.metric['hit_at_k'](predictions, ground_truth_entity_id, device=device, k=3)
        hits_at_10 += net.metric['hit_at_k'](predictions, ground_truth_entity_id, device=device, k=10)
        mrr += net.metric['mrr'](predictions, ground_truth_entity_id)
        
        examples_count += predictions.size()[0] # dataset.size * 2 

    hits_at_1_score = hits_at_1 / examples_count * 100
    hits_at_3_score = hits_at_3 / examples_count * 100
    hits_at_10_score = hits_at_10 / examples_count * 100
    mrr_score = mrr / examples_count
    
    # summary_writer.add_scalar('Metrics/Hits_1/' + metric_suffix, hits_at_1_score, global_step=epoch_id)
    # summary_writer.add_scalar('Metrics/Hits_3/' + metric_suffix, hits_at_3_score, global_step=epoch_id)
    # summary_writer.add_scalar('Metrics/Hits_10/' + metric_suffix, hits_at_10_score, global_step=epoch_id)
    # summary_writer.add_scalar('Metrics/MRR/' + metric_suffix, mrr_score, global_step=epoch_id)

    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score


def main(_):
    FLAGS = flags.FLAGS
    flags.DEFINE_integer("seed", default=1234, help="Seed value.")
    flags.DEFINE_string("dataset_path", default="./data", help="Path to dataset.")
    flags.DEFINE_string("checkpoint_path", default="./experiments/checkpoint", help="Path to model checkpoint (by default train from scratch).")
    flags.DEFINE_string("tensorboard_log_dir", default="./experiments/log", help="Path for tensorboard log directory.")
    flags.DEFINE_string("model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")

    # torch setting
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train/train.txt")
    test_path = os.path.join(path, "test/test.txt")
    params_path = os.path.join(FLAGS.model_dir, 'params.json')
    checkpoint_path = os.path.join(FLAGS.checkpoint_path, "checkpoint.tar")

    entity2id, relation2id = data_loader.create_mappings(train_path)

    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # dataset
    test_set = data_loader.FB15KDataset(test_path, entity2id, relation2id)
    test_generator = torch_data.DataLoader(test_set, batch_size=params.validation_batch_size)

    # model
    model = net.Net(entity_count=len(entity2id), relation_count=len(relation2id), dim=params.embedding_dim,
                                    margin=params.margin,
                                    device=params.device, norm=params.norm)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
    summary_writer = tensorboard.SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
    
    # Testing the best checkpoint on test dataset
    utils.load_checkpoint(checkpoint_path, model, optimizer)
    best_model = model.to(params.device)
    best_model.eval()
    scores = evaluate(model=best_model, data_generator=test_generator, entities_count=len(entity2id), device=params.device,
                  summary_writer=summary_writer, epoch_id=1, metric_suffix="test")
    print("Test scores: \n hit%1: {} \n hit%3: {} \nhit%10: {} \n mrr: {}".format(scores[0], scores[1], scores[2], scores[3]))


if __name__ == '__main__':
    app.run(main)
