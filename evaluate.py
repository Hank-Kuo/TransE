import os
from absl import app
from absl import flags
from typing import Tuple
from tqdm import tqdm

import metric
import model.net as net
import model.data_loader as data_loader
import utils

import torch
import torch.optim as optim
from torch.utils import data as torch_data

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("validation_batch_size", default=64, help="Maximum batch size during model validation.")
flags.DEFINE_integer("vector_length", default=50, help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0, help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer("norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_string("dataset_path", default="./data", help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_string("checkpoint_path", default="./experiments/checkpoint", help="Path to model checkpoint (by default train from scratch).")

HITS_AT_1_SCORE = float
HITS_AT_3_SCORE = float
HITS_AT_10_SCORE = float
MRR_SCORE = float
METRICS = Tuple[HITS_AT_1_SCORE, HITS_AT_3_SCORE, HITS_AT_10_SCORE, MRR_SCORE]


def test(model: torch.nn.Module, data_generator: torch_data.DataLoader, entities_count: int,
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

        hits_at_1 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=1)
        hits_at_3 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=3)
        hits_at_10 += metric.hit_at_k(predictions, ground_truth_entity_id, device=device, k=10)
        mrr += metric.mrr(predictions, ground_truth_entity_id)

        examples_count += predictions.size()[0]

    hits_at_1_score = hits_at_1 / examples_count * 100
    hits_at_3_score = hits_at_3 / examples_count * 100
    hits_at_10_score = hits_at_10 / examples_count * 100
    mrr_score = mrr / examples_count * 100
    summary_writer.add_scalar('Metrics/Hits_1/' + metric_suffix, hits_at_1_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_3/' + metric_suffix, hits_at_3_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/Hits_10/' + metric_suffix, hits_at_10_score, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/MRR/' + metric_suffix, mrr_score, global_step=epoch_id)

    return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score


def main(_):
    # torch setting
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train/train.txt")
    validation_path = os.path.join(path, "valid/valid.txt")
    test_path = os.path.join(path, "test/test.txt")
    path = FLAGS.checkpoint_path
    checkpoint_path = os.path.join(path, "checkpoint.tar")

    entity2id, relation2id = data_loader.create_mappings(train_path)

    # params
    vector_length = FLAGS.vector_length
    margin = FLAGS.margin
    norm = FLAGS.norm
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    # dataset
    test_set = data_loader.FB15KDataset(test_path, entity2id, relation2id)
    test_generator = torch_data.DataLoader(test_set, batch_size=FLAGS.validation_batch_size)

    # model
    model = net.Net(entity_count=len(entity2id), relation_count=len(relation2id), dim=vector_length,
                                    margin=margin,
                                    device=device, norm=norm)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Testing the best checkpoint on test dataset
    utils.load_checkpoint(checkpoint_path, model, optimizer)
    best_model = model.to(device)
    best_model.eval()
    scores = test(model=best_model, data_generator=test_generator, entities_count=len(entity2id), device=device,
                  summary_writer=summary_writer, epoch_id=1, metric_suffix="test")
    print("Test scores: \n hit%1: {} \n hit%3: {} \nh it%10: {} \n mrr: {}".format(scores[0], scores[1], scores[2], scores[3]))


if __name__ == '__main__':
    app.run(main)
