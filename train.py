import os
import argparse
from tqdm import tqdm
import logging

import model.net as net
import model.data_loader as data_loader
import utils
from evaluate import evaluate

import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
# from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1234, help="Seed value.")
parser.add_argument("--dataset_path", default="./data", help="Path to dataset.")
parser.add_argument("--model_dir", default="./experiments/base_model", help="Path to model checkpoint (by default train from scratch).")


def main():
    args = parser.parse_args()

    # torch setting
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os setting
    path = args.dataset_path
    train_path = os.path.join(path, "train/train.txt")
    validation_path = os.path.join(path, "valid/valid.txt")
    test_path = os.path.join(path, "test/test.txt")
    params_path = os.path.join(args.model_dir, 'params.json')
    checkpoint_dir = os.path.join(args.model_dir, 'checkpoint')
    tensorboard_log_dir = os.path.join(args.model_dir, 'log')
    utils.check_dir(tensorboard_log_dir)
    
    entity2id, relation2id = data_loader.create_mappings(train_path)

    # params
    params = utils.Params(params_path)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    train_data = data_loader.load_data(train_path, reverse=False)
    valid_data = data_loader.load_data(validation_path)

    # dataset
    train_set = data_loader.FB15KDataset(train_data, entity2id, relation2id)
    train_generator = torch_data.DataLoader(train_set, batch_size=params.batch_size)
    validation_set = data_loader.FB15KDataset(valid_data, entity2id, relation2id)
    validation_generator = torch_data.DataLoader(validation_set, batch_size=params.validation_batch_size)
    #test_set = data_loader.FB15KDataset(test_path, entity2id, relation2id)
    #test_generator = torch_data.DataLoader(test_set, batch_size=params.validation_batch_size)

    # model
    model = net.Net(entity_count=len(entity2id), relation_count=len(relation2id), dim=params.embedding_dim,
                                    margin=params.margin,
                                    device=params.device, norm=params.norm)  # type: torch.nn.Module
    
    
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
    #start_epoch_id, step, best_score = utils.load_checkpoint(checkpoint_dir, model, optimizer)
    model = model.to(params.device)

    summary_writer = tensorboard.SummaryWriter(log_dir=tensorboard_log_dir)
    start_epoch_id, step, best_score = 1, 0, 0.0

    print("Training Dataset: entity: {} relation: {} triples: {}".format(len(entity2id), len(relation2id), len(train_set)))
    print("Validation Dataset: triples: {}".format(len(validation_set)))
    # print("Test Dataset: triples: {}".format(len(test_set)))
    print(model)

    # Train
    for epoch_id in range(start_epoch_id, params.epochs + 1):
        print("Epoch {}/{}".format(epoch_id, params.epochs))
        
        loss_impacting_samples_count = 0
        samples_count = 0
        model.train()

        with tqdm(total=len(train_generator)) as t:
            for local_heads, local_relations, local_tails in train_generator:
                local_heads, local_relations, local_tails = (local_heads.to(params.device), local_relations.to(params.device),
                                                            local_tails.to(params.device))

                positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)

                # Preparing negatives.
                # Generate binary tensor to replace either head or tail. 1 means replace head, 0 means replace tail.
                head_or_tail = torch.randint(high=2, size=local_heads.size(), device=params.device)
                random_entities = torch.randint(high=len(entity2id), size=local_heads.size(), device=params.device)
                broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
                broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
                negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)

                optimizer.zero_grad()

                loss, pd, nd = model(positive_triples, negative_triples)
                loss.mean().backward()

                summary_writer.add_scalar('Loss/train', loss.mean().data.cpu().numpy(), global_step=step)
                summary_writer.add_scalar('Distance/positive', pd.sum().data.cpu().numpy(), global_step=step)
                summary_writer.add_scalar('Distance/negative', nd.sum().data.cpu().numpy(), global_step=step)

                loss = loss.data.cpu()
                loss_impacting_samples_count += loss.nonzero().size()[0]
                samples_count += loss.size()[0]

                optimizer.step()
                step += 1
                
                t.set_postfix(loss = loss_impacting_samples_count / samples_count * 100)
                t.update()

            summary_writer.add_scalar('Metrics/batch_loss', loss_impacting_samples_count / samples_count * 100,
                                    global_step=epoch_id)

            # validation
            if epoch_id % params.validation_freq == 0:
                model.eval()
                _, _, hits_at_10, _ = evaluate(model=model, data_generator=validation_generator,
                                        entities_count=len(entity2id),
                                        device=params.device, summary_writer=summary_writer,
                                        epoch_id=epoch_id, metric_suffix="val")
                logging.info('Eval: hit_10: {}'.format(hits_at_10))    
                score = hits_at_10
                if score > best_score:
                    best_score = score
                    utils.save_checkpoint(checkpoint_dir, model, optimizer, epoch_id, step, best_score)
            
    '''
    # Testing the best checkpoint on test dataset
    utils.load_checkpoint(checkpoint_dir, model, optimizer)
    best_model = model.to(params.device)
    best_model.eval()
    scores = evaluate(model=best_model, data_generator=test_generator, entities_count=len(entity2id), device=params.device,
                  summary_writer=summary_writer, epoch_id=1, metric_suffix="test")
    print("Test scores: \n hit%1: {} \n hit%3: {} \nh it%10: {} \n mrr: {}".format(scores[0], scores[1], scores[2], scores[3]))

    eval_path = os.path.join(args.model_dir, 'eval.json')
    evals_params = utils.Params(eval_path)
    evals_params.hit_1 = scores[0]
    evals_params.hit_3 = scores[1]
    evals_params.hit_10 = scores[2]
    evals_params.mrr = scores[3]
    evals_params.best_score = best_score
    evals_params.save(eval_path)
    '''




if __name__ == '__main__':
    main()
