import argparse
import json
import logging
import os
import sys

#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import pickle
from torchvision import datasets, transforms
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def generate_task_subsets(num_tasks, num_bits, task_sizes, random_state = 0):
    np.random.seed(random_state)
    task_subsets = np.zeros((num_tasks, num_bits))
    for i in range(num_tasks):
        task_subsets[i][np.random.choice(num_bits, size = task_sizes[i], replace = False)] = 1
    return task_subsets

def create_dataset(num_tasks, num_bits, task_freq, task_subsets, num_samples, num_batches = 1, random_state = 0):
    np.random.seed(random_state)
    # generate task-subsets
    # task_subsets = np.zeros((num_tasks, num_bits))
    # for i in range(num_tasks):
    #   task_subsets[i][np.random.choice(num_bits, size = task_sizes[i], replace = False)] = 1
    # print("task subsets")
    # print(task_subsets)
    all_tasks = np.arange(num_tasks)

    for i in range(num_batches):
        # sample tasks
        tasks = np.random.choice(all_tasks, size = num_samples, p = task_freq / task_freq.sum())

        # generate control bits with respect to task frequencies
        control_bits = np.eye(num_tasks)[tasks]
        # print("control bit shapes")
        # print(control_bits)
        # print(control_bits.shape)

        # generate task strings
        task_bits = np.random.randint(2, size = (num_samples, num_bits))
        # print("task bit shapes")
        # print(task_bits)
        # print(task_bits.shape)

        # generate output strings
        task_masks = task_subsets[tasks]
        # print("task mask shapes")
        # print(task_masks)
        # print(task_masks.shape)

        out = np.sum(task_bits * task_masks, axis = 1) % 2
        input = np.concatenate([control_bits, task_bits], axis = 1)
        # print(input.shape)

        yield input, out

def eval_subtasks(model, task_subsets, loss_func, samples_per_task = 50):
    n_tasks = task_subsets.shape[0]
    n = task_subsets.shape[1]

    task_bits = np.random.randint(2, size = (samples_per_task * n_tasks, n))
    output_bits = np.zeros(samples_per_task * n_tasks, dtype = int)
    # create outputs for every subtask
    for i in range(n_tasks):
        input_slice = task_bits[i * samples_per_task : (i + 1) * samples_per_task]
        outputs = np.sum(input_slice * task_subsets[i], axis = 1) % 2
        output_bits[i * samples_per_task:(i + 1) * samples_per_task] = outputs

    input_bits = np.eye(n_tasks)[np.repeat(np.arange(n_tasks), samples_per_task)]
    # print(input_bits.shape)

    total_input = np.concatenate([input_bits, task_bits], axis = 1)
    # calculate the loss for each (in bits)
    mod_outputs = model(torch.from_numpy(total_input).float().cuda())
    loss = loss_func(mod_outputs, torch.from_numpy(output_bits).long().cuda())
    # print(loss.shape)
    # return loss in bits for every subtask
    loss = loss.cpu()

    losses = loss.reshape(-1, samples_per_task).mean(axis = 1) / np.log(2)
    # print(losses.shape)
    return losses


def train(args):
    # is_distributed = len(args.hosts) > 1 and args.backend is not None
    # logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    logger.debug(f"use_cuda: {use_cuda}")
    logger.debug(args.model_dir)
    
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    # test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)
    
    batches = 10000
    num_tasks = 500
    n = 100
    alpha = 0.4

    mlp = nn.Sequential(
        nn.Linear(num_tasks + n, int(args.hidden_size)),
        nn.ReLU(),
        nn.Linear(int(args.hidden_size), 2)
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr = 1e-3)
    loss_func = nn.CrossEntropyLoss(reduction = "mean")
    losses = []
    big_losses = []

    # power law scaling with uniform task hardness
    power_law = (np.arange(num_tasks) + 1) ** (-alpha - 1)
    power_law = power_law / power_law.sum()
    task_subsets = generate_task_subsets(num_tasks, n, np.ones(num_tasks, int) * 3)
    train_data = create_dataset(500, 100, power_law, task_subsets, 20000, batches)

    model = mlp.to(device)
    
    for i, (x, y) in enumerate(train_data):
        mlp.train()
        mlp.cuda()
        optimizer.zero_grad()

        x, y = torch.from_numpy(x).float().cuda(), torch.from_numpy(y).long().cuda()
        outputs = mlp(x)
        loss = loss_func(outputs, y)
        loss.backward()
        losses.append(loss.item())
        logger.info(
            f"Batch {i}: Loss {loss.item():.6f}"
        )
        optimizer.step()

        if i % 5 == 0:
            mlp.eval()
            with torch.no_grad():
                big_losses.append(eval_subtasks(mlp, task_subsets, nn.CrossEntropyLoss(reduction = 'none'), 100))

    
    flag_id = f"{args.hidden_size}-{args.replica_id}-{batches}"

    save_model(model, args.model_dir, flag_id)
    
    path = os.path.join(args.model_dir, f"losses-{flag_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(losses, f)
    path = os.path.join(args.model_dir, f"big_losses-{flag_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(big_losses, f)


def save_model(model, model_dir, flag_id):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, f"model-{flag_id}.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--hidden-size",
        type=int,
        default = 10,
        metavar="N",
    )
    parser.add_argument(
        "--replica-id",
        type=int,
        default=int(np.random.randint(10000000)),
        metavar="N"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
