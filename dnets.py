""" quantized classifier with given model -- main script
"""

import argparse
from datetime import datetime
import logging
import os
import shutil
import sys
import numpy as np

import methods.ref as ref
import methods.pgdsimplex as sim
import utils.utils as util

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torchvision import datasets, transforms

RANDOM_SEED = 123456
INPUT_CHANNELS = 1
IM_SIZE = 28
INPUT_DIM = 3
HIDDEN_DIM = 2
OUTPUT_DIM = 2
DATASET_SIZE = 10
NUM_ITERS = 1000
NUM_EPOCHS = 0
BATCH_SIZE = 100
LEARNING_RATE = 0.0001
SAVE_DIR = './out'
GPU_ID = '0'
OPTIMIZER = 'SGD'       # ADAM
MOMENTUM = 0.95
WEIGHT_DECAY = 0.0
QUANT_LEVELS = 2        # 2: {-1, 1}, 3: {-1, 0, 1}
DATASET = 'MNIST'       # CIFAR10, CIFAR100, TINYIMAGENET200
ARCHITECTURE = 'MLP'    # LENET300, LENET5, VGG16, RESNET18
ROUNDING = 'ARGMAX'     
EVAL_SET = 'TEST'       # TRAIN
VAL_SET = 'TRAIN'       # TEST
LOSS_FUNCTION = 'CROSSENTROPY'   # HINGE
METHOD = 'PMF'          # REF, PGD, PICM
BETA_SCALE = 1.2
LR_SCALE = 1.
LR_INTERVAL = 100
BETA_INTERVAL = 100
LR_DECAY = 'STEP'       # EXP, MSTEP
LOG_INTERVAL = 100
PR_INTERVAL = 100
SAVE_NAME = ''          # best model file
DATA_PATH = './Datasets'
PRETRAINED_MODEL = ''   # pre-trained model file
EVAL = ''               # trained model to evaluate
RESUME = ''             # checkpoint file to resume

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for binary nets.")
    parser.add_argument("--input-channels", type=int, default=INPUT_CHANNELS, help="Input channels.")
    parser.add_argument("--im-size", type=int, default=IM_SIZE, help="Image size.")
    parser.add_argument("--input-dim", type=int, default=INPUT_DIM, help="Input dimension.")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Hidden layer size.")
    parser.add_argument("--output-dim", type=int, default=OUTPUT_DIM, help="Output dimension.")
    parser.add_argument("--dataset-size", type=int, default=DATASET_SIZE, help="No of datapoints.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--num-iters", type=int, default=NUM_ITERS, help="Number of iterations.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="Number of epochs.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="Output directory.")
    parser.add_argument("--gpu-id", type=str, default=GPU_ID, help="Cuda visible device.")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, 
            help="Type of optimizer [SGD, ADAM].")
    parser.add_argument("--momentum", type=float, default=MOMENTUM, help="Momentum.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--quant-levels", type=int, default=QUANT_LEVELS, 
            help="Quantization levels: {2: {-1, 1}, 3: {-1, 0, 1}}.")
    parser.add_argument("--dataset", type=str, default=DATASET, 
            help="Type of architecture [MNIST, CIFAR10, CIFAR100, TINYIMAGENET200].")
    parser.add_argument("--architecture", type=str, default=ARCHITECTURE, 
            help="Type of architecture [MLP, LENET300, LENET5, VGG16, RESNET18].")
    parser.add_argument("--rounding", type=str, default=ROUNDING, help="Type of rounding [ARGMAX].")
    parser.add_argument("--eval-set", type=str, default=EVAL_SET, help="Dataset to evaluate [TEST, TRAIN].")
    parser.add_argument("--val-set", type=str, default=VAL_SET, help="Dataset to validate [TEST, TRAIN].")
    parser.add_argument("--loss-function", type=str, default=LOSS_FUNCTION, 
            help="Loss function [HINGE, CROSSENTROPY].")
    parser.add_argument("--nesterov", action="store_true", help="Flag to use Nesterov momentum.")
    parser.add_argument("--method", type=str, default=METHOD, 
            help="Method to run [REF, PMF, PGD, PICM].")
    parser.add_argument("--beta-scale", type=float, default=BETA_SCALE, help="Scale to multiply beta.")
    parser.add_argument("--lr-scale", type=float, default=LR_SCALE, help="Scale to multiply learning rate.")
    parser.add_argument("--lr-interval", type=str, default=LR_INTERVAL, 
            help="No of iterations before changing lr.")
    parser.add_argument("--beta-interval", type=int, default=BETA_INTERVAL, 
            help="No of iterations before changing beta.")
    parser.add_argument("--lr-decay", type=str, default=LR_DECAY, 
            help="LR decay type [STEP, EXP, MSTEP].")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL, 
            help="No of iterations before printing loss.")
    parser.add_argument("--pr-interval", type=int, default=PR_INTERVAL, 
            help="No of iterations before projection to the simplex.")
    parser.add_argument("--save-name", type=str, default=SAVE_NAME, help="Name to save the best model.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL, help="Pretrained model to load weights.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to store datasets.")
    parser.add_argument("--eval", type=str, default=EVAL, help="Model file to evaluate.")
    parser.add_argument("--resume", type=str, default=RESUME, help="Checkpoint file to resume training.")
    return parser.parse_args()

def seed_torch():
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_fn(worker_id):
   random.seed(RANDOM_SEED + worker_id)
   np.random.seed(RANDOM_SEED + worker_id)

def main():
    # Get the CL arguments
    args = get_arguments()

    # pytorch setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    seed_torch()    # set random seed for all of torch
    device = torch.device("cuda" if use_cuda else "cpu") 
    
    # setup results directory and logging
    if args.save_dir == SAVE_DIR:
        args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method, 
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if args.eval != EVAL:
        args.save_dir = os.path.join(args.save_dir, 'eval')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_name = os.path.join(args.save_dir, 'best_model.pth.tar')
    if args.resume:
        folder, name = os.path.split(args.resume)
        shutil.copyfile(os.path.join(folder, 'best_model.pth.tar'), args.save_name)

    util.setup_logging(os.path.join(args.save_dir, 'log.txt'))
    results_file = os.path.join(args.save_dir, 'results.%s')
    results = util.ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("Saving to %s", args.save_dir)

    # load data
    args.data_path = os.path.join(args.data_path, args.dataset)
    if args.dataset == 'MNIST':
        args.input_channels = 1
        args.im_size = 28
        args.input_dim = 28*28*1
        args.output_dim = 10

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)

        if args.val_set == 'TRAIN':
            train_frac = 5./6
        else:
            train_frac = 1
        num_train = len(train_set)
        indices = list(range(num_train))
        args.dataset_size = int(np.floor(train_frac * num_train))

    elif 'CIFAR' in args.dataset:
        args.input_channels = 3
        args.im_size = 32
        args.input_dim = 32*32*3
        args.output_dim = 10
        if args.dataset == 'CIFAR100':
            args.output_dim = 100

        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        transform_train=transforms.Compose([
            transforms.RandomCrop(args.im_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset == 'CIFAR10':
            train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
        elif args.dataset == 'CIFAR100':
            train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
        else:
            print 'Dataset type "{0}" not recognized, exiting ...'.format(args.dataset)
            exit()

        if args.val_set == 'TRAIN':
            train_frac = 0.9
        else:
            train_frac = 1
        num_train = len(train_set)
        indices = list(range(num_train))
        args.dataset_size = int(np.floor(train_frac * num_train))

    elif args.dataset == 'TINYIMAGENET200':
        args.input_channels = 3
        args.output_dim = 200
        args.im_size =64
        args.input_dim = 64*64*3

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        transform_train=transforms.Compose([
            transforms.RandomResizedCrop(args.im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(args.im_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        traindir = os.path.join(args.data_path, 'train')
        util.create_val_folder(args)

        valdir = os.path.join(args.data_path, 'val/images')
        train_set = datasets.ImageFolder(traindir, transform=transform_train)
        val_set = datasets.ImageFolder(valdir, transform=transform_test)
        test_set = val_set  # no test label available!
        args.dataset_size = len(train_set)

    else:
        print 'Dataset type "{0}" not recognized, exiting ...'.format(args.dataset)
        exit()

    if args.num_epochs == 0:
        args.num_epochs = int(np.ceil(float(args.num_iters) * args.batch_size / float(args.dataset_size)))

    if args.dataset == 'TINYIMAGENET200':
        train_sampler = None
        val_sampler = None
    else:
        train_idx, val_idx = indices[:args.dataset_size], indices[args.dataset_size:]
        train_sampler = du.sampler.SubsetRandomSampler(train_idx)
        val_sampler = du.sampler.SubsetRandomSampler(val_idx)
        val_set = train_set

    train_loader = du.DataLoader(train_set, sampler=train_sampler, shuffle=(train_sampler is None),
            batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)
    val_loader = du.DataLoader(val_set, sampler=val_sampler, batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)
    test_loader = du.DataLoader(test_set, batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)

    if args.val_set == 'TEST':
        val_loader = test_loader

    # lr-decay
    if args.lr_decay != 'MSTEP':
        args.lr_interval = int(args.lr_interval)
    # for MSTEP, lr_interval is a comma separated list!
    
    print(args)
    logging.debug("Run arguments: %s", args)
    
    # loss-function
    if args.loss_function == 'HINGE':
        criterion = nn.MultiLabelMarginLoss()
    elif args.loss_function == 'CROSSENTROPY':
        criterion = nn.CrossEntropyLoss()
    else:
        print 'Loss type "{0}" not recognized, exiting ...'.format(args.loss_function)
        exit()

    if args.method == 'REF':
        ref.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results)
    elif args.method == 'PMF':
        args.projection = 'SOFTMAX'
        sim.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results)
    elif args.method == 'PGD':
        args.projection = 'EUCLIDEAN'
        sim.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results)
    elif args.method == 'PICM':
        args.projection = 'ARGMAX'
        sim.setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results)
    else:
        print 'Method "{0}" not recognized, exiting ...'.format(args.method)
        exit()

if __name__ == '__main__':
    main()
