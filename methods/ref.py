""" continuous classifier: 32bit floating point network -- REF
"""

import logging
import os
import numpy as np
from timeit import default_timer as timer

import models
import utils.utils as util

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BEST_ACC = 0.0

def train_step(model, device, data, target, optimizer, criterion):
    data, target = data.to(device, torch.float), target.to(device, torch.long)
    data.requires_grad_(True)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)   
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(args, model, device, loader, training=False):
    global BEST_ACC
    model.eval()
    correct1 = 0
    correct5 = 0
    tsize = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, torch.float), target.to(device, torch.long)
            output = model(data)
            # topk accuracy
            c1, c5 = util.accuracy(output.data, target, topk=(1,5))
            correct1 += c1
            correct5 += c5
            tsize += target.size(0)

    if training:
        model.train()   

    acc1 = 100. * correct1 / tsize
    acc5 = 100. * correct5 / tsize
    if (acc1 > BEST_ACC):
        BEST_ACC = acc1.item()
        if training:    # storing the continuous weights of the best model, done separately from checkpoint!
            util.save_model({'state_dict': model.state_dict(), 'best_acc1': BEST_ACC}, args.save_name) 
    
    return acc1.item(), acc5.item()

def init_weights(model, xavier=False):
    for p in model.parameters():
        if xavier and len(p.size()) >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.normal_(p, std=0.1)

def set_weights(device, params, w1, w2):
    for i, p in enumerate(params):
        if i == 0:
            p.data.copy_(w1)
        elif i == 1:
            p.data.copy_(w2)
        else:
            assert 0

def setup_and_run(args, criterion, device, train_loader, test_loader, val_loader, logging, results):
    global BEST_ACC
    print('\n#### Running REF ####')

    # architecture
    if args.architecture == 'MLP':
        model = models.MLP(args.input_dim, args.hidden_dim, args.output_dim).to(device)
    elif args.architecture == 'LENET300':
        model = models.LeNet300(args.input_dim, args.output_dim).to(device)
    elif args.architecture == 'LENET5':
        model = models.LeNet5(args.input_channels, args.im_size, args.output_dim).to(device)
    elif 'VGG' in args.architecture:
        assert(args.architecture == 'VGG11' or args.architecture == 'VGG13' or args.architecture == 'VGG16' 
                or args.architecture == 'VGG19')
        model = models.VGG(args.architecture, args.input_channels, args.im_size, args.output_dim).to(device)
    elif args.architecture == 'RESNET18':
        model = models.ResNet18(args.input_channels, args.im_size, args.output_dim).to(device)
    elif args.architecture == 'RESNET34':
        model = models.ResNet34(args.input_channels, args.im_size, args.output_dim).to(device)
    elif args.architecture == 'RESNET50':
        model = models.ResNet50(args.input_channels, args.im_size, args.output_dim).to(device)
    elif args.architecture == 'RESNET101':
        model = models.ResNet101(args.input_channels, args.im_size, args.output_dim).to(device)
    elif args.architecture == 'RESNET152':
        model = models.ResNet152(args.input_channels, args.im_size, args.output_dim).to(device)
    else:
        print 'Architecture type "{0}" not recognized, exiting ...'.format(args.architecture)
        exit()

    # optimizer
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
    else:
        print 'Optimizer type "{0}" not recognized, exiting ...'.format(args.optimizer)
        exit()

    # lr-scheduler
    if args.lr_decay == 'STEP':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_scale)
    elif args.lr_decay == 'EXP':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scale)
    elif args.lr_decay == 'MSTEP':
        x = args.lr_interval.split(',')
        lri = [int(v) for v in x]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lri, gamma=args.lr_scale)
        args.lr_interval = 1    # lr_interval handled in scheduler!
    else:
        print 'LR decay type "{0}" not recognized, exiting ...'.format(args.lr_decay)
        exit()

    init_weights(model, xavier=True)
    logging.info(model)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("Number of parameters: %d", num_parameters)

    start_epoch = -1
    iters = 0   # total no of iterations, used to do many things!
    # optionally resume from a checkpoint
    if args.eval:
        logging.info('Loading checkpoint file "{0}" for evaluation'.format(args.eval))
        if not os.path.isfile(args.eval):
            print 'Checkpoint file "{0}" for evaluation not recognized, exiting ...'.format(args.eval)
            exit()
        checkpoint = torch.load(args.eval)
        model.load_state_dict(checkpoint['state_dict'])

    elif args.resume:
        checkpoint_file = args.resume
        logging.info('Loading checkpoint file "{0}" to resume'.format(args.resume))
        if not os.path.isfile(checkpoint_file):
            print 'Checkpoint file "{0}" not recognized, exiting ...'.format(checkpoint_file)
            exit()
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        assert(args.architecture == checkpoint['architecture'])
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        BEST_ACC = checkpoint['best_acc1']
        iters = checkpoint['iters']
        logging.debug('best_acc1: {0}, iters: {1}'.format(BEST_ACC, iters))

    if not args.eval:
        logging.info('Training...')
        model.train()
        st = timer()                
    
        for e in range(start_epoch + 1, args.num_epochs):
            for i, (data, target) in enumerate(train_loader):
                l = train_step(model, device, data, target, optimizer, criterion)
                if i % args.log_interval == 0:
                    acc1, acc5 = evaluate(args, model, device, val_loader, training=True)
                    logging.info('Epoch: {0},\t Iter: {1},\t Loss: {loss:.5f},\t Val-Acc1: {acc1:.2f} '
                                 '(Best: {best:.2f}),\t Val-Acc5: {acc5:.2f}'.format(e, i, 
                                     loss=l, acc1=acc1, best=BEST_ACC, acc5=acc5))
    
                if iters % args.lr_interval == 0:
                    lr = args.learning_rate
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']                        
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        if lr != param_group['lr']:
                            logging.info('lr: {0}'.format(param_group['lr']))   # print if changed
                iters += 1

            # save checkpoint
            acc1, acc5 = evaluate(args, model, device, val_loader, training=True)
            results.add(epoch=e, iteration=i, train_loss=l, val_acc1=acc1, best_val_acc1=BEST_ACC)
            util.save_checkpoint({'epoch': e, 'architecture': args.architecture, 'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 
                'best_acc1': BEST_ACC, 'iters': iters}, is_best=False, path=args.save_dir)
            results.save()

        et = timer()
        logging.info('Elapsed time: {0} seconds'.format(et - st))
    
        acc1, acc5 = evaluate(args, model, device, val_loader, training=True)
        logging.info('End of training, Val-Acc: {acc1:.2f} (Best: {best:.2f}), Val-Acc5: {acc5:.2f}'.format(acc1=acc1, 
            best=BEST_ACC, acc5=acc5))
        # load saved model
        saved_model = torch.load(args.save_name)
        model.load_state_dict(saved_model['state_dict'])
    # end of training

    # eval-set
    if args.eval_set != 'TRAIN' and args.eval_set != 'TEST':
        print 'Evaluation set "{0}" not recognized ...'.format(args.eval_set)

    logging.info('Evaluating REF on the {0} set...'.format(args.eval_set))
    st = timer()       
    if args.eval_set == 'TRAIN':
        acc1, acc5 = evaluate(args, model, device, train_loader)
    else: 
        acc1, acc5 = evaluate(args, model, device, test_loader)
    et = timer()
    logging.info('Accuracy: top-1: {acc1:.2f}, top-5: {acc5:.2f}%'.format(acc1=acc1, acc5=acc5))
    logging.info('Elapsed time: {0} seconds'.format(et - st))
