import time

import torch

import torch.nn as nn

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
)


def train(
        model, device, train_loader, criterion, optimizer_list, epoch, args, **kwargs
):
    print("->->->->->->->->->-> One epoch with Natural training <-<-<-<-<-<-<-<-<-<-")
    train_loader, val_loader = train_loader

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    dummy_model = kwargs["dummy_model"]
    optimizer, mask_optimizer = optimizer_list

    model.train()
    dummy_model.train()
    end = time.time()

    for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
        train_images, train_targets = train_data_batch[0].to(device), train_data_batch[1].to(device)
        val_images, val_targets = val_data_batch[0].to(device), val_data_batch[1].to(device)
        if args.accelerate:
            switch_to_prune(model)
            output = model(train_images)
            loss = criterion(output, train_targets)
            mask_optimizer.zero_grad()
            loss.backward()
            mask_optimizer.step()

            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))

            switch_to_finetune(model)
            output = model(val_images)
            loss = criterion(output, val_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

        else:
            switch_to_finetune(model)
            output = model(val_images)
            loss = criterion(output, val_targets)

            optimizer.zero_grad()
            loss.backward()

            #patch for the rounding bug
            #set to None all the gradients for the popup scores
            for param in model.parameters():
                if not param.requires_grad:
                    param.grad = None
            
            optimizer.step()

            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

            #upper level step 
            switch_to_prune(model)
            switch_to_finetune(dummy_model)

            #the parameters of the dummy model should be set to m * theta of the model
            param_list = []
            score_list = []
            name_list = []
            current_name = None
            current_score = None
            for (name, param), (_, dummy_param) in reversed(list(zip(model.named_parameters(), dummy_model.named_parameters()))):
                dummy_param.data.copy_(param.data)
                if 'popup_scores' in name:
                    current_name = name[:-13]
                    current_score = param.data
                    score_list.append(param.data.detach())
                if name == current_name + '.weight':
                    name_list.append(name)
                    dummy_param.data.copy_(current_score * dummy_param.data)
                    param_list.append(param.data.detach())
            
            param_list.reverse()
            score_list.reverse()
        
            with torch.no_grad():
                for param in dummy_model.parameters():
                    param.grad = torch.zeros_like(param.data)
            
            #compute grad_z l(z = m * theta)
            z_loss = criterion(dummy_model(train_images), train_targets)
            z_loss.backward()

            #retrieve grad_z l(z = m * theta)
            grad_z_list = []
            
            for (name, param) in dummy_model.named_parameters():
                if name in name_list:
                    grad_z_list.append(param.grad.view(-1).detach())                       
            
            grad_z = torch.cat(grad_z_list)
            
            param = torch.cat([param.view(-1) for param in param_list])
            score = torch.cat([score.view(-1) for score in score_list])

            hypergradient = (param - args.lr2 * score * grad_z) * grad_z

            #check the mathematical relation that should hold between grad_m and grad_z

            def grad2vec(parameters):
                grad_vec = []
                for name, param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)

            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images), train_targets)

            loss_mask.backward()

            mask_grad_vec = grad2vec(model.named_parameters())

            if i <= 2:
                #print if mask_grad_vec and param * grad_z are equal
                print('theory check : ', torch.allclose(mask_grad_vec, param * grad_z, atol=1e-6))
            
            def append_grad_to_vec(vec, parameters):

                if not isinstance(vec, torch.Tensor):
                    raise TypeError('expected torch.Tensor, but got: {}'
                                    .format(torch.typename(vec)))

                pointer = 0
                for name, param in parameters:

                    if 'popup_scores' in name:
                        num_param = param.numel()

                        if param.grad is None:
                            param.grad = torch.zeros_like(param.data)
                        param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)

                        pointer += num_param
            
            append_grad_to_vec(hypergradient, model.named_parameters())

            mask_optimizer.step()

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
