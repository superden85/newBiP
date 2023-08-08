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
        model, device, train_loader, criterion, optimizer_list, epoch, args, dummy_model
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

    optimizer, mask_optimizer = optimizer_list

    model.train()
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
            optimizer.step()

            acc1, acc5 = accuracy(output, val_targets, topk=(1, 5))
            losses.update(loss.item(), val_images.size(0))
            top1.update(acc1[0], val_images.size(0))
            top5.update(acc5[0], val_images.size(0))

            """ switch_to_bilevel(model)
            optimizer.zero_grad()
            output = model(train_images)
            loss = criterion(output, train_targets)
            loss.backward() """

            def grad2vec(parameters):
                grad_vec = []
                for param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)

            """ param_grad_vec = grad2vec(model.parameters())

            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images), train_targets)

            loss_mask.backward()

            mask_grad_vec = grad2vec(model.parameters())

            implicit_gradient = -args.lr2 * mask_grad_vec * param_grad_vec """
            
            #calculating the first part
            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images), train_targets)

            loss_mask.backward()

            #first_part = grad2vec(model.parameters())

            #calculating the second part with the dummy model
            switch_to_finetune(dummy_model)

            #the parameters of the dummy model should be set to m * theta of the model
            score_list = []
            param_list = []
            for (name, vec) in model.named_modules():
                #retrieve the mask
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        score_list.append(attr.view(-1))
                #retrieve the parameters
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            param_list.append(attr.view(-1))

            score_list = torch.cat(score_list)
            param_list = torch.cat(param_list)

            pointer = 0
            for (name, vec) in dummy_model.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            numel = attr.numel()
                            vec.w = score_list[pointer: pointer + numel].view_as(attr) * param_list[pointer: pointer + numel].view_as(attr)
                            pointer += numel
            
            with torch.no_grad():
                for param in dummy_model.parameters():
                    param.grad = None
            
            z_loss = criterion(dummy_model(train_images), train_targets)

            z_loss.backward()

            grad_z_list = []
            for (name, vec) in dummy_model.named_modules():
                if not isinstance(vec, (nn.BatchNorm2d, nn.BatchNorm2d)):
                    if hasattr(vec, "weight"):
                        attr = getattr(vec, "weight")
                        if attr is not None:
                            grad_z_list.append(attr.grad.view(-1))
            
            grad_z_list = torch.cat(grad_z_list)

            implicit_gradient = -args.lr2 * score_list * grad_z_list ** 2 
            
            def append_grad_to_vec(vec, parameters):

                if not isinstance(vec, torch.Tensor):
                    raise TypeError('expected torch.Tensor, but got: {}'
                                    .format(torch.typename(vec)))

                pointer = 0
                for param in parameters:
                    num_param = param.numel()

                    param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)

                    pointer += num_param

            pointer = 0
            for param in model.parameters():
                numel = param.numel()
                if param.requires_grad:
                    param.grad.copy_(param.grad + implicit_gradient[pointer: pointer + numel].view_as(param).data)
                pointer += numel

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
