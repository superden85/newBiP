import time

import torch

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
)

import numpy as np


def train(
        model, device, train_loader, criterion, optimizer_list, epoch, args
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

    #print stats
    l0, l1 = 0, 0
    mini, maxi = 1000, -1000
    for (name, vec) in model.named_modules():
        if hasattr(vec, "popup_scores"):
            attr = getattr(vec, "popup_scores")
            if attr is not None:
                l0 += torch.sum(attr != 0).item()
                l1 += (torch.sum(torch.abs(attr)).item())
                mini = min(mini, torch.min(attr).item())
                maxi = max(maxi, abs(torch.max(attr).item()))
    
    print("l0 norm of mask: ", l0)
    print("l1 norm of mask: ", l1)
    print("min of mask: ", mini)
    print("max of mask: ", maxi)


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
            
            #Lower level step
            #We do 1 step of SGD on the parameters of the model

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

            #upper level step
            #we have to calculate first the implicit gradient
            
            switch_to_bilevel(model)
            optimizer.zero_grad()
            output = model(train_images)
            loss = criterion(output, train_targets)
            loss.backward()

            def grad2vec(parameters):
                grad_vec = []
                for param in parameters:
                    grad_vec.append(param.grad.view(-1).detach())
                return torch.cat(grad_vec)

            param_grad_vec = grad2vec(model.parameters())

            switch_to_prune(model)
            mask_optimizer.zero_grad()
            loss_mask = criterion(model(train_images), train_targets)

            loss_mask.backward()

            mask_grad_vec = grad2vec(model.parameters())
            implicit_gradient = -args.lr2 * mask_grad_vec * param_grad_vec            
            
            #then the outer gradient is simply:
            outer_gradient = mask_grad_vec + implicit_gradient

            #the linear minimization problem is very simple we don't need to use a solver
            #mstar is equal to 1 if c is negative, 0 otherwise

            m_star = torch.zeros_like(outer_gradient)
            m_star[outer_gradient < 0] = 1

            #we want to have a diminishing step size
            step_size = 2/(epoch * len(train_loader) + i + 2)

            #then we update the parameters

            if not isinstance(m_star, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))

            pointer = 0
            for param in model.parameters():
                num_param = param.numel()

                #update only if it is a popup score
                #i.e. if param.requires_grad = True

                if param.requires_grad:
                    param.data = ((1 - step_size) * param.data + step_size * m_star[pointer:pointer + num_param].view_as(param).data)

                pointer += num_param

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))       

            if i<=10:
                #print stats
                l0, l1 = 0, 0
                mini, maxi = 1000, -1000
                for (name, vec) in model.named_modules():
                    if hasattr(vec, "popup_scores"):
                        attr = getattr(vec, "popup_scores")
                        if attr is not None:
                            l0 += torch.sum(attr != 0).item()
                            l1 += (torch.sum(torch.abs(attr)).item())
                            mini = min(mini, torch.min(attr).item())
                            maxi = max(maxi, abs(torch.max(attr).item()))
                
                print("l0 norm of mask: ", l0)
                print("l1 norm of mask: ", l1)
                print("min of mask: ", mini)
                print("max of mask: ", maxi)    


        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
    #return the intermediate mask
    if args.exp_mode == 'prune' and args.save_masks:
        intermediate_mask = []
        for (name, vec) in model.named_modules():
            if hasattr(vec, "popup_scores"):
                attr = getattr(vec, "popup_scores")
                if attr is not None:
                    intermediate_mask.append(attr.view(-1).detach())
        return torch.cat(intermediate_mask)