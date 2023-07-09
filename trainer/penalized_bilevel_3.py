import time

import torch

from utils.eval import accuracy
from utils.general_utils import AverageMeter, ProgressMeter
from utils.model import (
    switch_to_bilevel,
    switch_to_prune,
    switch_to_finetune,
    get_epoch_data,
    get_epoch_data_2
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

    for i, (train_data_batch, val_data_batch) in enumerate(zip(train_loader, val_loader)):
        train_images, train_targets = train_data_batch[0].to(device), train_data_batch[1].to(device)
        val_images, val_targets = val_data_batch[0].to(device), val_data_batch[1].to(device)
        if args.accelerate:
            switch_to_prune(model)
            output = model(train_images)
            loss = criterion(output, train_targets)
            
            # add a regularization term, defined as the (1-exp(-alpha*mask)) T vector full of ones
            loss += args.lambd * (1 - torch.exp(-args.alpha * model.mask)).sum()

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

            loss*=args.lambd

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

            loss*=args.lambd

            # add a regularization term, defined as the m * (1 - m) T vector full of ones
            for (name, vec) in model.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        loss += (1 - args.lambd) * (attr * (1 - attr)).sum()
            
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

            loss_mask *= args.lambd

            # add a regularization term, defined as the (1-exp(-alpha*mask)) T vector full of ones
            for (name, vec) in model.named_modules():
                if hasattr(vec, "popup_scores"):
                    attr = getattr(vec, "popup_scores")
                    if attr is not None:
                        loss_mask += (1 - args.lambd) * (attr * (1 - attr)).sum()
            
            loss_mask.backward()

            mask_grad_vec = grad2vec(model.parameters())
            implicit_gradient = -args.lr2 * mask_grad_vec * param_grad_vec            
            
            #then the outer gradient is simply:
            outer_gradient = mask_grad_vec + implicit_gradient

            #the linear minimization problem is very simple we don't need to use a solver
            #here we have that 1 T m <= k * total

            m_star = torch.ones_like(outer_gradient)

            #flat_m_star and m_star access the same memory
            flat_m_star = m_star.flatten()

            #get the lowest elements of the gradient
            flat_outer_gradient = outer_gradient.flatten()
            idx = torch.argsort(flat_outer_gradient)
            j = int(args.k * m_star.numel())
            
            #set to 1 only if it the gradient is negative and we are in the top k%
            flat_m_star[idx[j:]] = 0
            flat_m_star[flat_outer_gradient >= 0] = 0

            #we want to have a diminishing step size
            step_size = 2/(epoch * len(train_loader) + i + 2)
            
            #print l1 norm of m_star
            print("l1 norm of m_star: ", torch.norm(m_star, p=1).item())
            #then we update the parameters

            if not isinstance(m_star, torch.Tensor):
                raise TypeError('expected torch.Tensor, but got: {}'
                                .format(torch.typename(vec)))

            pointer = 0
            l1_check = 0
            for param in model.parameters():
                num_param = param.numel()

                #update only if it is a popup score
                #i.e. if param.requires_grad = True

                if param.requires_grad:
                    param.data = ((1 - step_size) * param.data + step_size * m_star[pointer:pointer + num_param].view_as(param).data)
                    l1_check += torch.sum(torch.abs(param.data)).item()
                    print(l1_check)
                pointer += num_param

            print('final l1 check :', l1_check)

            output = model(train_images)
            acc1, acc5 = accuracy(output, train_targets, topk=(1, 5))  # log
            losses.update(loss.item(), train_images.size(0))
            top1.update(acc1[0], train_images.size(0))
            top5.update(acc5[0], train_images.size(0))           


        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if i == 0:
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
        
    #return data related to the mask of this epoch
    epoch_data = get_epoch_data_2(model)

    return epoch_data